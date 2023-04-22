# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for RetMol. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

'''
implement the retrieval generation (iteration + select) mechanism here
'''
import os
import sys
import pickle
from typing import List
from joblib import delayed
import joblib
from time import time
import pandas as pd
import numpy as np
import random
import pytorch_lightning as pl

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

import guacamol.guacamol_baseline.crossover as co
import guacamol.guacamol_baseline.mutate as mu
from guacamol.guacamol.utils.chemistry import canonicalize_list
from guacamol.guacamol.utils.data import remove_duplicates
from guacamol.guacamol.goal_directed_generator import GoalDirectedGenerator

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, project_home + '/inference')
from inference import MegaMolBART

sys.path.insert(1, project_home + '/MolBART')
from csv_data_retrieval import collate_fn_inference


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    assert (all([s > 0 for s in population_scores]))
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))


def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print('bad smiles')
    return new_population


class RetrievalGAGenerator(GoalDirectedGenerator):
    """
    Mock generator that returns pre-defined molecules
    """

    def __init__(self, project_home,
                 n_retrievals, n_repeat, n_trials, n_top_gens, ret_mode,
                 batch_size, model_path, model_ckpt_itr=50000, max_mol_len=200):

        ## generation configs
        self.n_retrievals = n_retrievals
        self.n_repeat = n_repeat
        self.n_trials = n_trials
        self.n_top_gens = n_top_gens
        self.ret_mode = ret_mode
        self.batch_size = batch_size
        self.pool = joblib.Parallel(n_jobs=10)
        self.mutation_rate = 0.1

        # model stuff
        self.wf = MegaMolBART(model_path=model_path, model_ckpt_itr=model_ckpt_itr, decoder_max_seq_len=max_mol_len)

        # input molecules
        with open(os.path.join(project_home, 'data/guacamol/guacamol_v1_train.smiles')) as f:
            data = f.readlines()
        self.input_dataset = [x.strip() for x in data]

        # retrieval stuff
        data_folder = 'data/guacamol/retrieval_database/ret_size_1000'
        self.benchmark_retrieval_database = {
            'Aripiprazole similarity': os.path.join(project_home, data_folder, 'similarity_aripiprazole.csv'),
            'Albuterol similarity': os.path.join(project_home, data_folder, 'similarity_albuterol.csv'),
            'Mestranol similarity': os.path.join(project_home, data_folder, 'similarity_mestranol.csv'),
            'C11H24': os.path.join(project_home, data_folder, 'isomers_c11h24.csv'),
            'C9H10N2O2PF2Cl': os.path.join(project_home, data_folder, 'isomers_c9h102o2pf2cl.csv'),
            'Median molecules 1': os.path.join(project_home, data_folder, 'median_camphor_menthol.csv'),
            'Median molecules 2': os.path.join(project_home, data_folder, 'median_tadalafil_sildenafil.csv'),
            'Osimertinib MPO': os.path.join(project_home, data_folder, 'hard_osimertinib.csv'),
            'Fexofenadine MPO': os.path.join(project_home, data_folder, 'hard_fexofenadine.csv'),
            'Ranolazine MPO': os.path.join(project_home, data_folder, 'ranolazine_mpo.csv'),
            'Perindopril MPO': os.path.join(project_home, data_folder, 'perindopril_rings.csv'),
            'Amlodipine MPO': os.path.join(project_home, data_folder, 'amlodipine_rings.csv'),
            'Sitagliptin MPO': os.path.join(project_home, data_folder, 'sitagliptin_replacement.csv'),
            'Zaleplon MPO': os.path.join(project_home, data_folder, 'zaleplon_with_other_formula.csv'),
            'Valsartan SMARTS': os.path.join(project_home, data_folder, 'valsartan_smarts.csv'),
            'Deco Hop': os.path.join(project_home, data_folder, 'decoration_hop.csv'),
            'Scaffold Hop': os.path.join(project_home, data_folder, 'scaffold_hop.csv'),
        }
        self.benchmark_retrieval_threshold = {
            'Aripiprazole similarity': 1,
            'Albuterol similarity': 1,
            'Mestranol similarity': 1,
            'C11H24': 1,
            'C9H10N2O2PF2Cl': 1,
            'Median molecules 1': 0.4,
            'Median molecules 2': 0.4,
            'Osimertinib MPO': 0.95,
            'Fexofenadine MPO': 0.99,
            'Ranolazine MPO': 0.92,
            'Perindopril MPO': 0.8,
            'Amlodipine MPO': 0.89,
            'Sitagliptin MPO': 0.89,
            'Zaleplon MPO': 0.75,
            'Valsartan SMARTS': 0.99,
            'Deco Hop': 1,
            'Scaffold Hop': 1,
        }
        self.benchmark_retrieval_database_embs = {
            'Aripiprazole similarity': os.path.join(project_home, data_folder,
                                                    'similarity_aripiprazole.csv_smiles_megamolbart_embs.npy'),
            'Albuterol similarity': os.path.join(project_home, data_folder,
                                                 'similarity_albuterol.csv_smiles_megamolbart_embs.npy'),
            'Mestranol similarity': os.path.join(project_home, data_folder,
                                                 'similarity_mestranol.csv_smiles_megamolbart_embs.npy'),
            'C11H24': os.path.join(project_home, data_folder,
                                   'isomers_c11h24.csv_smiles_megamolbart_embs.npy'),
            'C9H10N2O2PF2Cl': os.path.join(project_home, data_folder,
                                           'isomers_c9h102o2pf2cl.csv_smiles_megamolbart_embs.npy'),
            'Median molecules 1': os.path.join(project_home, data_folder,
                                               'median_camphor_menthol.csv_smiles_megamolbart_embs.npy'),
            'Median molecules 2': os.path.join(project_home, data_folder,
                                               'median_tadalafil_sildenafil.csv_smiles_megamolbart_embs.npy'),
            'Osimertinib MPO': os.path.join(project_home, data_folder,
                                            'hard_osimertinib.csv_smiles_megamolbart_embs.npy'),
            'Fexofenadine MPO': os.path.join(project_home, data_folder,
                                             'hard_fexofenadine.csv_smiles_megamolbart_embs.npy'),
            'Ranolazine MPO': os.path.join(project_home, data_folder,
                                           'ranolazine_mpo.csv_smiles_megamolbart_embs.npy'),
            'Perindopril MPO': os.path.join(project_home, data_folder,
                                            'perindopril_rings.csv_smiles_megamolbart_embs.npy'),
            'Amlodipine MPO': os.path.join(project_home, data_folder,
                                           'amlodipine_rings.csv_smiles_megamolbart_embs.npy'),
            'Sitagliptin MPO': os.path.join(project_home, data_folder,
                                            'sitagliptin_replacement.csv_smiles_megamolbart_embs.npy'),
            'Zaleplon MPO': os.path.join(project_home, data_folder,
                                         'zaleplon_with_other_formula.csv_smiles_megamolbart_embs.npy'),
            'Valsartan SMARTS': os.path.join(project_home, data_folder,
                                             'valsartan_smarts.csv_smiles_megamolbart_embs.npy'),
            'Deco Hop': os.path.join(project_home, data_folder,
                                     'decoration_hop.csv_smiles_megamolbart_embs.npy'),
            'Scaffold Hop': os.path.join(project_home, data_folder,
                                         'scaffold_hop.csv_smiles_megamolbart_embs.npy'),
        }

    def generate_optimized_molecules(self,
                                     scoring_function,
                                     number_molecules,
                                     starting_population,
                                     benchmark_name):

        # success molecules 
        mol_from_ret = []
        seed = 0

        pl.utilities.seed.seed_everything(seed)

        # initial lead
        lead = random.sample(self.input_dataset, self.batch_size)
        lead_attr = scoring_function.score_list(lead)

        # get the retrieval database and retrieval embeddings
        ret_database = pd.read_csv(self.benchmark_retrieval_database[benchmark_name])
        ret_smiles_all = ret_database.smiles.tolist()
        ret_attr = ret_database.value.tolist()

        all_generateds_per_lead = [None] * len(lead)
        all_generated_attrs_per_lead = [None] * len(lead)

        for itr in range(self.n_repeat):

            start = time()

            # retrieve
            retrieve_status = []
            if self.ret_mode == 'per-itr-random':
                retrievals_lead = []
                for l_idx in range(len(lead)):
                    ret_smiles_all_tmp = [ret_smiles_all[i] for i in range(len(ret_smiles_all)) if
                                          ret_attr[i] >= lead_attr[l_idx]]
                    if len(ret_smiles_all_tmp) >= 10:
                        retrievals = random.sample(ret_smiles_all_tmp, self.n_retrievals)
                        retrieve_status.append('normal')
                    else:
                        retrievals = ret_smiles_all_tmp
                        retrievals += all_generateds_per_lead[l_idx][:self.n_retrievals - len(retrievals)]
                        retrieve_status.append('from gen')
                    retrievals_lead.append(retrievals)
                assert (len(retrievals_lead[0]) == self.n_retrievals)
                assert (len(retrievals_lead) == len(lead))
                print(retrieve_status)

            # batched version
            batch = [[{'encoder_smiles': lead[i], 'retrieved_smiles': retrievals_lead[i]}] \
                     * self.n_trials for i in range(len(lead))]
            batch = [item for sublist in batch for item in sublist]
            batch_input = collate_fn_inference(batch)

            gens_, _ = self.wf.model.sample_molecules(batch_input, sampling_alg='random_batch_jitter', jitter_std=1)
            gens_ = [gens_[self.n_trials * b:self.n_trials * (b + 1)] for b in range(len(lead))]
            gens2_, _ = self.wf.model.sample_molecules(batch_input, sampling_alg='random_batch_jitter', jitter_std=2)
            gens2_ = [gens2_[self.n_trials * b:self.n_trials * (b + 1)] for b in range(len(lead))]
            gens3_, _ = self.wf.model.sample_molecules(batch_input, sampling_alg='random_batch_jitter', jitter_std=3)
            gens3_ = [gens3_[self.n_trials * b:self.n_trials * (b + 1)] for b in range(len(lead))]

            batch = [{'encoder_smiles': lead[i], 'retrieved_smiles': retrievals_lead[i]} for i in range(len(lead))]
            batch_input = collate_fn_inference(batch)
            gens_greedy_, _ = self.wf.model.sample_molecules(batch_input, sampling_alg='greedy', jitter_std=None)

            for b_idx in range(len(lead)):
                gens_[b_idx].append(gens_greedy_[b_idx])
                gens_[b_idx] += gens2_[b_idx]
                gens_[b_idx] += gens3_[b_idx]

            # get only valid generations for each lead
            is_valids = [[Chem.MolFromSmiles(g) for g in gs] for gs in gens_]
            is_valids_idx = [[idx for idx in range(len(gens_[g_idx])) if is_valids[g_idx][idx] is not None]
                             for g_idx in range(len(gens_))]
            gens_ = [list(set([gens_[item][idx] for idx in is_valids_idx[item]]))
                     for item in range(len(is_valids_idx))]
            gens_ = [canonicalize_list(gs) for gs in gens_]  # canonicalize
            mols_ = [[Chem.MolFromSmiles(g) for g in gs] for gs in gens_]
            assert (len(is_valids_idx) == len(is_valids) == len(gens_) == len(lead))

            # compute the scores for each generation for each lead
            scores = [scoring_function.score_list(gs) for gs in gens_]

            # include only molecules with a positive score
            pos_score_idx = [[idx for idx in range(len(score)) if score[idx] > 0] for score in scores]
            scores = [[scores[l_idx][idx] for idx in pos_score_idx[l_idx]] for l_idx in range(len(pos_score_idx))]
            mols_ = [[mols_[l_idx][idx] for idx in pos_score_idx[l_idx]] for l_idx in range(len(pos_score_idx))]
            gens_ = [[gens_[l_idx][idx] for idx in pos_score_idx[l_idx]] for l_idx in range(len(pos_score_idx))]

            # use GA to mutate and add to the above molecules
            population_size = 300
            offspring_mol = []
            mol_from_ret_current_batch = []
            for l_idx in range(len(mols_)):

                if all([s > 1e-3 for s in scores[l_idx]]) and len(scores[l_idx]) > 0:
                    if all_generateds_per_lead[l_idx] is not None:
                        tmp_mols = [Chem.MolFromSmiles(s) for s in all_generateds_per_lead[l_idx][:population_size]]
                        m_pool = make_mating_pool(tmp_mols + mols_[l_idx],
                                                  all_generated_attrs_per_lead[l_idx][:population_size] + scores[l_idx],
                                                  population_size)
                    else:
                        m_pool = make_mating_pool(mols_[l_idx], scores[l_idx], min(population_size, len(mols_[l_idx])))
                    offspring = self.pool(
                        delayed(reproduce)(m_pool, self.mutation_rate) for _ in range(population_size))
                    offspring = sanitize(offspring)
                    offspring_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in offspring)
                    if max(scores[l_idx]) >= max(offspring_scores):
                        mol_from_ret_current_batch.append(1)
                    else:
                        mol_from_ret_current_batch.append(0)
                    offspring_mol.append(offspring)
                    scores[l_idx] += offspring_scores
                    gens_[l_idx] += [Chem.MolToSmiles(m) for m in offspring]

            for l_idx in range(len(lead)):
                if all_generated_attrs_per_lead[l_idx] is None:
                    all_generated_attrs_per_lead[l_idx] = scores[l_idx]
                    all_generateds_per_lead[l_idx] = gens_[l_idx]
                all_generated_attrs_per_lead[l_idx] += scores[l_idx]
                all_generateds_per_lead[l_idx] += gens_[l_idx]

                # de-duplicate
                de_duplicate_indices = [all_generateds_per_lead[l_idx].index(x) for x in
                                        set(all_generateds_per_lead[l_idx])]
                all_generateds_per_lead[l_idx] = [all_generateds_per_lead[l_idx][i] for i in de_duplicate_indices]
                all_generated_attrs_per_lead[l_idx] = [all_generated_attrs_per_lead[l_idx][i] for i in
                                                       de_duplicate_indices]

                # sort the generated molecules by value, decreasing
                attr_indices = np.argsort(-np.array(all_generated_attrs_per_lead[l_idx]))

                # only get the top N best saved
                all_generateds_per_lead[l_idx] = [all_generateds_per_lead[l_idx][i] for i in attr_indices[0:1000]]
                all_generated_attrs_per_lead[l_idx] = [all_generated_attrs_per_lead[l_idx][i] for i in
                                                       attr_indices[0:1000]]

                # get next lead
                if max(all_generated_attrs_per_lead[l_idx]) >= lead_attr[l_idx]:
                    set_molecules = set()
                    best_idx = random.choice([i for i, x in enumerate(all_generated_attrs_per_lead[l_idx]) if
                                              x == max(all_generated_attrs_per_lead[l_idx])])  # more greedy
                    lead[l_idx] = all_generateds_per_lead[l_idx][best_idx]
                    lead_attr[l_idx] = all_generated_attrs_per_lead[l_idx][best_idx]

            # update lead
            print('itr{}: leads: {}'.format(itr, lead))
            print('itr{}: lead attr: {}'.format(itr, lead_attr))
            end = time()
            print('itr{}: time: {:.4f}'.format(itr, end - start))

        # get the best N molecules from all generations
        all_gens = [item for sublist in all_generateds_per_lead for item in sublist]
        all_gens = remove_duplicates(canonicalize_list(all_gens))
        all_attr = scoring_function.score_list(all_gens)
        assert (len(all_gens) >= number_molecules)
        attr_indices = np.argsort(-np.array(all_attr))
        molecules = [all_gens[i] for i in attr_indices[0:1000]]
        molecules = list(set(molecules))
        molecules = canonicalize_list(molecules)
        molecules = remove_duplicates(molecules)
        print('number of molecules: {}'.format(len(molecules)))
        print('benchmark: {}'.format(benchmark_name))
        print('\n\n')

        self.molecules = molecules

        with open('log_mol_from_ret_benchmark-{}.pkl'.format(benchmark_name), 'wb') as f:
            pickle.dump(mol_from_ret, f)

        return self.molecules
