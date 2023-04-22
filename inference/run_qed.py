# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for RetMol. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

#!/usr/bin/env python3

'''
example run:

python run_qed.py --model_path models/retmol_zinc --similarity_thres 0.4 --chunk 1
'''
import os
import sys
import pandas as pd
import pickle
from time import time
import torch
from tqdm import tqdm
import numpy as np
from functools import partial
from multiprocessing import Pool
import pytorch_lightning as pl

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed

from inference import MegaMolBART

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, project_home + '/MolBART')
from csv_data_retrieval import collate_fn_inference

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_ckpt_itr", default=50000, type=int)
    parser.add_argument("--attr", default='qed')
    parser.add_argument("--samp_alg", default='random_batch_jitter',
                        help='[random_batch_jitter_and_beam, random_batch_jitter, beam]')
    parser.add_argument('--jitter_std', default=1, type=float)
    parser.add_argument("--similarity_thres", required=True, type=float)  # default=0.6, type=float)
    parser.add_argument('--n_neighbors', default=1000, type=int)
    parser.add_argument('--n_retrievals', default=10, type=int)
    parser.add_argument('--not_enough_ret_mode', default='ignore', help='ignore, best-in-train, best-in-gen')
    parser.add_argument('--n_repeat', default=20, type=int)
    parser.add_argument('--n_trials', default=50, type=int)
    parser.add_argument('--n_top_gens', default=3, type=int)
    parser.add_argument('--beam_ratio', default=0.1, type=float)
    parser.add_argument('--n_chunks', default=8, type=int)
    parser.add_argument('--chunk', required=True, type=int)
    parser.add_argument("--save_ret_vals", default='False', type=str)
    args = parser.parse_args()

    model_path = os.path.join(project_home, args.model_path)
    model_ckpt_itr = args.model_ckpt_itr
    attr = args.attr  # 'logp-sa'
    samp_alg = args.samp_alg
    jitter_std = args.jitter_std
    similarity_thres = args.similarity_thres  # 0.6
    n_neighbors = args.n_neighbors  # 1000
    n_retrievals = args.n_retrievals
    n_repeat = args.n_repeat  # 10
    not_enough_ret_mode = args.not_enough_ret_mode
    n_trials = args.n_trials  # 100
    n_top_gens = args.n_top_gens
    beam_ratio = args.beam_ratio
    n_chunks = args.n_chunks
    chunk = args.chunk
    save_ret_vals = args.save_ret_vals
    print_stats = False

    # set up path
    save_path = os.path.join(project_home,
                             'results/{}/control_gen_sim1st_ckpt{}_{}_thres{}_neighbors{}_retrievals{}_failmode-{}_'
                             'repeats{}_trials{}_ngenperitr{}_{}-std{}'.format(
                                 args.model_path.split('/')[-1].strip(), model_ckpt_itr, attr, similarity_thres,
                                 n_neighbors, n_retrievals, not_enough_ret_mode, n_repeat, n_trials, n_top_gens,
                                 samp_alg, jitter_std))
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)

    with torch.no_grad():
        wf = MegaMolBART(model_path=model_path, model_ckpt_itr=model_ckpt_itr)

        ###############################################################################################
        # attribute controlled generation
        ###############################################################################################
        ## load the saved attributes
        start_t = time()
        data_attr = pd.read_csv(os.path.join(project_home,
                                             'data/retrieval-precompute/mol_property_precompute/mol_properties.csv'))
        data_attr = data_attr[data_attr[attr] >= 0.9]  # QED task specific
        n_best_train_idx = np.argsort(data_attr[attr].tolist())[-n_retrievals:]
        n_best_train = [data_attr.smiles.tolist()[i] for i in n_best_train_idx]

        ## load the retrieval set
        from tdc.generation import MolGen

        data = MolGen(name='ZINC', path=os.path.join(project_home, 'data/zinc'))
        zinc250_train = data.get_data().smiles.tolist()
        zinc250_train_set = zinc250_train
        leads_and_attr = pd.read_csv(os.path.join(project_home,
                                                  'data/zinc/inference_inputs/qed_test_w_vals.csv'))
        leads_and_attr = leads_and_attr.rename(columns={0: "smiles", 1: attr})

        # get the pre-computed similarity
        leads_and_sim = pickle.load(open(os.path.join(project_home,
                                                      'data/retrieval-precompute/similarity_precompute/'
                                                      'zinc250_JTNN_QMO_TrainSetRetrieval_TestSetQuery_TanimotoTop10000_with-qed.pkl'),
                                         'rb'))

        leads = leads_and_attr.smiles.tolist()
        chunk_size = int(len(leads) / n_chunks)
        assert (chunk_size * n_chunks == len(leads))
        leads = leads[(chunk - 1) * chunk_size:chunk * chunk_size]

        final_generated = []
        final_generated_attr = []
        final_generated_sim = []
        itr_per_lead = []
        all_generated = []
        all_generated_attr = []
        all_generated_sim = []
        leads_seen = []
        leads_attr_seen = []
        failed_reasons_all = []
        ret_vals_mean_all = []

        for lead in tqdm(leads, position=0, leave=True):
            pl.utilities.seed.seed_everything(1234)

            init_lead_attr = leads_and_attr[leads_and_attr.smiles == lead][attr].iloc[0]
            init_lead = lead
            init_lead_fp = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(init_lead), 2, nBits=2048)


            # get the init retrieval set
            def tanimoto_dist_func(lead_fp, ret):
                return DataStructs.TanimotoSimilarity(
                    lead_fp,
                    AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ret), 2, nBits=2048))


            with Pool() as pool:
                sims = pool.map(partial(
                    tanimoto_dist_func,
                    AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(lead), 2, nBits=2048)),
                    data_attr.smiles.tolist())
            data_attr['sim'] = sims
            data_attr_for_this_lead = data_attr.sort_values(by='sim', ascending=False)  # rank by similarity

            all_generateds_per_lead = []
            all_generated_attrs_per_lead = []
            all_generated_sims_per_lead = []
            failed_reasons = []
            ret_vals_mean = []

            # ## compute retrieval at the beginning
            # ## recompute the similarity scores using tanimoto distance and rerank
            start_t = time()
            init_retrievals = leads_and_sim[init_lead][0]  # use saved
            retrievals_attr = leads_and_sim[init_lead][1]  # use saved
            n_best_retrievals_idx = np.argsort(retrievals_attr)[-n_retrievals:]
            n_best_retrievals = [init_retrievals[i] for i in n_best_retrievals_idx]

            for itr in tqdm(range(n_repeat), position=0, leave=True):
                ## get the attribute value greater than the starting molecule
                start_t = time()
                if type(lead) == str:
                    lead = [lead]
                lead_attr = []
                for l in lead:
                    l_attr = qed(Chem.MolFromSmiles(l))
                    lead_attr.append(l_attr)

                ## get the top 10 retrievals with attr values > lead
                retrievals_lead = []  # this is a list of lists
                failed_lead_idx = []
                for l_idx in range(len(lead)):
                    retrievals = []
                    retrievals_top_attr_values = []
                    for idx in range(len(init_retrievals)):
                        if retrievals_attr[idx] > lead_attr[l_idx]:
                            retrievals.append(init_retrievals[idx])
                            if len(retrievals) >= n_retrievals:
                                break

                    # if retrieval set smaller than n_retrievals, additionally use the generated molecules set for retrieval
                    if len(retrievals) < n_retrievals:
                        # retrievals += all_generated_sims_per_lead[:n_retrievals-len(retrievals)]
                        if not_enough_ret_mode == 'best-in-ret':
                            retrievals += n_best_retrievals[:n_retrievals - len(retrievals)]
                        elif not_enough_ret_mode == 'best-in-train':
                            # retrievals += n_best_train[:n_retrievals-len(retrievals)]
                            retrievals += data_attr_for_this_lead[:n_retrievals - len(retrievals)]
                        elif not_enough_ret_mode == 'best-in-gen':
                            retrievals += all_generated_sims_per_lead[:n_retrievals - len(retrievals)]
                        else:
                            pass

                    if len(retrievals) < n_retrievals:
                        failed_reasons.append('not enough retrievals')
                        failed_lead_idx.append(l_idx)
                    else:
                        retrievals_lead.append(retrievals)

                # compute retrievals average penalized logP value
                ret_for_val_comp = [item for sublist in retrievals_lead for item in sublist]
                ret_vals = np.array([qed(Chem.MolFromSmiles(rs)) for rs in ret_for_val_comp])
                ret_vals_mean.append(ret_vals.mean().item())

                lead = [lead[i] for i in range(len(lead)) if i not in failed_lead_idx]
                if len(lead) == 0:
                    break
                assert (len(lead) == len(retrievals_lead))

                # encode the lead molecule with generative model's encoder
                # batched version
                start_t = time()
                if samp_alg == 'random_batch_jitter':
                    batch = [[{'encoder_smiles': lead[i], 'retrieved_smiles': retrievals_lead[i]}] * n_trials for i in
                             range(len(lead))]
                    batch = [item for sublist in batch for item in sublist]
                    batch_input = collate_fn_inference(batch)
                    gens_, _ = wf.model.sample_molecules(batch_input, sampling_alg='random_batch_jitter',
                                                         jitter_std=jitter_std)
                    batch = [{'encoder_smiles': lead[i], 'retrieved_smiles': retrievals_lead[i]} for i in
                             range(len(lead))]
                    batch_input = collate_fn_inference(batch)
                    gens_greedy_, _ = wf.model.sample_molecules(batch_input, sampling_alg='greedy', jitter_std=None)
                    gens_greedy_ = [item for sublist in gens_greedy_ for item in sublist]
                    gens_ += gens_greedy_

                gens = []
                gens_attr = []
                is_valids = []
                is_valids_idx = []
                for v_idx in range(len(gens_)):
                    try:
                        g = gens_[v_idx]
                        mol = Chem.MolFromSmiles(g)
                        prop = qed(mol)
                        is_valids.append(mol)
                        is_valids_idx.append(v_idx)
                    except:
                        continue
                gens_ = [gens_[idx] for idx in is_valids_idx]

                if len(is_valids) == 0:
                    failed_reasons.append('no valid molecules generated')
                else:
                    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(g, 2, nBits=2048) for g in is_valids]
                    sim_scores = [DataStructs.TanimotoSimilarity(init_lead_fp, gen_fp) for gen_fp in gen_fps]
                    gen_attrs = [qed(g) for g in is_valids]
                    assert (len(gen_fps) == len(sim_scores) == len(gen_attrs))

                    # get current best according to similarity score
                    sim_scores_rm_identical = [sim_scores[i] for i in range(len(sim_scores)) if sim_scores[i] < 1]
                    gens_rm_identical = [gens_[i] for i in range(len(gens_)) if sim_scores[i] < 1]
                    de_duplicate_indices = [gens_rm_identical.index(x) for x in set(gens_rm_identical)]
                    sim_scores_rm_identical = [sim_scores_rm_identical[i] for i in de_duplicate_indices]
                    gens_rm_identical = [gens_rm_identical[i] for i in de_duplicate_indices]
                    attr_indices = np.argsort(-np.array(sim_scores_rm_identical))
                    cur_best_gen_sim = [gens_rm_identical[idx] for idx in attr_indices[:n_top_gens]]

                    # get the generations that meet the requirements
                    lead_set = set(lead)
                    gens = [gens_[idx] for idx in range(len(gens_)) if \
                            sim_scores[idx] > similarity_thres and sim_scores[idx] < 1 and \
                            gen_attrs[idx] > max(lead_attr) and gens_[idx] not in lead_set]
                    gens_attr = [gen_attrs[idx] for idx in range(len(gens_)) if \
                                 sim_scores[idx] > similarity_thres and sim_scores[idx] < 1 and \
                                 gen_attrs[idx] > max(lead_attr) and gens_[idx] not in lead_set]
                    gens_sim = [sim_scores[idx] for idx in range(len(gens_)) if \
                                sim_scores[idx] > similarity_thres and sim_scores[idx] < 1 and \
                                gen_attrs[idx] > max(lead_attr) and gens_[idx] not in lead_set]

                if len(gens) > 0:
                    # merge existing attrs and gens
                    all_generateds_per_lead += gens
                    all_generated_attrs_per_lead += gens_attr
                    all_generated_sims_per_lead += gens_sim

                    # de-duplicate
                    de_duplicate_indices = [all_generateds_per_lead.index(x) for x in set(all_generateds_per_lead)]
                    all_generateds_per_lead = [all_generateds_per_lead[i] for i in de_duplicate_indices]
                    all_generated_attrs_per_lead = [all_generated_attrs_per_lead[i] for i in de_duplicate_indices]
                    all_generated_sims_per_lead = [all_generated_sims_per_lead[i] for i in de_duplicate_indices]

                    # rank by attr value and select the best among existing 
                    attr_indices = np.argsort(-np.array(all_generated_attrs_per_lead))
                    all_generateds_per_lead = [all_generateds_per_lead[i] for i in
                                               attr_indices]  # rank generated mols by attr value
                    all_generated_attrs_per_lead = [all_generated_attrs_per_lead[i] for i in attr_indices]
                    all_generated_sims_per_lead = [all_generated_sims_per_lead[i] for i in attr_indices]
                    gens = all_generateds_per_lead[:n_top_gens]

                    cur_best_gen = all_generateds_per_lead[0]  # [attr_indices[0]]
                    cur_best_attr = all_generated_attrs_per_lead[0]  # [attr_indices[0]]
                    cur_best_sim = all_generated_sims_per_lead[0]  # [attr_indices[0]]

                    if any([v >= 0.9 for v in all_generated_attrs_per_lead]):
                        print(
                            'SUCCESS! current best generation: {}, attr={:.4f} (init={:.4f}), similarity={:.4f}'.format(
                                cur_best_gen, cur_best_attr, init_lead_attr, cur_best_sim))
                        failed_reasons.append('success')
                        # succeed!
                        break

                    lead = gens

                else:
                    failed_reasons.append('constraint not satisfied')

            # record final generation for this lead
            if all([v < 0.9 for v in all_generated_attrs_per_lead]):
                # failed
                final_generated.append(None)
                final_generated_attr.append(None)
                final_generated_sim.append(None)
                all_generated.append(None)
                all_generated_attr.append(None)
                all_generated_sim.append(None)
                itr_per_lead.append(0)
                # break # next lead
            else:
                # select the best one among all generated for this lead
                indices = np.argsort(-np.array(all_generated_attrs_per_lead))
                final_generated.append(all_generateds_per_lead[indices[0]])
                final_generated_attr.append(all_generated_attrs_per_lead[indices[0]])
                final_generated_sim.append(all_generated_sims_per_lead[indices[0]])
                all_generated.append(all_generateds_per_lead)
                all_generated_attr.append(all_generated_attrs_per_lead)
                all_generated_sim.append(all_generated_sims_per_lead)
                itr_per_lead.append(itr)

            leads_seen.append(init_lead)
            leads_attr_seen.append(init_lead_attr)
            ret_vals_mean_all.append(ret_vals_mean)

            print('success/failure so far: {}/{}'.format(
                sum([1 for item in final_generated if item is not None]),
                sum([1 for item in final_generated if item is None])))

        if save_ret_vals.lower() == 'false':
            pd.DataFrame({'original_smiles': leads_seen,
                          'original_{}'.format(attr): leads_attr_seen,
                          'gen_smiles': final_generated,
                          'gen_{}'.format(attr): final_generated_attr,
                          'gen_smiles_all': all_generated,
                          'gen_{}_all'.format(attr): all_generated_attr,
                          'gen_sim': final_generated_sim,
                          'all_gen_sim': all_generated_sim,
                          'iterations': itr_per_lead,
                          }).to_csv(
                os.path.join(save_path, 'chunk_{}.csv'.format(chunk))
            )
        elif save_ret_vals.lower() == 'true':
            with open(os.path.join(save_path, 'chunk_{}_ret_vals.csv'.format(chunk)), 'wb') as f:
                pickle.dump(ret_vals_mean_all, f)

        ###############################################################################################
