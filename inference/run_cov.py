# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for RetMol. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

#!/usr/bin/env python3

'''
python run_cov.py --model_path models/retmol_chembl --sim_thres 0.6
'''
import os
import sys
import pandas as pd
import pickle
from time import time
import torch
from tqdm import tqdm
import subprocess
from functools import partial
from multiprocessing import Pool
import pytorch_lightning as pl

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, project_home + '/inference')
from inference import MegaMolBART
sys.path.insert(1, project_home + '/inference/utils')
from properties import *

sys.path.insert(1, project_home + '/MolBART')
from csv_data_retrieval import collate_fn_inference

from pdb import set_trace

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_ckpt_itr", default=50000, type=int)
    parser.add_argument("--attr", default='qed,sa,docking')
    parser.add_argument("--samp_alg", default='random_batch_jitter',
                        help='[random_batch_jitter_and_beam, random_batch_jitter, beam]')
    parser.add_argument('--jitter_std', default=1, type=float)
    parser.add_argument('--n_retrievals', default=10, type=int)
    parser.add_argument('--ret_size', default=100, type=int)
    parser.add_argument('--not_enough_ret_mode', default='ignore', help='ignore, best-in-train, best-in-gen')
    parser.add_argument('--n_repeat', default=10, type=int)
    parser.add_argument('--n_trials', default=100, type=int)
    parser.add_argument('--n_top_gens', default=1, type=int)
    parser.add_argument('--beam_ratio', default=0.1, type=float)
    parser.add_argument('--max_mol_len', default=200, type=int, help='original default is 512')
    parser.add_argument('--n_gens', default=6300, type=int)
    parser.add_argument('--sim_thres', default=0.6, type=float)
    args = parser.parse_args()

    model_path = os.path.join(project_home, args.model_path)
    model_ckpt_itr = args.model_ckpt_itr
    attr = args.attr  # 'logp-sa'
    samp_alg = args.samp_alg
    jitter_std = args.jitter_std
    n_retrievals = args.n_retrievals
    n_repeat = args.n_repeat  # 10
    not_enough_ret_mode = args.not_enough_ret_mode
    n_trials = args.n_trials  # 100
    n_top_gens = args.n_top_gens
    beam_ratio = args.beam_ratio
    max_mol_len = args.max_mol_len
    n_gens = args.n_gens
    ret_size = args.ret_size
    sim_thres = args.sim_thres

    # load the saved attributes
    df = pd.read_csv(os.path.join(project_home, 'data/cov/input_molecules.csv'))
    df_input = df[df.docking_scores > -10]

    # load the input test molecules
    leads = df_input.smiles.tolist()
    leads_attrs = df_input.docking_scores.tolist()


    def tanimoto_dist_func(lead_fp, ret):
        return DataStructs.TanimotoSimilarity(
            lead_fp,
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ret), 2, nBits=2048))


    # set up path
    save_path = os.path.join('results/{}/cov'.format(args.model_path.split('/')[-1].strip()))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = '{}/COV_CXF1_ckpt{}_{}_retrievals{}_failmode-{}_repeats{}_trials{}_ngenperitr{}_{}-std{}-max-mol-len-' \
                '{}_ngens-{}-retsize-{}-sim{}-sample-select-QEDFirst-KeepLead.csv'.format(
        save_path, model_ckpt_itr, '-'.join(attr.split(',')), n_retrievals, not_enough_ret_mode,
        n_repeat, n_trials, n_top_gens, samp_alg, jitter_std, max_mol_len, n_gens, sim_thres, ret_size)

    os.makedirs(os.path.join(project_home, 'data/cov/tmp/docking_temporary_data'), exist_ok=True)
    os.makedirs(os.path.join(project_home, 'data/cov/tmp/docking_temporary_output'), exist_ok=True)


    # scoring functions
    class SimFunc(object):
        def __init__(self, lead):
            lead_mol = Chem.MolFromSmiles(lead)
            self.lead_fp = AllChem.GetMorganFingerprintAsBitVect(lead_mol, 2, 2048)

        def __call__(self, smiles_list):
            mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
            mol_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
            sim = DataStructs.BulkTanimotoSimilarity(self.lead_fp, mol_fps)
            return sim


    def docking_func(smiles_list):
        all_results = []
        docking_scores = []

        for n in range(len(smiles_list)):
            try:
                # prepare ligand
                mol = Chem.MolFromSmiles(smiles_list[n])
                mol = Chem.AddHs(mol, addCoords=True)
                params = AllChem.ETKDGv3()
                params.useRandomCoords = False
                params.randomSeed = 0
                conf_stat = AllChem.EmbedMolecule(mol, params)
                AllChem.UFFOptimizeMolecule(mol, maxIters=100)
                writer = Chem.SDWriter(
                    os.path.join(project_home, 'data/cov/tmp/docking_temporary_data/generated_smiles111.sdf'))
                mol.SetProp("_Name", "generated_smiles111")
                mol.SetProp("_SMILES", "%s" % smiles_list[n])
                writer.write(mol)
                writer.close()
                subprocess.call(['mk_prepare_ligand.py', '-i', os.path.join(
                    project_home,
                    'data/cov/tmp/docking_temporary_data/generated_smiles111.sdf'), '--o', os.path.join(
                    project_home,
                    'data/cov/tmp/docking_temporary_data/generated_smiles111.pdbqt')])

                # run docking
                os.chdir(os.path.join(project_home, 'data/cov'))
                result = subprocess.run([os.path.join(
                    project_home, 'AutoDock-GPU/bin/autodock_gpu_256wi'),
                    '--ffile', os.path.join(
                        project_home, 'data/cov/7l11_dry.maps.fld'),
                    '--lfile', os.path.join(
                        project_home,
                        'data/cov/tmp/docking_temporary_data/generated_smiles111.pdbqt'),
                    '--resnam', os.path.join(
                        project_home,
                        'data/cov/tmp/docking_temporary_output/log_generated_smiles111.txt')],
                    stdout=subprocess.PIPE)
                result = result.stdout.decode('utf-8')
                all_results.append(result)
                docking_scores.append(float(result.split('\n')[-13].split()[-2]))
            except:
                docking_scores.append(100)

        set_trace()

        return docking_scores


    pl.utilities.seed.seed_everything(1234)
    with torch.no_grad():
        wf = MegaMolBART(model_path=model_path, model_ckpt_itr=model_ckpt_itr, decoder_max_seq_len=max_mol_len)

        ###############################################################################################
        # attribute controlled generation
        ###############################################################################################

        itr_per_lead = []
        all_generated = []
        leads_seen = []
        failed_reasons_all = []
        success_counter = 0
        lead_itr = 0

        lead_sequences = []
        lead_attr_sequences = []

        for lead_idx in tqdm(range(len(leads))):

            lead = leads[lead_idx]

            # scoring functions
            scoring_fns = {}
            thresholds = {}
            if 'qed' in attr:
                scoring_fns['qed'] = qed_func()
                thresholds['qed'] = 0.6
            if 'sa' in attr:
                scoring_fns['sa'] = sa_func()
                thresholds['sa'] = -4
            if 'sim' in attr:
                scoring_fns['sim'] = sim_func(lead)
                thresholds['sim'] = sim_thres
            if 'docking' in attr:
                scoring_fns['docking'] = docking_func
                thresholds['docking'] = 100
            sim_func = SimFunc(lead)

            leads_seen.append(lead)
            lead_sequence = []
            lead_attr_sequence = []

            ret_tmp = df.smiles.tolist()
            retrieval_database = {'smiles': ret_tmp}

            print('lead: {}'.format(lead))

            lead_itr += 1
            pl.utilities.seed.seed_everything(1234)

            init_lead = lead
            init_lead_attr = leads_attrs[lead_idx]

            if type(lead) == str:
                lead = [lead]

            all_generateds_per_lead = []
            best_gens_qed = []
            best_gens_docking = []
            failed_reasons = []
            for itr in tqdm(range(n_repeat)):

                # get the retrieval set for the current lead, and rerank by similarity
                retrievals_lead = []
                for l in lead:
                    with Pool() as pool:
                        sims = pool.map(partial(
                            tanimoto_dist_func,
                            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(l), 2, nBits=2048)),
                            retrieval_database['smiles'])
                    retrieval_database['sim'] = sims
                    tmp_df = pd.DataFrame(retrieval_database)
                    data_attr_for_this_lead = tmp_df.sort_values(by='sim', ascending=False)  # rank by similarity

                    # get retreivals
                    retrievals = data_attr_for_this_lead.smiles.tolist()[:n_retrievals]
                    retrievals_lead.append(retrievals)

                assert (len(retrievals_lead[0]) == n_retrievals)
                assert (len(retrievals_lead) == len(lead))

                # compute lead attributes
                lead_attrs = []
                lead_conditions = []
                for l in lead:
                    lead_attr = {}
                    lead_condition = []
                    for a in attr.split(','):
                        lead_attr[a] = scoring_fns[a]([l])[0] if a != 'sa' and a != 'docking' else -scoring_fns[a]([l])[
                            0]
                        lead_condition.append(scoring_fns[a]([l])[0] >= thresholds[a])
                    lead_attrs.append(lead_attr)
                    lead_conditions.append(lead_condition)

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
                    gens_ += gens_greedy_

                # get only unique and valid generations
                gens_ = list(set(gens_))
                gens_mols_ = []
                for v_idx in range(len(gens_)):
                    try:
                        g = gens_[v_idx]
                        mol = Chem.MolFromSmiles(g)
                        if mol is not None:
                            gens_mols_.append(mol)
                    except:
                        continue
                # canonicalize
                gens_ = [Chem.MolToSmiles(m, isomericSmiles=False) for m in gens_mols_]

                # get only unique and valid generations, one more time (to remove some errors when computing similarity)
                gens_ = list(set(gens_))
                gens_mols_ = []
                for v_idx in range(len(gens_)):
                    try:
                        g = gens_[v_idx]
                        mol = Chem.MolFromSmiles(g)
                        if mol is not None:
                            gens_mols_.append(mol)
                    except:
                        continue
                # canonicalize (to remove some errors when computing similarity)
                gens_ = [Chem.MolToSmiles(m, isomericSmiles=False) for m in gens_mols_]

                # get only the molecules above similarity
                sims = sim_func(gens_)
                gens_ = [gens_[g_idx] for g_idx in range(len(gens_)) if sims[g_idx] >= sim_thres]

                if len(gens_) == 0:
                    print('no valid molecules generated')
                    failed_reasons.append('no valid molecules generated')

                else:
                    # compute generated molecules' attributes
                    gens_attrs = {}
                    for a in attr.split(','):
                        gens_attrs[a] = scoring_fns[a](gens_) if a != 'sa' and a != 'docking' else \
                            [-s for s in scoring_fns[a](gens_)]
                    gens_attrs_per_gen = [{a: gens_attrs[a][g_idx] for a in attr.split(',')}
                                          for g_idx in range(len(gens_))]

                    qed_scores = gens_attrs['qed']
                    good_qed_score_idx = [i for i in range(len(qed_scores)) if qed_scores[i] >= 0.6]
                    sa_scores = gens_attrs['sa']
                    good_sa_score_idx = [i for i in range(len(sa_scores)) if sa_scores[i] >= -4]
                    docking_scores = gens_attrs['docking']

                    # first optimize for QED
                    if len(good_qed_score_idx) == 0:
                        print('no mol satisfy qed, optimizing for qed')

                        # update lead
                        best_idx = qed_scores.tolist().index(max(qed_scores))
                        lead = [gens_[best_idx]]
                        new_lead_sim = sim_func(lead)
                        assert (len(new_lead_sim) == 1)
                        new_lead_attr = gens_attrs_per_gen[best_idx]
                        new_lead_attr['sim'] = new_lead_sim[0]
                        print('update lead; prev attr = {}; new attr = {}'.format(lead_attrs[0], new_lead_attr))
                        print('new lead: {}'.format(lead[0]))

                        # update retrieval set
                        ranked_gens = [gens_[i] for i in range(len(gens_)) if gens_attrs['qed'][i] >= 0.6]
                        best_gens_qed += ranked_gens
                        best_gens_qed = list(set(best_gens_qed))
                        retrieval_database['smiles'] = df.smiles.tolist() + best_gens_qed
                        print('retrieval database size: {}'.format(len(retrieval_database['smiles'])))

                    # then optimize for docking
                    else:
                        print('optimizing for docking')
                        good_idx = set(good_qed_score_idx).intersection(set(good_sa_score_idx))
                        docking_scores = [docking_scores[i] for i in good_idx]
                        gens_ = [gens_[i] for i in good_idx]
                        gens_attrs = [gens_attrs_per_gen[i] for i in good_idx]
                        assert (
                            all([docking_scores[i] == gens_attrs[i]['docking'] for i in range(len(docking_scores))]))

                        # update lead
                        best_idx = docking_scores.index(max(docking_scores))
                        lead = [gens_[best_idx]]
                        new_lead_sim = sim_func(lead)
                        assert (len(new_lead_sim) == 1)
                        new_lead_attr = gens_attrs[best_idx]
                        new_lead_attr['sim'] = new_lead_sim[0]
                        print('update lead; prev attr = {}; new attr = {}'.format(lead_attrs[0], new_lead_attr))
                        print('new lead: {}'.format(lead[0]))

                        # update retrieval 
                        ranked_gens = [gens_[i] for i in range(len(gens_)) if
                                       gens_attrs[i]['docking'] >= init_lead_attr]  # lead_attrs[0]['docking']]
                        best_gens_docking += ranked_gens
                        best_gens_docking = list(set(best_gens_docking))
                        retrieval_database['smiles'] = df.smiles.tolist() + best_gens_docking
                        print('retrieval database size: {}'.format(len(retrieval_database['smiles'])))

                    # add the next input to the sequence
                    lead_sequence.append(lead[0])
                    lead_attr_sequence.append(new_lead_attr)

            success_indicator = 1 if len(all_generateds_per_lead) > 0 else 0
            success_counter += success_indicator
            all_generated.append(all_generateds_per_lead)
            lead_sequences.append(lead_sequence)
            lead_attr_sequences.append(lead_attr_sequence)
            itr_per_lead.append(itr + 1)
            print('number of successes so far: {}/{}={:.4f}'.format(
                success_counter, lead_itr, success_counter / lead_itr))

            res = {'original_smiles': leads_seen,
                   'gen_smiles_all': all_generated,
                   'gen_sequences': lead_sequences,
                   'gen_sequence_attrs': lead_attr_sequences,
                   'iterations': itr_per_lead,
                   }
            with open(save_path, 'wb') as f:
                pickle.dump(res, f)
