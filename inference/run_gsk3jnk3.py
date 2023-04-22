# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for RetMol. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

#!/usr/bin/env python3

'''
example run:

python run_gsk3jnk3.py \
    --model_path models/retmol_chembl \
    --test_data_path data/gsk3jnk3/inference_input/inference_inputs.csv \
    --ret_data_path chembl-4attr \
    --ret_mode per-itr-faiss
'''

import os
import sys
import pandas as pd
from time import time
import torch
from tqdm import tqdm
import faiss
import random
import numpy as np
from functools import partial
from multiprocessing import Pool
import pytorch_lightning as pl

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from utils_inference.properties import *
from inference import MegaMolBART
from cheminformatics.utils.inference_original import MegaMolBART as MegaMolBartOriginal

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, project_home + '/MolBART')
from csv_data_retrieval import collate_fn_inference
sys.path.insert(1, project_home + '/MolBART/megatron_molbart')
from csv_data import collate_fn_inference as collate_fn_inference_original

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_ckpt_itr", default=50000, type=int)
    parser.add_argument("--test_data_path", required=True, help='path of the random generated input data')
    parser.add_argument("--ret_data_path", required=True, help='chembl or multiobj with -2attr or 4attr or gsk3-jnk3')
    parser.add_argument("--attr", default='gsk3,jnk3,qed,sa')
    parser.add_argument("--samp_alg", default='random_batch_jitter_and_greedy',
                        help='[random_batch_jitter, greedy, random_batch_jitter_and_greedy]')
    parser.add_argument('--jitter_std', default=1, type=float)
    parser.add_argument('--ret_model', default='megamolbart', type=str)
    parser.add_argument('--ret_mode', required=True, help='per-lead-tanimoto, per-lead-faiss, per-itr-tanimoto, '
                                                          'per-itr-faiss, per-lead-random, per-itr-random, per-lead-topn')
    parser.add_argument('--n_retrievals', default=10, type=int)
    parser.add_argument('--n_repeat', default=10, type=int)
    parser.add_argument('--n_trials', default=1, type=int)
    parser.add_argument('--n_top_gens', default=1, type=int)
    parser.add_argument('--max_mol_len', default=200, type=int, help='original default is 512')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_gens', default=6300, type=int)
    parser.add_argument('--ret_subsample', default=-1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    model_path = os.path.join(project_home, args.model_path)
    model_ckpt_itr = args.model_ckpt_itr
    test_data_path = args.test_data_path
    ret_data_path = args.ret_data_path
    attr = args.attr  # 'logp-sa'
    samp_alg = args.samp_alg
    jitter_std = args.jitter_std
    ret_model = args.ret_model
    ret_mode = args.ret_mode
    n_retrievals = args.n_retrievals
    n_repeat = args.n_repeat  # 10
    n_trials = args.n_trials  # 100
    n_top_gens = args.n_top_gens
    max_mol_len = args.max_mol_len
    batch_size = args.batch_size
    n_gens = args.n_gens
    ret_subsample = args.ret_subsample
    seed = args.seed


    def tanimoto_dist_func(lead_fp, ret):
        return DataStructs.TanimotoSimilarity(
            lead_fp,
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ret), 2, nBits=2048))

    # set up path
    save_path = os.path.join(project_home,
                             'results/{}/{}'.format(args.model_path.split('/')[-1].strip(),
                                                    test_data_path.split('/')[-1].split('.')[0].strip()))
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    save_path = '{}/ckpt{}_{}_retrievals{}_repeats{}_trials{}_ngenperitr{}_{}-std{}-max-mol-len-{}_bs-{}_ret-{}-' \
                'randsub{}_retmode-{}-{}_ngens-{}-trajectory-seed{}.csv'.format(
        save_path, model_ckpt_itr, '-'.join(attr.split(',')), n_retrievals,
        n_repeat, n_trials, n_top_gens, samp_alg, jitter_std, max_mol_len,
        batch_size, ret_data_path, ret_subsample, ret_mode, ret_model, n_gens, seed)

    pl.utilities.seed.seed_everything(seed)
    with torch.no_grad():
        wf = MegaMolBART(model_path=model_path, model_ckpt_itr=model_ckpt_itr, decoder_max_seq_len=max_mol_len)
        emb_model = MegaMolBartOriginal(model_path=os.path.join(project_home, 'models/megamolbart/checkpoints'),
                                        model_ckpt_itr=134000, decoder_max_seq_len=max_mol_len).model

        ###############################################################################################
        # attribute controlled generation
        ###############################################################################################
        ## load the saved attributes
        ret_path_tmp = 'gsk3_jnk3_qed_sa' if ret_data_path.split('-')[1] == '4attr' else 'dual_gsk3_jnk3'
        if ret_data_path == 'chembl-all':
            data_attr = pd.read_csv(os.path.join(project_home,
                                                 'data/chembl/all.txt'), header=None).rename(columns={0: 'smiles'})
            ret_smiles_all = data_attr.smiles.tolist()
        elif ret_data_path == 'gsk3-jnk3':
            ret_smiles_gsk3 = pd.read_csv(os.path.join(project_home,
                                                       'data/gsk3jnk3/gsk3/actives.txt')).smiles.tolist()
            ret_smiles_jnk3 = pd.read_csv(os.path.join(project_home,
                                                       'data/gsk3jnk3/jnk3/actives.txt')).smiles.tolist()
        elif ret_data_path == 'gsk3-jnk3-chembl':
            ret_smiles_gsk3 = pd.read_csv(os.path.join(project_home,
                                                       'data/gsk3jnk3/gsk3/actives_from_chembl.csv')).smiles.tolist()
            ret_smiles_jnk3 = pd.read_csv(os.path.join(project_home,
                                                       'data/gsk3jnk3/jnk3/actives_from_chembl.csv')).smiles.tolist()
        else:
            data_attr = pd.read_csv(os.path.join(project_home,
                                                 'data/gsk3jnk3/{}/actives_from_{}.csv'.format(
                                                     ret_path_tmp,
                                                     ret_data_path.split('-')[0])))
            ret_smiles_all = data_attr.smiles.tolist()

        # load the input test molecules
        leads_all = pd.read_csv(os.path.join(project_home, test_data_path)).smiles.tolist()
        leads_idx_all = random.sample(list(range(len(leads_all))), n_gens)
        leads_all = [leads_all[l_idx] for l_idx in leads_idx_all]
        n_chunks = int(np.math.ceil(n_gens / batch_size))

        # load the saved embeddings if using faiss to compute similarity
        if 'faiss' in ret_mode:
            import faiss

            if ret_model == 'cddd':
                ret_embs = np.load(os.path.join(project_home,
                                                'data/gsk3jnk3/{}/actives_from_{}_embs.npy'.format(
                                                    ret_path_tmp,
                                                    ret_data_path.split('-')[0])))
                searcher = faiss.IndexFlatL2(ret_embs.shape[1])
                searcher.add(ret_embs)
            elif ret_model == 'megamolbart':
                if 'gsk3-jnk3' not in ret_data_path:
                    ret_embs = np.load(os.path.join(project_home,
                                                    'data/gsk3jnk3/{}/actives_from_{}_megamolbart_embs.npy'.format(
                                                        ret_path_tmp,
                                                        ret_data_path.split('-')[0])))

                    if ret_subsample > 0:
                        tmp_attr = np.array(data_attr.gsk3.tolist()) + np.array(data_attr.jnk3.tolist())
                        largest_indices = np.array(
                            random.sample(range(len(data_attr)), ret_subsample))  # indices of random subset
                        data_attr = data_attr.iloc[largest_indices, :]
                        ret_smiles_all = [ret_smiles_all[s_idx] for s_idx in largest_indices]
                        ret_embs = ret_embs[largest_indices, :]

                    searcher = faiss.IndexFlatL2(ret_embs.shape[1])
                    searcher.add(ret_embs)
                else:
                    if ret_data_path == 'gsk3-jnk3-chembl':
                        ret_embs_gsk3 = np.load(os.path.join(project_home,
                                                             'data/gsk3jnk3/gsk3/actives_from_chembl_megamolbart_embs.npy'))
                        ret_embs_jnk3 = np.load(os.path.join(project_home,
                                                             'data/gsk3jnk3/jnk3/actives_from_chembl_megamolbart_embs.npy'))
                    elif ret_data_path == 'gsk3-jnk3':
                        ret_embs_gsk3 = np.load(os.path.join(project_home,
                                                             'data/gsk3jnk3/gsk3/actives_megamolbart_embs.npy'))
                        ret_embs_jnk3 = np.load(os.path.join(project_home,
                                                             'data/gsk3jnk3/jnk3/actives_megamolbart_embs.npy'))
                    assert (ret_embs_gsk3.shape[0] == len(ret_smiles_gsk3))
                    assert (ret_embs_jnk3.shape[0] == len(ret_smiles_jnk3))
                    searcher_gsk3 = faiss.IndexFlatL2(ret_embs_gsk3.shape[1])
                    searcher_gsk3.add(ret_embs_gsk3)
                    searcher_jnk3 = faiss.IndexFlatL2(ret_embs_jnk3.shape[1])
                    searcher_jnk3.add(ret_embs_jnk3)

        generated_all = []
        leads_seen_all = []

        for chunk_id in tqdm(range(n_chunks), position=0, leave=True):

            lead = leads_all[chunk_id * batch_size: (chunk_id + 1) * batch_size]
            generated_all_batch = [[] for _ in range(len(lead))]
            leads_seen_all += lead

            pl.utilities.seed.seed_everything(chunk_id)

            init_lead = lead
            if type(lead) == str:
                lead = [lead]

            start = time()
            if ret_mode == 'per-lead-tanimoto':
                retrievals_lead = []
                with Pool() as pool:
                    for l in lead:
                        sims = pool.map(partial(
                            tanimoto_dist_func,
                            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(l), 2, nBits=2048)),
                            data_attr.smiles.tolist())
                        data_attr['sim'] = sims
                        data_attr_for_this_lead = data_attr.sort_values(by='sim', ascending=False)  # rank by similarity

                        ## get retreivals
                        retrievals = data_attr_for_this_lead.smiles.tolist()[:n_retrievals]
                        retrievals_lead.append(retrievals)

                assert (len(retrievals_lead[0]) == n_retrievals)
                assert (len(retrievals_lead) == len(lead))

            elif ret_mode == 'per-lead-faiss':
                retrievals_lead = []
                batch = [{'encoder_smiles': s} for s in lead]
                batch_input = collate_fn_inference(batch)
                memory = emb_model.encode(batch_input)
                query_embs = memory[0]
                for l_idx in range(len(lead)):
                    l = lead[l_idx]
                    l_emb = query_embs[l_idx, :].reshape(1, query_embs.shape[1])
                    _, ret = searcher.search(l_emb, n_retrievals)
                    retrievals_lead.append([ret_smiles_all[r_idx] for r_idx in ret[0]])
                assert (len(retrievals_lead[0]) == n_retrievals)
                assert (len(retrievals_lead) == len(lead))

            elif ret_mode == 'per-lead-random':
                retrievals_lead = []
                for l in lead:
                    retrievals = random.sample(data_attr.smiles.tolist(), n_retrievals)
                    retrievals_lead.append(retrievals)
                assert (len(retrievals_lead[0]) == n_retrievals)
                assert (len(retrievals_lead) == len(lead))

            elif ret_mode == 'per-lead-topn':
                retrievals_lead = []
                for l in lead:
                    retrievals = data_attr.smiles.tolist()[:n_retrievals]
                    retrievals_lead.append(retrievals)
                assert (len(retrievals_lead[0]) == n_retrievals)
                assert (len(retrievals_lead) == len(lead))

            for itr in tqdm(range(1, n_repeat), position=0, leave=True):

                # get the retrieval set for the current lead, and rerank by similarity
                if ret_mode == 'per-itr-tanimoto':
                    retrievals_lead = []
                    for l in lead:
                        def tanimoto_dist_func(lead_fp, ret):
                            return DataStructs.TanimotoSimilarity(
                                lead_fp,
                                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ret), 2, nBits=2048))


                        with Pool() as pool:
                            sims = pool.map(partial(
                                tanimoto_dist_func,
                                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(l), 2, nBits=2048)),
                                data_attr.smiles.tolist())
                        data_attr['sim'] = sims
                        data_attr_for_this_lead = data_attr.sort_values(by='sim', ascending=False)  # rank by similarity

                        ## get retreivals
                        retrievals = data_attr_for_this_lead.smiles.tolist()[:n_retrievals]
                        retrievals_lead.append(retrievals)

                    assert (len(retrievals_lead[0]) == n_retrievals)
                    assert (len(retrievals_lead) == len(lead))

                elif ret_mode == 'per-itr-faiss':
                    retrievals_lead = []
                    batch = [{'encoder_smiles': s} for s in lead]
                    batch_input = collate_fn_inference_original(batch)
                    memory = emb_model.encode(batch_input)
                    query_embs = memory[0]
                    for l_idx in range(len(lead)):
                        l = lead[l_idx]
                        l_emb = query_embs[l_idx, :].reshape(1, query_embs.shape[1])
                        if 'gsk3-jnk3' not in ret_data_path:
                            _, ret = searcher.search(l_emb.cpu().numpy(), n_retrievals)
                            retrievals_lead.append([ret_smiles_all[r_idx] for r_idx in ret[0]])
                        else:
                            _, ret_gsk3 = searcher_gsk3.search(l_emb.cpu().numpy(), int(n_retrievals / 2))
                            _, ret_jnk3 = searcher_jnk3.search(l_emb.cpu().numpy(), int(n_retrievals / 2))
                            retrievals_lead.append([ret_smiles_gsk3[r_idx] for r_idx in ret_gsk3[0]] \
                                                   + [ret_smiles_jnk3[r_idx] for r_idx in ret_jnk3[0]])
                    assert (len(retrievals_lead[0]) == n_retrievals)
                    assert (len(retrievals_lead) == len(lead))

                elif ret_mode == 'per-itr-random':
                    retrievals_lead = []
                    for l in lead:
                        if 'gsk3-jnk3' not in ret_data_path:
                            retrievals = random.sample(data_attr.smiles.tolist(), n_retrievals)
                        else:
                            retrievals = []
                            retrievals += random.sample(ret_smiles_gsk3, int(n_retrievals / 2))
                            retrievals += random.sample(ret_smiles_jnk3, int(n_retrievals / 2))
                        retrievals_lead.append(retrievals)
                    assert (len(retrievals_lead[0]) == n_retrievals)
                    assert (len(retrievals_lead) == len(lead))

                # encode the lead molecule with generative model's encoder
                # batched version
                start_t = time()
                batch = [[{'encoder_smiles': lead[i], 'retrieved_smiles': retrievals_lead[i]}] * n_trials for i in
                         range(len(lead))]
                batch = [item for sublist in batch for item in sublist]
                batch_input = collate_fn_inference(batch)
                gens_, _ = wf.model.sample_molecules(batch_input, sampling_alg='random_batch_jitter',
                                                     jitter_std=jitter_std)
                gens_ = [gens_[n_trials * b:n_trials * (b + 1)] for b in
                         range(len(lead))]  # reshape to a list (leads) of lists (gens for a lead)
                batch = [{'encoder_smiles': lead[i], 'retrieved_smiles': retrievals_lead[i]} for i in range(len(lead))]
                batch_input = collate_fn_inference(batch)
                gens_greedy_, _ = wf.model.sample_molecules(batch_input, sampling_alg='greedy', jitter_std=None)

                if samp_alg == 'random_batch_jitter_and_greedy':
                    for b_idx in range(len(lead)):
                        gens_[b_idx].append(gens_greedy_[b_idx])
                        generated_all_batch[b_idx] += gens_[b_idx]
                    lead = [random.choice(g) for g in gens_]

                if samp_alg == 'greedy':
                    for b_idx in range(len(lead)):
                        generated_all_batch[b_idx].append(gens_greedy_[b_idx])
                    lead = gens_greedy_

                if samp_alg == 'random_batch_jitter':
                    for b_idx in range(len(lead)):
                        generated_all_batch[b_idx] += (gens_[b_idx])
                    lead = [g[0] for g in gens_]

            generated_all += generated_all_batch

            pd.DataFrame({
                'original_smiles': leads_seen_all,
                'gen_smiles_all': generated_all,
            }).to_csv(save_path)
