# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from MolecularAI/MolBART.
#
# Source:
# https://github.com/MolecularAI/MolBART/blob/master/megatron_molbart/csv_data.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_MOLBART).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

# coding=utf-8

import os, sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from pysmilesutils.augment import SMILESAugmenter
from megatron.data.samplers import DistributedBatchSampler
from megatron import mpu, get_args
from util import DEFAULT_CHEM_TOKEN_START, DEFAULT_VOCAB_PATH, DEFAULT_MAX_SEQ_LEN, REGEX

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, os.path.join(project_home, 'MolBART/utils'))
from utils.tokenizer import load_tokenizer

default_tokenizer = load_tokenizer(vocab_path=DEFAULT_VOCAB_PATH,
                                   chem_token_start=DEFAULT_CHEM_TOKEN_START,
                                   regex=REGEX)


def check_seq_len(tokens, mask, max_seq_len=DEFAULT_MAX_SEQ_LEN):
    """ Warn user and shorten sequence if the tokens are too long, otherwise return original

    Args:
        tokens (List[List[str]]): List of token sequences
        mask (List[List[int]]): List of mask sequences

    Returns:
        tokens (List[List[str]]): List of token sequences (shortened, if necessary)
        mask (List[List[int]]): List of mask sequences (shortened, if necessary)
    """

    seq_len = max([len(ts) for ts in tokens])
    if seq_len > max_seq_len:
        tokens_short = [ts[:max_seq_len] for ts in tokens]
        mask_short = [ms[:max_seq_len] for ms in mask]
        return (tokens_short, mask_short)
    return (tokens, mask)


def collate_fn_inference(batch):
    """ Used by DataLoader to concatenate/collate inputs."""

    encoder_smiles = [x['encoder_smiles'] for x in batch]
    # print(encoder_smiles[0])
    enc_token_output = default_tokenizer.tokenize(encoder_smiles, mask=False, pad=True)

    enc_mask = enc_token_output['masked_pad_masks']
    enc_tokens = enc_token_output['original_tokens']

    enc_token_ids = default_tokenizer.convert_tokens_to_ids(enc_tokens)
    enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
    enc_pad_mask = torch.tensor(enc_mask,
                                dtype=torch.int64).transpose(0, 1)

    collate_output = {
        'encoder_input': enc_token_ids.cuda(),
        'encoder_pad_mask': enc_pad_mask.cuda(),
    }
    return collate_output


def collate_fn(batch):
    """ Used by DataLoader to concatenate/collate inputs."""

    encoder_smiles = [x['encoder_smiles'] for x in batch]
    decoder_smiles = [x['decoder_smiles'] for x in batch]
    enc_token_output = default_tokenizer.tokenize(encoder_smiles, mask=True, pad=True)
    dec_token_output = default_tokenizer.tokenize(decoder_smiles, pad=True)

    enc_mask = enc_token_output['masked_pad_masks']
    enc_tokens = enc_token_output['masked_tokens']
    dec_tokens = dec_token_output['original_tokens']
    dec_mask = dec_token_output['original_pad_masks']

    (enc_tokens, enc_mask) = check_seq_len(enc_tokens, enc_mask)
    (dec_tokens, dec_mask) = check_seq_len(dec_tokens, dec_mask)

    enc_token_ids = default_tokenizer.convert_tokens_to_ids(enc_tokens)
    dec_token_ids = default_tokenizer.convert_tokens_to_ids(dec_tokens)
    enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
    enc_pad_mask = torch.tensor(enc_mask, dtype=torch.int64).transpose(0, 1)
    dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
    dec_pad_mask = torch.tensor(dec_mask, dtype=torch.int64).transpose(0, 1)

    collate_output = {
        'encoder_input': enc_token_ids,
        'encoder_pad_mask': enc_pad_mask,
        'decoder_input': dec_token_ids[:-1, :],
        'decoder_pad_mask': dec_pad_mask[:-1, :],
        'target': dec_token_ids.clone()[1:, :],
        'target_pad_mask': dec_pad_mask.clone()[1:, :],
        'target_smiles': decoder_smiles,
    }

    return collate_output


class MoleculeDataset(Dataset):
    """Simple Molecule dataset that reads from a single DataFrame."""

    def __init__(self, df, split='train', zinc=False):
        """
        Args:
            df (pandas.DataFrame): DataFrame object with RDKit molecules and lengths.
        """

        if zinc:
            self.mols = df['smiles'].tolist()
        else:
            self.mols = df['canonical_smiles'].tolist()

        self.aug = SMILESAugmenter()
        val_idxs = df.index[df['set'] == 'val'].tolist()
        test_idxs = df.index[df['set'] == 'test'].tolist()
        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))
        idx_map = {'train': train_idxs, 'val': val_idxs,
                   'test': test_idxs}
        self.mols = [self.mols[idx] for idx in idx_map[split]]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mol = self.mols[idx]
        try:
            enc_smi = self.aug(mol)
        except:
            enc_smi = mol
        try:
            dec_smi = self.aug(mol)
        except:
            dec_smi = mol
        output = {'encoder_smiles': enc_smi[0], 'decoder_smiles': dec_smi[0]}
        return output


class MoleculeDataLoader(object):
    """Loads data from a csv file containing molecules."""

    def __init__(
            self,
            file_path,
            batch_size=32,
            num_buckets=20,
            num_workers=32,
            vocab_path=DEFAULT_VOCAB_PATH,
            chem_token_start=DEFAULT_CHEM_TOKEN_START,
            regex=REGEX
    ):

        path = Path(file_path)
        if path.is_dir():
            self.df = self._read_dir_df(file_path)
        else:
            self.df = pd.read_csv(path)

        # currently using the dumbest split: the first 80% are train, then the last 20% are split into valid and test setes
        if 'set' not in self.df.columns:
            set_col = []
            set_col += ['train'] * int(len(self.df) * 0.8)
            set_col += ['valid'] * int(len(self.df) * 0.1)
            set_col += ['test'] * int(len(self.df) - len(set_col))
            assert (len(set_col) == len(self.df))
            self.df['set'] = set_col

        train_dataset = MoleculeDataset(self.df, split='train', zinc=True)
        val_dataset = MoleculeDataset(self.df, split='val', zinc=True)
        self.tokenizer = load_tokenizer(vocab_path, chem_token_start, regex)

        world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        sampler = torch.utils.data.SequentialSampler(train_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                                                True, rank, world_size)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=batch_sampler, num_workers=num_workers,
                                                        pin_memory=True, collate_fn=collate_fn)

        sampler = torch.utils.data.SequentialSampler(val_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                                                True, rank, world_size)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_sampler=batch_sampler,
                                                      num_workers=num_workers, pin_memory=True,
                                                      collate_fn=collate_fn)

    def get_data(self):
        return (self.train_loader, self.val_loader)

    def _read_dir_df(self, path):
        args = get_args()
        names = os.listdir(path)
        m = len(names)
        world_size = max(mpu.get_data_parallel_world_size(), args.world_size)
        rank = max(mpu.get_data_parallel_rank(), args.rank)
        partition = int(m / world_size) + 1
        idx = partition * rank % m
        selected_names = names[idx:(idx + partition)]
        dfs = [pd.read_csv(path + '/' + f) for f in selected_names]

        zinc_df = pd.concat(dfs, ignore_index=True, copy=False)
        return zinc_df
