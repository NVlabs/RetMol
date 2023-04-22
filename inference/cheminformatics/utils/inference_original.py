# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from cheminformatics.
#
# Source:
# https://github.com/NVIDIA/cheminformatics/blob/master/megamolbart/tests/test_megamolbart.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_CHEMINFORMATICS).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

#!/usr/bin/env python3

import os
import sys
import logging
from functools import partial
from pathlib import Path
from typing import List
from rdkit import Chem
import torch
import pandas as pd

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, os.path.join(project_home, 'MolBART/megatron_molbart'))
from megatron_bart import MegatronBART
from checkpointing import load_checkpoint, get_checkpoint_name

sys.path.insert(1, os.path.join(project_home, 'inference/cheminformatics/common'))
sys.path.insert(1, os.path.join(project_home, 'MolBART/molbart'))
sys.path.insert(1, os.path.join(project_home, 'MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism'))
sys.path.insert(1, os.path.join(project_home, 'MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism/megatron'))

from cuchemcommon.workflow import BaseGenerativeWorkflow, add_jitter
from decoder import DecodeSampler
from megatron import get_args
from megatron.initialize import initialize_megatron
from tokenizer import MolEncTokenizer
from util import (REGEX, DEFAULT_CHEM_TOKEN_START, DEFAULT_MAX_SEQ_LEN,
                  DEFAULT_VOCAB_PATH, CHECKPOINTS_DIR,
                  DEFAULT_NUM_LAYERS, DEFAULT_D_MODEL, DEFAULT_NUM_HEADS)

logger = logging.getLogger(__name__)


@add_jitter.register(torch.Tensor)
def _(embedding, radius, cnt, shape):
    if shape is not None:
        embedding = torch.reshape(embedding, (1, shape[0], shape[1])).to(embedding.device)
    permuted_emb = embedding.permute(1, 0, 2)

    distorteds = []
    for i in range(cnt):
        noise = torch.normal(0, radius, permuted_emb.shape).to(embedding.device)
        distorted = (noise + permuted_emb).permute(1, 0, 2)
        distorteds.append(distorted)

    return distorteds


class MegaMolBART(BaseGenerativeWorkflow):

    def __init__(self,
                 model_path,
                 model_ckpt_itr,
                 max_seq_len=DEFAULT_MAX_SEQ_LEN,
                 vocab_path=DEFAULT_VOCAB_PATH,
                 regex=REGEX,
                 default_chem_token_start=DEFAULT_CHEM_TOKEN_START,
                 checkpoints_dir=CHECKPOINTS_DIR,
                 num_layers=DEFAULT_NUM_LAYERS,
                 hidden_size=DEFAULT_D_MODEL,
                 num_attention_heads=DEFAULT_NUM_HEADS,
                 decoder_max_seq_len=None) -> None:
        super().__init__()

        torch.set_grad_enabled(False)  # Testing this instead of `with torch.no_grad():` context since it doesn't exit

        self.device = 'cuda'  # Megatron arg loading seems to only work with GPU
        self.min_jitter_radius = 1.0
        self.max_model_position_embeddings = max_seq_len
        self.model_path = model_path
        self.model_ckpt_itr = model_ckpt_itr

        args = {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'max_position_embeddings': self.max_model_position_embeddings,
            'tokenizer_type': 'GPT2BPETokenizer',
            'vocab_file': vocab_path,
            'load': checkpoints_dir
        }

        with torch.no_grad():
            try:
                initialize_megatron(args_defaults=args, ignore_unknown_args=True)
            except:
                pass
            args = get_args()
            self.tokenizer = self.load_tokenizer(args.vocab_file, regex, default_chem_token_start)
            self.model = self.load_model(args, self.tokenizer, decoder_max_seq_len)

    def load_tokenizer(self, tokenizer_vocab_path, regex, default_chem_token_start):
        """Load tokenizer from vocab file

        Params:
            tokenizer_vocab_path: str, path to tokenizer vocab

        Returns:
            MolEncTokenizer tokenizer object
        """

        tokenizer_vocab_path = Path(tokenizer_vocab_path)
        tokenizer = MolEncTokenizer.from_vocab_file(
            tokenizer_vocab_path,
            regex,
            default_chem_token_start)

        return tokenizer

    def load_model(self, args, tokenizer, decoder_max_seq_len=None):
        """Load saved model checkpoint

        Params:
            tokenizer: MolEncTokenizer tokenizer object
            decoder_max_seq_len: int, maximum sequence length
            args: Megatron initialized arguments

        Returns:
            MegaMolBART trained model
        """

        vocab_size = len(tokenizer)
        pad_token_idx = tokenizer.vocab[tokenizer.pad_token]

        if not decoder_max_seq_len:
            decoder_max_seq_len = args.max_position_embeddings

        sampler = DecodeSampler(tokenizer, decoder_max_seq_len)
        model = MegatronBART(
            sampler,
            pad_token_idx,
            vocab_size,
            args.hidden_size,
            args.num_layers,
            args.num_attention_heads,
            args.hidden_size * 4,
            args.max_position_embeddings,
            dropout=0.1,
            num_beams=1
        )

        checkpoint_name = get_checkpoint_name(os.path.join(project_home, 'models/megamolbart/checkpoints'), 134000)
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        model.load_state_dict(state_dict['model'])

        model = model.cuda()
        model.eval()
        return model

    def smiles2embedding(self, smiles, pad_length=None):
        """Calculate embedding and padding mask for smiles with optional extra padding

        Params
            smiles: string, input SMILES molecule
            pad_length: optional extra

        Returns
            embedding array and boolean mask
        """

        assert isinstance(smiles, str)
        if pad_length:
            assert pad_length >= len(smiles) + 2

        tokens = self.tokenizer.tokenize([smiles], pad=True)

        # Append to tokens and mask if appropriate
        if pad_length:
            for i in range(len(tokens['original_tokens'])):
                n_pad = pad_length - len(tokens['original_tokens'][i])
                tokens['original_tokens'][i] += [self.tokenizer.pad_token] * n_pad
                tokens['masked_pad_masks'][i] += [1] * n_pad

        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens['original_tokens'])).cuda().T
        pad_mask = torch.tensor(tokens['masked_pad_masks']).bool().cuda().T
        encode_input = {"encoder_input": token_ids, "encoder_pad_mask": pad_mask}

        embedding = self.model.encode(encode_input)
        torch.cuda.empty_cache()
        return embedding, pad_mask

    def inverse_transform(self, embeddings, mem_pad_mask, k=1, sanitize=True):
        mem_pad_mask = mem_pad_mask.clone()
        smiles_interp_list = []

        batch_size = 1  
        with torch.no_grad():
            for memory in embeddings:

                if isinstance(memory, list):
                    memory = torch.FloatTensor(memory).cuda()

                decode_fn = partial(self.model._decode_fn,
                                    mem_pad_mask=mem_pad_mask.type(torch.LongTensor).cuda(),
                                    memory=memory)

                mol_strs, _ = self.model.sampler.greedy_decode(decode_fn,
                                                               batch_size=batch_size,
                                                               device='cuda', )

                for smiles in mol_strs:
                    if sanitize:
                        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
                        if mol:
                            sanitized_smiles = Chem.MolToSmiles(mol)
                            smiles_interp_list.append(sanitized_smiles)
                            logger.debug(f'Sanitized SMILES {sanitized_smiles} added...')
                            break
                    smiles_interp_list.append(smiles)

        return smiles_interp_list
