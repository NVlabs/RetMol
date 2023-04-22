# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from guacamol benchmark.
#
# Source:
# https://github.com/BenevolentAI/guacamol/blob/master/guacamol/goal_directed_generator.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_GUACAMOL).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------
import os
import sys
from abc import ABCMeta, abstractmethod
from typing import List, Optional
import pytorch_lightning as pl
from guacamol.guacamol.scoring_function import ScoringFunction

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, project_home + '/inference')
from inference import MegaMolBART


class GoalDirectedGenerator(metaclass=ABCMeta):
    """
    Interface for goal-directed molecule generators.
    """

    def __init__(self, model_path, ret_data_path, 
                model_ckpt_itr=50000, max_mol_len=200):
        '''
        my defined initialization
        '''
        self.wf = MegaMolBART(model_path=model_path, model_ckpt_itr=model_ckpt_itr, decoder_max_seq_len=max_mol_len)
        self.ret_dataset = None


    @abstractmethod
    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]], benchmark_name: str):
        """
        Given an objective function, generate molecules that score as high as possible.

        Args:
            scoring_function: scoring function
            number_molecules: number of molecules to generate
            starting_population: molecules to start the optimization from (optional)
            benchmark_name: benchmark name

        Returns:
            A list of SMILES strings for the generated molecules.
        """
        pl.utilities.seed.seed_everything(1234)
