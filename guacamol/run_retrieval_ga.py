# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for RetMol. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

'''
example run

python run_retrieval_ga.py --benchmark_id 1 --n_retrievals 10 --n_repeat 10 --n_trials 100 --batch_size 5
'''

import os
import datetime
import json
import logging
from collections import OrderedDict
from typing import Any, Dict

import guacamol
from guacamol.guacamol.goal_directed_benchmark import GoalDirectedBenchmark, GoalDirectedBenchmarkResult
from guacamol.guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.guacamol.utils.data import get_time_string
from retrieval_ga_generator import RetrievalGAGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

project_home = os.environ['PROJECT_HOME']


def assess_goal_directed_generation(goal_directed_molecule_generator: GoalDirectedGenerator,
                                    benchmark: GoalDirectedBenchmark) -> Dict:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        goal_directed_molecule_generator: Model to evaluate
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    result = _evaluate_goal_directed_benchmarks(
        goal_directed_molecule_generator=goal_directed_molecule_generator,
        benchmark=benchmark)

    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results['guacamol_version'] = guacamol.__version__
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['results'] = vars(result)

    return benchmark_results


def _evaluate_goal_directed_benchmarks(goal_directed_molecule_generator: GoalDirectedGenerator,
                                       benchmark: GoalDirectedBenchmark
                                       ) -> GoalDirectedBenchmarkResult:
    """
    Evaluate a model with the given benchmarks.
    Should not be called directly except for testing purposes.

    Args:
        goal_directed_molecule_generator: model to assess
        benchmarks: list of benchmarks to evaluate
        json_output_file: Name of the file where to save the results in JSON format
    """

    logger.info(f'Running benchmark: {benchmark.name}')
    result = benchmark.assess_model(goal_directed_molecule_generator)
    logger.info(f'Results for the benchmark "{result.benchmark_name}":')
    logger.info(f'  Score: {result.score:.6f}')
    logger.info(f'  Execution time: {str(datetime.timedelta(seconds=int(result.execution_time)))}')
    logger.info(f'  Metadata: {result.metadata}')
    logger.info('Finished execution of the benchmarks')

    return result


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_id", default=6, type=int,
                        help='range from 0 to 16 (total 17 benchmarks, removing retrodiscovery)')
    parser.add_argument("--n_retrievals", default=10, type=int)
    parser.add_argument("--n_repeat", default=10, type=int)
    parser.add_argument("--n_trials", default=100, type=int)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--ret_mode", default='per-itr-random', type=str)
    parser.add_argument("--gen_model", default='chembl-pretrain', type=str)
    args = parser.parse_args()

    benchmark_id = args.benchmark_id  # 0
    n_retrievals = args.n_retrievals  # 10
    n_repeat = args.n_repeat  # 100
    n_trials = args.n_trials  # 100
    batch_size = args.batch_size  # 1
    ret_mode = args.ret_mode  # 'per-itr-random'
    gen_model = args.gen_model  # 'chembl-pretrain'

    # get model path
    if gen_model == 'chembl-pretrain':
        model_path = os.path.join(project_home, 'models/retmol_chembl')
    elif gen_model == 'zinc-pretrain':
        model_path = os.path.join(project_home, 'models/retmol_zinc')
    else:
        raise NotImplementedError

    # benchmarks
    benchmarks = goal_directed_benchmark_suite(version_name='v3')
    print(len(benchmarks))
    benchmark = benchmarks[benchmark_id]

    # set log directory
    res_dir = os.path.join(project_home,
                           'results/guacamol/results_retrieval_ga/{}'.format('_'.join(benchmark.name.split())))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    json_output_file = 'nret-{}_nrep-{}_ntrial-{}_bs-{}_retmode-{}_genmodel-{}.json'.format(
        n_retrievals, n_repeat,
        n_trials, batch_size, ret_mode,
        gen_model)
    with open(os.path.join(res_dir, json_output_file), 'wt') as f:
        f.write('\n')

    # init generator
    goal_directed_molecule_generator = RetrievalGAGenerator(project_home,
                                                            n_retrievals=n_retrievals, n_repeat=n_repeat,
                                                            n_trials=n_trials,
                                                            n_top_gens=1, ret_mode=ret_mode, batch_size=batch_size,
                                                            model_path=model_path)

    # generate and assess
    benchmark_results = assess_goal_directed_generation(
        goal_directed_molecule_generator=goal_directed_molecule_generator,
        benchmark=benchmark)

    # logging
    logger.info(f'Save results to file {json_output_file}')
    with open(os.path.join(res_dir, json_output_file), 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))
