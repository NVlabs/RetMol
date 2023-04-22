# ---------------------------------------------------------------
# Taken from the following link as is from:
# https://github.com/NVIDIA/cheminformatics/blob/master/common/cuchemcommon/utils/logger.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_CHEMINFORMATICS).
# ---------------------------------------------------------------

import logging
import os
from datetime import datetime

from cuchemcommon.context import Context

from .sysinfo import get_machine_config, print_machine_config

BENCHMARK_FILE = '/data/benchmark.csv'

logger = logging.getLogger(__name__)


def initialize_logfile(benchmark_file=BENCHMARK_FILE):
    """Initialize benchmark file with header if needed"""

    config = get_machine_config()
    config_message = print_machine_config(config)

    if not os.path.exists(benchmark_file):
        with open(benchmark_file, 'w') as fh:
            fh.write(f'# {config_message}\n')
            fh.write('date,benchmark_type,step,time(hh:mm:ss.ms),n_molecules,n_workers,metric_name,metric_value\n')
    return benchmark_file


class MetricsLogger(object):

    def __init__(self,
                 task_name,
                 n_molecules):

        self.task_name = task_name
        self.n_molecules = n_molecules
        self.start_time = None
        self.metric_name = None
        self.metric_value = None

        self.metric_func = None
        self.metric_func_args = None
        self.metric_func_kwargs = {}

    def __enter__(self):
        self.start_time = datetime.now()

        return self

    def __exit__(self, type, value, traceback):
        context = Context()

        runtime = datetime.now() - self.start_time
        logger.info('### Runtime {} time (hh:mm:ss.ms) {}'.format(self.task_name, runtime))
        n_workers = len(context.dask_client.cluster.workers)

        if self.metric_func and context.is_benchmark:
            self.metric_value = self.metric_func(*self.metric_func_args,
                                                 **self.metric_func_kwargs)

        if self.metric_value is None:
            self.metric_name = ''
            self.metric_value = ''
        else:
            logger.info('Calculated {} is {}'.format(self.metric_name, self.metric_value))

        log_results(self.start_time, context.compute_type, self.task_name,
                    runtime,
                    n_molecules=self.n_molecules,
                    n_workers=n_workers,
                    metric_name=self.metric_name,
                    metric_value=self.metric_value,
                    benchmark_file=context.benchmark_file)


def log_results(date,
                benchmark_type,
                step,
                time,
                n_molecules,
                n_workers,
                metric_name='',
                metric_value='',
                benchmark_file=BENCHMARK_FILE):
    """Log benchmark results to a file"""

    out_list = [date, benchmark_type, step, time, n_molecules, n_workers, metric_name, metric_value]
    out_fmt = ','.join(['{}'] * len(out_list)) + '\n'

    with open(benchmark_file, 'a') as fh:
        out_string = out_fmt.format(*out_list)
        fh.write(out_string)
