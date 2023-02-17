import argparse
import json
import multiprocessing
import os

import torch

from preprocess import task_prepare_onmo
from solvers import *  # noqa: F401,F403

# seed
SEED = 12345
# SEED = 23456
# SEED = 34567
# SEED = 45678
# SEED = 56789

# default
ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = 'runs'
default_config = {
    'envs': {
        'RUN_ROOT': os.path.join(ROOT, RUNS),
        'DATA_ROOT': os.path.join(ROOT, 'data'),
        'ROUGH_ROOT': os.path.join(ROOT, 'roughs'),
        'CPU_COUNT': max(8, multiprocessing.cpu_count() // 4),
        'GPU_COUNT': torch.cuda.device_count(),
    },
    'solver': 'BERT4RecSolver',
    'dataset': 'ml1m',
    'dataloader': {
        'sequence_len': 20,
        'max_num_segments': 0,
        'random_cut_prob': 0.0,
        'mlm_mask_prob': 0.15,
        'nwp_replace_prob': 0.02,
        'use_session_token': False,
        'random_seed': SEED,
    },
    'model': {
        'num_layers': 4,
        'hidden_dim': 256,
        'temporal_dim': 0,
        'num_heads': 4,
        'dropout_prob': 0.15,
        'random_seed': SEED,
    },
    'train': {
        'epoch': 400,
        'patience': 200,
        'batch_size': 256,
        'optimizer': {
            'algorithm': 'adamw',
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.05,
            'amsgrad': False,
        },
        'scheduler': {
            'algorithm': 'step',
            'step_size': 50,
            'gamma': 1.0,
        },
    },
    'metric': {
        'ks': [1, 5, 10, 20, 50],
        'pivot': 'NDCG@10',
    },
    'memo': "",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help="data folder")
    return parser.parse_args()


def update_dict_diff(base, diff):
    for key, value in diff.items():
        if isinstance(value, dict) and value:
            partial = update_dict_diff(base.get(key, {}), value)
            base[key] = partial
        else:
            base[key] = diff[key]
    return base


if __name__ == '__main__':

    # args
    args: argparse.Namespace = parse_args()
    
    # Do preprocessing
    task_prepare_onmo(args.data_dir)

    # settle dirs
    run_root: str = os.path.join(ROOT, args.data_dir)
    run_dir: str = run_root
    if not os.path.isdir(run_root):
        raise Exception(f"You need to create a `{RUNS}` directory.")
    if not os.path.isdir(run_dir):
        raise Exception("You need to create your run directory.")

    # check config file
    final_config_path: str = os.path.join(run_dir, 'config.json')
    if not os.path.isfile(final_config_path):
        raise Exception("You need to create a `config.json` in your run directory.")

    # get and update config
    config: dict = dict(default_config)
    
    partial_config_path = os.path.join(run_root, 'config.json')
    if os.path.isfile(partial_config_path):
        with open(partial_config_path, 'r') as fp:
            partial_config: dict = json.load(fp)
            update_dict_diff(config, partial_config)

    # settle config
    config['name'] = "onmo/sasrec"
    config['run_dir'] = run_dir
    
    config['envs']['RUN_ROOT'] = run_dir
    config['envs']['DATA_ROOT'] = run_dir
    config['envs']['ROUGH_ROOT'] = run_dir
    config['dataset'] = ''
    
    # lock config
    with open(os.path.join(run_dir, 'config-lock.json'), 'w') as fp:
        json.dump(config, fp, indent=4)

    # run
    solver_class = globals()[config['solver']]
    solver = solver_class(config)
    solver.solve()
