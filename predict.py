import json
import os
import pickle

import torch
from torch.utils.data import DataLoader

from datasets import NWPPredictDataset
from models import SASRec
from preprocess import task_predict_onmo
from solvers import *  # noqa: F401,F403

if __name__ == '__main__':
    # freeze_support()
    model_path = 'runs\onmo\sasrec\pths\model.pth'
    run_dir = 'runs\onmo\sasrec'
    data_path = 'roughs/onmo/predict_FC.csv' 

    # load config

    final_config_path: str = os.path.join(run_dir, 'config-lock.json')
    if not os.path.isfile(final_config_path):
        raise Exception("You need to create a `config.json` in your run directory.")

    # get and update config
    with open(final_config_path, 'r') as fp:
        config: dict = json.load(fp)

    # Initialize Solver

    solver_class = globals()[config['solver']]
    solver = solver_class(config)

    # Test result

    predict_data = task_predict_onmo(data_path)

    predict_data_tensor = DataLoader(
                NWPPredictDataset(
                    name=config["dataset"],
                    df_predict=predict_data,
                    target='predict',
                    ns='random',
                    sequence_len=config['dataloader']['sequence_len'],
                    max_num_segments=config['dataloader']['max_num_segments'],
                    use_session_token=config['dataloader']['use_session_token']
                ),
                batch_size=config['train']['batch_size'],
                shuffle=False,
                num_workers=config['envs']['CPU_COUNT'],
                pin_memory=True,
                drop_last=False
            )

    ranks = solver.predict(predict_data_tensor)

    with open('data\onmo\iid2iindex.pkl', 'rb') as fp:
        iid2iindex = pickle.load(fp)

    
    labels = []
    for rec in ranks:
        value = [i for i in iid2iindex if iid2iindex[i]==rec+1]
        labels.append(value[0])
                
      
    print(labels)
    print('test')
    
    