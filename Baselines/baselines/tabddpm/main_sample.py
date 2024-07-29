import os
import argparse
from baselines.tabddpm.sample import sample

import math
import pandas as pd
import json
import src

INFO_PATH='data_profile'

def main(args):
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    sample_save_path = f'synthetic/{dataname}'

    print(sample_save_path)

    real_data_path = f'data/{dataname}'
    real = pd.read_csv(real_data_path+'/train.csv')

    num_samples=real.shape[0]
    batch_size= int(math.ceil(num_samples / 100.0)) * 100


    T_dict={'seed': 0, 
   'normalization': 'quantile', 
   'num_nan_policy': 'mean',
    'cat_nan_policy': None, 
    'cat_min_frequency': None, 
    'cat_encoding': None, 
    'y_policy': 'default'}

    model_params={'num_classes': 2, 
    'is_y_cond': False, 
    'rtdl_params': {'d_layers': [1024, 2048, 2048, 1024], 
    'dropout': 0.0}}


    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)
    
    args.train = True

    with open(f'{INFO_PATH}/{dataname}.json', 'r') as f:
        info = json.load(f)
    ''' 
    Modification of configs
    '''
    print('START SAMPLING')
    
    sample(
        num_samples=num_samples,
        batch_size=batch_size,
        disbalance=None,
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        task_type=info['task_type'],
        model_type='mlp',
        model_params=model_params,
        T_dict=T_dict,
        device=device,
        ddim=args.ddim,
        steps=args.steps,
        save_path = args.save_path
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tabddpm')
    parser.add_argument('--dataname', type = str, default = 'adult')
    parser.add_argument('--gpu', type = int, default=0)
    parser.add_argument('--ddim', action = 'store_true', default = False, help='Whether to use ddim sampling.')
    parser.add_argument('--steps', type=int, default = 100)

    args = parser.parse_args()
