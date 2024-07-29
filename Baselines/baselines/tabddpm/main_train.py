import os
import argparse
import json

from baselines.tabddpm.train import train

import src


INFO_PATH='data_profile'

def main(args):

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    args.train = True

    with open(f'{INFO_PATH}/{dataname}.json', 'r') as f:
        info = json.load(f)


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

    ''' 
    Modification of configs
    '''
    print('START TRAINING')

    train(
        model_save_path=model_save_path,
        real_data_path=real_data_path,
        task_type=info['task_type'],
        model_type='mlp',
        model_params=model_params,
        T_dict=T_dict,
        steps=args.tabddpm_epochs,
        device=device
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--dataname', type = str, default = 'adult')
    parser.add_argument('--gpu', type = int, default=0)
    parser.add_argument('--tabddpm_epochs', type = int, default=10000)

    args = parser.parse_args()