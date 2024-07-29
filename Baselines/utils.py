
from baselines.sdv.main import train_ctgan
from baselines.sdv.main import train_tvae
from baselines.sdv.main import train_copulagan

from baselines.stasy.main import main as train_stasy
from baselines.tabddpm.main_train import main as train_tabddpm

from baselines.stasy.sample import main as sample_stasy
from baselines.tabddpm.main_sample import main as sample_tabddpm

from baselines.tabsyn.vae.main import main as train_vae
from baselines.tabsyn.main import main as train_tabsyn
from baselines.tabsyn.sample import main as sample_tabsyn


import argparse
import importlib
import ml_collections

def execute_function(method, mode):

    if mode == 'evaluation':
        method = 'all'

    if method == 'vae':
        mode = 'train'

    main_fn = eval(f'{mode}_{method}')

    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='tabsyn', help='Method: tabsyn or baseline.')
    parser.add_argument('--seed', type=str, default='train', help='Seed for data split.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')

    ''' configs for SDV + CTABGAN '''

    parser.add_argument('-e', '--sdv_epochs', default=300, type=int, help='Number of training epochs for CTGAN,TVAE,CopulaGAN, CTABGAN')

    # configs for traing StaSy
    parser.add_argument('--stasy_epochs', type=int, default=100, help='Number of training epochs for StaSy')

    # configs for traing TabSyn's VAE
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')
    parser.add_argument('--vae_epochs', type=int, default=100, help='Number of training epochs for VAE')

    parser.add_argument('--tabsyn_epochs', type=int, default=100, help='Number of training epochs for Tabsyn')

    # configs for TabDDPM
    parser.add_argument('--ddim', action = 'store_true', default=False, help='Whether use DDIM sampler')
    parser.add_argument('--tabddpm_epochs', type=int, default=10000, help='Number of training epochs for TabDDPM')

    # configs for sampling in general
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')
    parser.add_argument('--steps', type=int, default=50, help='NFEs.')

    args = parser.parse_args()

    return args