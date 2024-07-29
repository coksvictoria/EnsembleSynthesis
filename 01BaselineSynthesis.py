# run_training.py

# Install required packages
import os
os.system('pip install -r requirements.txt --quiet')

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle

# Dataset name
name = 'adult'


main.py --dataname {name} --method 'ctgan' --mode train --sdv_epochs 300
main.py --dataname {name} --method 'tvae' --mode train --sdv_epochs 300
main.py --dataname {name} --method 'copulagan' --mode train --sdv_epochs 300
main.py --dataname {name} --method 'stasy' --mode train
main.py --dataname {name} --method 'stasy' --mode sample
main.py --dataname {name} --method 'tabddpm' --mode train --tabddpm_epochs 10000
main.py --dataname {name} --method 'tabddpm' --mode sample
main.py --dataname {name} --method 'tabsyn' --mode train 
main.py --dataname {name} --method 'tabsyn' --mode sample