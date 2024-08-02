import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Names of thermophysical properties
phys_property = ['Hvap', 'Pc', 'Tb', 'Tc', 'Tm', 'Vc']
# Dataset type
data_type = ['train', 'test']
# Set path to results
results = os.path.join(os.getcwd(), 'Final_Results')

## Initialize dictionaries to save results to
test_stats, train_stats = {}, {}

# Loop over thermophysical properties
for loopA, property in enumerate(phys_property):
   
   # Loop over dataset types
    for loopB, set_type in enumerate(data_type):
        
        # Read in files
        observations    = np.loadtxt(os.path.join(results, property, property + '_Y_'+ set_type +'_true.txt'))
        predictions     = np.loadtxt(os.path.join(results, property, property + '_Y_'+ set_type +'_pred.txt'))
        credible_ints   = np.loadtxt(os.path.join(results, property, property + '_Y_'+ set_type +'_pred_95CI.txt'))
        mol_weight      = np.loadtxt(os.path.join(results, property, 'model_4', set_type+'_data.csv'), skiprows = 1, usecols = 0, delimiter = ',')
       
        plt.errorbar(observations, predictions, credible_ints, ecolor='k', fmt='ro', markerfacecolor = 'w', label = '95% CI')
        plt.errorbar(mol_weight, predictions, credible_ints, ecolor='k', fmt='ko', markerfacecolor = 'w', label = '95% CI')
        plt.legend()
        plt.show()
        exit()