import os
import numpy as np
import pandas as pd
import error_funcs as myFxns

# Names of thermophysical properties
phys_property = ['Hvap', 'Pc', 'Tb', 'Tc', 'Tm', 'Vc']
# Dataset type
data_type = ['train', 'test']
# Set path to results
results = os.path.join(os.getcwd(), 'Final_Results')

## Initialize dictionaries to save results to
test_stats, train_stats = {}, {}
test_stats_jb, train_stats_jb = {}, {}

# Loop over thermophysical properties
for loopA, property in enumerate(phys_property):
   
   # Loop over dataset types
    for loopB, set_type in enumerate(data_type):
        
        # Read in files
        observations    = np.loadtxt(os.path.join(results, property, property + '_Y_'+ set_type +'_true.txt'))
        predictions     = np.loadtxt(os.path.join(results, property, property + '_Y_'+ set_type +'_pred.txt'))
        predictions_jb  = np.loadtxt(os.path.join(results, property, 'model_4', set_type + '_data.csv'), skiprows = 1, delimiter=',')[:,1]
        
        # Compute RMSE and MAE for GP model
        mae  = myFxns.MAE(observations, predictions)
        rmse = myFxns.RMSE(observations, predictions)

        # Compute RMSE and MAE for Joback model
        mae_jb  = myFxns.MAE(observations, predictions_jb)
        rmse_jb = myFxns.RMSE(observations, predictions_jb)

        # Save metrics to relevant dictionary
        if set_type == 'train':
            train_stats[property]   = {'n':     len(observations), 
                                       'MAE':   mae, 
                                       'RMSE':  rmse}
            
            train_stats_jb[property] = {'n':     len(observations), 
                                       'MAE':   mae_jb, 
                                       'RMSE':  rmse_jb}
        elif set_type == 'test':
            test_stats[property]    = {'n':     len(observations), 
                                       'MAE':   mae, 
                                       'RMSE':  rmse}
            
            test_stats_jb[property] = {'n':     len(observations), 
                                       'MAE':   mae_jb, 
                                       'RMSE':  rmse_jb}

train_error = pd.DataFrame(train_stats).to_csv(os.path.join(results, 'train_error'))
test_error  = pd.DataFrame(test_stats).to_csv(os.path.join(results, 'test_error'))

train_error_jb = pd.DataFrame(train_stats_jb).to_csv(os.path.join(results, 'train_error_jb'))
test_error_jb  = pd.DataFrame(test_stats_jb).to_csv(os.path.join(results, 'test_error_jb'))