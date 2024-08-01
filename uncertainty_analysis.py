import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import error_funcs as myFxns

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

        # Compute RMSE and MAE 
        mae  = myFxns.MAE(observations, predictions)
        rmse = myFxns.RMSE(observations, predictions)

        # Save metrics to relevant dictionary
        if set_type == 'train':
            train_stats[property]   = {'n':     [len(observations)], 
                                       'MAE':   mae, 
                                       'RMSE':  rmse}
        elif set_type == 'test':
            test_stats[property]    = {'n':     [len(observations)], 
                                       'MAE':   mae, 
                                       'RMSE':  rmse}

train_error = pd.DataFrame(train_stats).to_csv(os.path.join(results, 'train_error'))
test_error  = pd.DataFrame(test_stats).to_csv(os.path.join(results, 'test_error'))


# credible_int    = np.loadtxt(os.path.join(results, property, property + '_Y_'+ set_type +'_pred_95CI.txt'))


# # creating the bar plot
# plt.figure()
# plt.bar(train_df.keys(), train_df.loc['BIC'])
# plt.xlabel("Property")
# plt.ylabel("BIC")
# plt.savefig(os.path.join(results_directory, 'BIC'), dpi = 300)

# plt.figure()
# plt.bar(train_df.keys(), train_df.loc['MAE'])
# plt.xlabel("Property")
# plt.ylabel("MAE")
# plt.savefig(os.path.join(results_directory, 'MAE'), dpi = 300)

# plt.figure()
# plt.bar(train_df.keys(), train_df.loc['RMSE'])
# plt.xlabel("Property")
# plt.ylabel("RMSE")
# plt.savefig(os.path.join(results_directory, 'RMSE'), dpi = 300)


 # # Plot credible interval
        # plt.figure()
        # plt.title(set)
        # plt.errorbar(x = data, y = prediction, yerr=[cred_int, cred_int], ecolor='k', fmt='ko', markerfacecolor = 'w', label = '95% CI')
        # plt.xlabel('Experimental '+property)
        # plt.ylabel('Predicted '+property)
        # plt.plot(data, data, 'r-', label = 'Parity')
        # plt.legend()
        # plt.savefig(os.path.join(results_directory, property+'_'+set), dpi = 300)
