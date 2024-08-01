import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Names of thermophysical properties
phys_property = ['Hvap', 'Pc', 'Tb', 'Tc', 'Tm', 'Vc']
# Dataset type
set_type = ['train', 'pred']
# Set path to results
results_directory = os.path.join(os.getcwd(), 'Final_Results')

def compute_mae(data, prediction):
    mae = 1 / len(data) * sum(abs(data - prediction))
    return mae

def compute_rmse(data, prediction):
    mse = 1 / len(data) * sum((data - prediction) ** 2)
    return np.sqrt(mse)

def compute_bic(data, log_likelihood, no_params = 4):
    return 2 * no_params * np.log(len(data)) - 2 * log_likelihood

# Initialize dictionary
test_stats = {}
train_stats = {}

k = 0
for property in phys_property:
    k += 1
    for set in set_type:
        
        # Set up path to relevant files
        data_file = os.path.join(data_directory, property + '_Y_'+ set +'_true.txt')
        prediction_file = os.path.join(data_directory, property + '_Y_'+ set +'_pred.txt')
        cred_int_file = os.path.join(data_directory, property + '_Y_'+ set +'_pred_95CI.txt')
        
        # Read in relevant files
        data = np.loadtxt(data_file)
        prediction = np.loadtxt(prediction_file)
        cred_int = np.loadtxt(cred_int_file)

        # Compute RMSE and MAE
        mae = compute_mae(data, prediction)
        rmse = compute_rmse(data, prediction)

        # Plot credible interval
        plt.figure()
        plt.title(set)
        plt.errorbar(x = data, y = prediction, yerr=[cred_int, cred_int], ecolor='k', fmt='ko', markerfacecolor = 'w', label = '95% CI')
        plt.xlabel('Experimental '+property)
        plt.ylabel('Predicted '+property)
        plt.plot(data, data, 'r-', label = 'Parity')
        plt.legend()
        plt.savefig(os.path.join(results_directory, property+'_'+set), dpi = 300)

        # Save metrics to relevant dictionary
        if set == 'train':
            bic = compute_bic(data, log_likelihood[k - 1])
            train_stats[property] = {'n': len(data), 'MAE': mae, 'RMSE': rmse, 'BIC': bic}
        elif set == 'test':
            test_stats[property] = {'n': len(data), 'MAE': mae, 'RMSE': rmse}

print('============================Training set===========================')
train_df = pd.DataFrame(train_stats)  
print(train_df)

print('============================Testing set============================')
test_df = pd.DataFrame(test_stats)
print(test_df)


# creating the bar plot
plt.figure()
plt.bar(train_df.keys(), train_df.loc['BIC'])
plt.xlabel("Property")
plt.ylabel("BIC")
plt.savefig(os.path.join(results_directory, 'BIC'), dpi = 300)

plt.figure()
plt.bar(train_df.keys(), train_df.loc['MAE'])
plt.xlabel("Property")
plt.ylabel("MAE")
plt.savefig(os.path.join(results_directory, 'MAE'), dpi = 300)

plt.figure()
plt.bar(train_df.keys(), train_df.loc['RMSE'])
plt.xlabel("Property")
plt.ylabel("RMSE")
plt.savefig(os.path.join(results_directory, 'RMSE'), dpi = 300)

