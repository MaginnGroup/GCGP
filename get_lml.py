import os
import numpy as np
import pandas as pd

# Thermophysical property
phys_property = ['Hvap', 'Pc', 'Tb', 'Tc', 'Tm', 'Vc']

# Model architecture
model_arch = ['1', '2', '3', '4', '5']

# Kernel type
kernel = ['Matern12', 'Matern32', 'Matern52', 'RBF', 'RQ']

# Boolean flag
flag = ['True', 'False']

# Path to results
results = os.path.join('kernel_sweep_code_and_results', 'kernel_sweep_all_results')

# Initialize list to store LML values
lml_data = []

# Loop over thermophysical property
for loopA, property in enumerate(phys_property):
    # Loop over model architecture
    for loopB, model_no in enumerate(model_arch):
        # Loop over kernels
        for loopC, kern in enumerate(kernel):
            # Loop over True/Falses
            for loopD, truefalse in enumerate(flag):
                # Construct file name
                file_name = f'model_summary_{property}_{model_no}_{kern}_{truefalse}.txt'
                # Ignore the files that don't exist
                if file_name in [
                    'model_summary_Tb_5_Matern12_False.txt',
                    'model_summary_Tb_5_RQ_True.txt',
                    'model_summary_Tc_2_Matern12_True.txt'
                ]:
                    continue
                try:
                    # Read in dataframe
                    df = pd.read_csv(os.path.join(results, file_name), skiprows=3, index_col=0, names=['Values'], delimiter=':')
                    # Extract LML
                    log_marginal_likelihood = df.loc[df.index.str.strip() == 'Log-marginal Likelihood', 'Values'].values[0]
                    # Append to list as a dictionary
                    lml_data.append({
                        'Property': property,
                        'Model': model_no,
                        'Kernel': kern,
                        'Flag': truefalse,
                        'LML': log_marginal_likelihood
                    })
                except Exception as e:
                    print(f"Failed to process file {file_name}: {e}")

# Convert list of dictionaries to DataFrame
lml_df = pd.DataFrame(lml_data)

# Save to CSV
lml_df.to_csv('lml_values.csv', index=False)

print("LML values saved to lml_values.csv")
