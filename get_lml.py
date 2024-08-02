import os
import numpy as np

# thermophysical property
phys_property = ['Hvap', 'Pc', 'Tb', 'Tc', 'Tm', 'Vc']

# model architecture
model_arch = ['1', '2', '3', '4', '5']

# Kernel type
kernel = ['Matern12', 'Matern32', 'Matern52', 'RBF', 'RQ']

# Boolean flag?? (idk what this does)
flag = ['True', 'False']

# Path to results
results = os.path.join('kernel_sweep_code_and_results', 'kernel_sweep_all_results')

# initialize
lml = []
# loop over thermophysical property
for loopA, property in enumerate(phys_property):
    # Loop over model architecture
    for loopB, model_no in enumerate(model_arch):
        # Loop over kernels
        for loopC, kern in enumerate(kernel):
            # Loop over True/ Falses???
            for loopD, truefalse in enumerate(flag):
                file_name = 'model_summary_' + property + '_' + model_no + '_' + kern + '_' + truefalse + '.txt'
                lml = np.loadtxt(os.path.join(results, file_name), skiprows = 10, usecols=2)
                print(here)
                exit()
