import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re


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

        # Assume 'results' and 'property' are defined elsewhere in your code
        filename = os.path.join(results, property, 'model_4', 'model_summary.txt')


        # The file format includes complex data structures like NumPy arrays, 
        # which aren't easily parsed by standard tools. This complexity requires 
        # special handling, which isn't straightforward.

        # The format mixes dictionary-like structures with additional metadata, 
        # making it harder to split and parse correctly. 
        # If the format were more standardized (e.g., using JSON or YAML), parsing would be simpler.

        # Instead of using a standard serialization format like JSON, YAML, or even Python's pickle, 
        # the data is saved as a raw string with Python-specific constructs. 
        # This makes it less portable and more difficult to interpret without executing the code.

        # with open(filename, 'r') as file:
        #     data = file.read()

        # # Split the data into dictionary part and metadata part
        # dict_part, meta_part = data.split("Condition Number:")

        # # Define a regex pattern to match the dictionary entries
        # pattern = r"'?([^:]+)'?\s*:\s*(array\(.*?\))"

        # # Find all matches in the dictionary part
        # matches = re.findall(pattern, dict_part)

        # # Convert matches to a dictionary, passing 'array' from numpy to eval
        # data_dict = {key.strip(): eval(value, {"array": np.array}) for key, value in matches}

        # # Convert the dictionary to a DataFrame
        # df = pd.DataFrame.from_dict(data_dict, orient='index')

        # # Display the DataFrame
        # print(df)

        # exit()