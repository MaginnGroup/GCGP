import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_plot_settings
from matplotlib.patches import Patch

### Tb_5_Matern12_False is fake
### Tb_5_RQ_True is fake

global_plot_settings.set_plot_settings()

# Read the CSV file containing the LML values
df = pd.read_csv('lml_values.csv')

# Function to rename kernels
def rename_kernel(kernel):
    if kernel == "Matern52":
        return r"Matérn, $\nu = 5/2$"
    elif kernel == "Matern32":
        return r"Matérn, $\nu = 3/2$"
    elif kernel == "Matern12":
        return r"Matérn, $\nu = 1/2$"
    elif kernel == "RBF":
        return "SE"
    else:
        return kernel
    
# Define your selected colors with the new color replacing mustard yellow
selected_colors = ['#6699ff', '#b1c800', '#336699', '#ff9999', '#66cc99']
hatches = ['//', None, '++', 'oo', '--']

# Create a bar chart for each thermophysical property
for property in df['Property'].unique():
    print('--------'+property)
   
    # Filter the DataFrame for the current thermophysical property
    property_df = df[df['Property'] == property]

    # Filter the DataFrame for the isotropic kernel (flag = False)
    filtered_df = property_df[property_df['Flag'] == False]

    # Get unique models and kernels
    models = filtered_df['Model'].unique()
    kernels = filtered_df['Kernel'].unique()
    renamed_kernels = [rename_kernel(kernel) for kernel in kernels]

    # Set the positions and width for the bars
    bar_width = 0.15
    index = np.arange(len(models))

    # Determine the max LML and its correspnding index, model, and kernel
    max_LML = max(filtered_df['LML'])
    min_LML = min(filtered_df['LML'])
    max_index = filtered_df['LML'].idxmax()
    max_model = filtered_df.loc[max_index, 'Model']
    max_kernel = filtered_df.loc[max_index, 'Kernel']
    max_model_idx = np.where(models == max_model)[0][0]
    max_kernel_idx = kernels.tolist().index(max_kernel)

    # Create a bar chart with all kernels
    plt.figure(figsize=(10, 4))
    
    # Loop over each kernel
    for i, kernel in enumerate(kernels):
        
        # Filter data for the specific kernel
        kernel_df = filtered_df[filtered_df['Kernel'] == kernel]
        print(kernel_df)

        # Extract LML values and corresponding model numbers
        lml_values = kernel_df['LML'].values
        model_indices = [np.where(models == model)[0][0] for model in kernel_df['Model']]
        print(model_indices)
        
        # Offset the bar positions by multiplying i by bar_width
        bars = plt.bar(index + i * bar_width, 
                       lml_values,
                       bar_width, 
                       label = renamed_kernels[i], 
                       color = selected_colors[i], 
                       hatch = hatches[i], 
                       edgecolor = 'black')
        
        # Add LML value callouts above each bar, with bold for the optimal value
        for j, (bar, value) in enumerate(zip(bars, lml_values)):
            if j == max_model_idx and i == max_kernel_idx:
                weight = 'bold'
            else:
                weight = 'normal'
            
            # Position text slightly above the bar height with a balanced offset
            plt.text(bar.get_x() + 0.5 * bar.get_width(), 
                     bar.get_height(),
                     f'{value:.2f}', 
                     ha = 'center', 
                     va = 'top', 
                     rotation = 90, 
                     fontsize = 9, 
                     fontweight = weight)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Log Evidence')
    plt.xticks(index + bar_width * (len(kernels) - 1) / 2, models)

    offset = abs(max_LML - min_LML) * 0.3
    # all LML values negative
    if max_LML <= 0 and min_LML <= 0:
        plt.ylim(min_LML - offset, 0)
    # all LML values positive
    elif max_LML >= 0 and min_LML >= 0:
        plt.ylim(0, max_LML + offset)
    # LML values both psitive and negative
    else:
        plt.ylim(min_LML - offset, max_LML + offset)
    
    # Include the custom "Optimal" patch and kernel patches with black outlines in the legend
    plt.legend(ncol = len(kernels), loc='upper center', handles=[Patch(facecolor=selected_colors[i], hatch = hatches[i], edgecolor='black', label=renamed_kernels[i]) for i in range(len(kernels))])
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plt.savefig(f'lml_bar_chart_{property}.png')
    plt.close()