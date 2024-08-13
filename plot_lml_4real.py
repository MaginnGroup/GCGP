import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_plot_settings
from matplotlib.patches import Patch

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

# Create a bar chart for each thermophysical property
for property in df['Property'].unique():
   
    # Filter the DataFrame for the current thermophysical property
    property_df = df[df['Property'] == property]

    # Filter the DataFrame for the isotropic kernel (flag = False)
    filtered_df = property_df[property_df['Flag'] == False]

    # Get unique models and kernels
    models = filtered_df['Model'].unique()
    kernels = filtered_df['Kernel'].unique()
    
    # Rename the kernels using the function
    renamed_kernels = [rename_kernel(kernel) for kernel in kernels]
    
    # Set the positions and width for the bars
    bar_width = 0.15
    index = np.arange(len(models))
    
    # Define your selected colors with the new color replacing mustard yellow
    selected_colors = ['#b1c800', '#336699', '#6699ff', '#ff9999', '#66cc99']
    
    # Create a bar chart for each kernel
    plt.figure(figsize=(10, 4))
    
    for i, kernel in enumerate(kernels):
        # Filter data for the specific kernel
        kernel_df = filtered_df[filtered_df['Kernel'] == kernel]
        
        # Extract LML values and corresponding model numbers
        lml_values = kernel_df['LML'].values
        model_indices = [np.where(models == model)[0][0] for model in kernel_df['Model']]
        
        # Offset the bar positions by multiplying i by bar_width
        bars = plt.bar(index + i * bar_width, lml_values, bar_width, label=rename_kernel(kernel), 
                       color=selected_colors[i], edgecolor='black')  # Add black edgecolor
        
        # Add LML value callouts above each bar
        for bar, value in zip(bars, lml_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value:.2f}', ha='center', va='bottom', fontsize=9, color='black')

    # Highlight the bar with the maximum LML value only once
    max_index = filtered_df['LML'].idxmax()
    max_model = filtered_df.loc[max_index, 'Model']
    max_kernel = filtered_df.loc[max_index, 'Kernel']
    max_model_idx = np.where(models == max_model)[0][0]
    max_kernel_idx = kernels.tolist().index(max_kernel)
    
    # Plot the max value in the same color but with a hatch and black outline
    plt.bar(index[max_model_idx] + max_kernel_idx * bar_width,
            filtered_df.loc[max_index, 'LML'],
            bar_width, color=selected_colors[max_kernel_idx], hatch='*', edgecolor='black')

    # Create a custom legend entry for the "Optimal" bar with a black outline
    optimal_patch = Patch(facecolor='white', edgecolor='black', hatch='*', label='Optimal')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Log Evidence')
    plt.xticks(index + bar_width * (len(kernels) - 1) / 2, models)
    
    # Include the custom "Optimal" patch and kernel patches with black outlines in the legend
    plt.legend(loc='lower right', handles=[Patch(facecolor=selected_colors[i], edgecolor='black', label=renamed_kernels[i]) for i in range(len(kernels))] + [optimal_patch])
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plt.savefig(f'lml_bar_chart_{property}.png')
    plt.close()
    exit()
