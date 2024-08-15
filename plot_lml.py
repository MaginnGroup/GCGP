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
labels = ['(a) ' + '$\Delta H_{' + 'vap' + '}$', 
          '(b) ' + '$P_c$', 
          '(c) ' + '$T_b$', 
          '(d) ' + '$T_c$',
          '(e) ' + '$T_m$', 
          '(f) ' + '$V_c$']

count = 0
# Create a bar chart for each thermophysical property
for property in df['Property'].unique():

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

    # Determine the max and min LML values
    max_LML = max(filtered_df['LML'])
    min_LML = min(filtered_df['LML'])
    offset = abs(max_LML - min_LML) * 0.85

    # Calculate the ratio between max and min LML values
    if min_LML != 0:  # To avoid division by zero
        ratio = abs(min_LML / max_LML)
    else:
        ratio = float('inf')  # Infinite ratio if min_LML is zero

    # Use a threshold for the ratio to decide whether to use a broken axis
    ratio_threshold =  6  # You can adjust this threshold as needed

    if ratio > ratio_threshold:  # Use broken axis when the ratio is large
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [1, 1]}, figsize = (10,2.5))
        fig.subplots_adjust(hspace=0.05)  # Adjust space between plots

        # Set limits for the broken Y-axis
        if max_LML and min_LML < 0:
            ax1.set_ylim(max_LML * 3, 0)  # Zoomed-in upper part
            ax2.set_ylim(1.75 * min_LML, 0.5 * min_LML)  # Zoomed-out lower part
        else:
            exit('no code available for this case')

        # Break marks
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(axis='x', which='both', bottom=False)
        ax2.tick_params(axis='x', which='both', top=False)
        d = .015  # Proportional size of break marks
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)        # Top diagonal marks
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)  # Switch to bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom diagonal marks
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
    else:  # Use standard axis
        fig, ax1 = plt.subplots(figsize=(10, 2.5))
        ax2 = ax1

        # Set Y-axis limits as per conditions
        if max_LML <= 0 and min_LML <= 0:
            ax1.set_ylim(min_LML - offset, 0)
        elif max_LML >= 0 and min_LML >= 0:
            ax1.set_ylim(0, max_LML + offset)
        else:
            ax1.set_ylim(min_LML - offset, max_LML + offset)

    # Identify the index of the maximum LML value for the entire property
    max_index = filtered_df['LML'].idxmax()
    max_model = filtered_df.loc[max_index, 'Model']
    max_kernel = filtered_df.loc[max_index, 'Kernel']

    # Loop over each kernel
    for i, kernel in enumerate(kernels):
        
        # Filter data for the specific kernel
        kernel_df = filtered_df[filtered_df['Kernel'] == kernel]
        
        ## Manually make this zero since the filename cannot be reproduced manually
        if kernel == 'Matern12' and property == 'Tb':
            kernel_df.loc[-1] = ['Tb', 5, 'Matern12', False, 0]

        # Extract LML values and corresponding model numbers
        lml_values = kernel_df['LML'].values
        model_indices = [np.where(models == model)[0][0] for model in kernel_df['Model']]

        # Offset the bar positions by multiplying i by bar_width
        bars1 = ax1.bar(index + model_indices[i] * bar_width, 
                        lml_values,
                        bar_width, 
                        label=renamed_kernels[i], 
                        color=selected_colors[i], 
                        hatch=hatches[i], 
                        edgecolor='black')
        
        bars2 = ax2.bar(index + model_indices[i] * bar_width, 
                        lml_values,
                        bar_width, 
                        label=renamed_kernels[i], 
                        color=selected_colors[i], 
                        hatch=hatches[i], 
                        edgecolor='black')
        
        # Add LML value callouts above or below each bar, depending on the value
        for j, (bar1, bar2, value) in enumerate(zip(bars1, bars2, lml_values)):
            # Check if this bar corresponds to the maximum LML value
            weight = 'bold' if (kernel == max_kernel and model_indices[j] == np.where(models == max_model)[0][0]) else 'normal'
            
            # Set vertical alignment based on the value's sign
            va = 'bottom' if value > 0 else 'top'
            scale = bar1.get_height() * 0.05 if value > 0 else 0
            
            if value > ax2.get_ylim()[1]:  # Only add to ax1
                ax1.text(bar1.get_x() + 0.5 * bar1.get_width(), 
                         bar1.get_height() + scale,
                         f'{value:.2f}', 
                         ha='center', 
                         va=va, 
                         rotation=90, 
                         fontsize=8, 
                         fontweight=weight)
            elif value < ax1.get_ylim()[0]:  # Only add to ax2
                ax2.text(bar2.get_x() + 0.5 * bar2.get_width(), 
                         bar2.get_height() + scale,
                         f'{value:.2f}', 
                         ha='center', 
                         va=va, 
                         rotation=90, 
                         fontsize=8, 
                         fontweight=weight)
            else:  # Value falls within both ranges, decide based on closeness to limits
                if value >= ax1.get_ylim()[0]:
                    ax1.text(bar1.get_x() + 0.5 * bar1.get_width(), 
                             bar1.get_height() + scale,
                             f'{value:.2f}', 
                             ha='center', 
                             va=va, 
                             rotation=90, 
                             fontsize=8, 
                             fontweight=weight)
                else:
                    ax2.text(bar2.get_x() + 0.5 * bar2.get_width(), 
                             bar2.get_height() + scale,
                             f'{value:.2f}', 
                             ha='center', 
                             va=va, 
                             rotation=90, 
                             fontsize=8, 
                             fontweight=weight)

    # Add labels and title
    if property == 'Vc':
        plt.xlabel('Model')
        plt.xticks(index + bar_width * (len(kernels) - 1) / 2, models)
    else:
        plt.xticks(index + bar_width * (len(kernels) - 1) / 2, ['','','','',''])
    ax1.set_ylabel('Log Evidence')

    if property == 'Hvap':
        ax1.legend(ncol=len(kernels), loc=(0.02, 1.01), handles=[Patch(facecolor=selected_colors[i], hatch=hatches[i], edgecolor='black', label=renamed_kernels[i]) for i in range(len(kernels))])
    plt.tight_layout()
    ax2.text(-0.25, min_LML - 0.85*offset, labels[count], fontweight='bold', fontsize=15)
    
    # Save the plot as a PNG file
    plt.savefig(f'lml_bar_chart_{property}.png')
    plt.close()
    count += 1
