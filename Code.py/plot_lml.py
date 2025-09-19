##################################

# Code written by Kyla Jones
# Modified by Barnabas Agbodekhe

##################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import global_plot_settings
from matplotlib.patches import Patch
from copy import deepcopy
import os

#global_plot_settings.set_plot_settings()


def save_fig(save_path, ext='png', close=True):                
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    # Extract the directory and filename from the given path
    directory = os.path.split(save_path)[0]
    filename = "%s.%s" % (os.path.split(save_path)[1], ext)
    if directory == '':
        directory = '.'
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # The final path to save to
    savepath = os.path.join(directory, filename)
    # Actually save the figure
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    # Close it
    if close:
        plt.close()


# Read the CSV file containing the LML values
df = pd.read_csv('lml_values.csv')

# Function to create bar charts for each property
# Create a bar chart for each thermophysical property
def create_bar_charts(df, flag_value=False):
    #selected_colors = ['#b1c800', '#6699ff', '#336699', '#16a085', '#99e6b3', '#ffcc66']
    #selected_colors = ['#ff6666', '#6699ff', '#336699', '#16a085', '#99e6b3','#ff0000']
    #selected_colors = ['#c70039', '#ff5733', '#008080', '#40e0d0', '#4682b4', '#1e90ff']
    #selected_colors = ['#90ee90', '#d62728', '#6699ff', '#0057b7', '#ff6347' , '#2ca02c']
    selected_colors = ['#BEBEBE', '#d62728', '#6699ff', '#0057b7', '#ff6347' , 'gray']

    hatches = [None, '//', '++', 'oo', '--', 'xx']
    labels = ['(a) ' + '$\Delta H_{' + 'vap' + '}$', 
              '(b) ' + '$P_c$', 
              '(c) ' + '$T_b$', 
              '(d) ' + '$T_c$',
              '(e) ' + '$T_m$', 
              '(f) ' + '$V_c$']
    count = 0
    for property in df['Property'].unique():

        figsize_length = 12
        if property == 'Hvap' or property == 'Tm' :
            figsize_width = 5.5
        elif property == 'Tc':
            figsize_width = 4.5
        elif property == 'Vc':
            figsize_width = 7.5
        else:
            figsize_width = 3.7

        # Filter the DataFrame for the current thermophysical property
        property_df = df[df['Property'] == property]


        # Filter the DataFrame based on the flag value
        filtered_df = deepcopy(property_df)
        for i in reversed(range(len(filtered_df))):
            #row_index = filtered_df.index[i]
            if filtered_df.iloc[i, 2] != 'RQ' and filtered_df.iloc[i, 3] != flag_value:
                index_to_drop = filtered_df.index[i]
                filtered_df.drop(index=index_to_drop, inplace=True)

        filtered_df = filtered_df.sort_values(by=['Model', 'Flag'])

        models = filtered_df['Model'].unique()
        kern_flag_combo = filtered_df[['Kernel', 'Flag']].drop_duplicates()
        kern_flag_combo['Flag'] = kern_flag_combo['Flag'].replace(False, 'I')
        kern_flag_combo['Flag'] = kern_flag_combo['Flag'].replace(True, 'A')
        for i in range(len(kern_flag_combo)):
            kern_anisotropy = kern_flag_combo.apply(lambda row: f"{row['Kernel']} {row['Flag']}", axis=1).tolist()


        filtered_df["kernel_ls"] =  ""   

        for i in range(len(filtered_df)):
            row_index = filtered_df.index[i]
            if filtered_df.loc[row_index, "Kernel"] == 'Matern12':
                filtered_df.loc[row_index, "kernel_ls"] = 'Matern12 I'
            elif filtered_df.loc[row_index, "Kernel"] == 'Matern32':
                filtered_df.loc[row_index, "kernel_ls"] = 'Matern32 I'
            elif filtered_df.loc[row_index, "Kernel"] == 'Matern52':
                filtered_df.loc[row_index, "kernel_ls"] = 'Matern52 I'
            elif filtered_df.loc[row_index, "Kernel"] == 'RBF':
                filtered_df.loc[row_index, "kernel_ls"] = 'RBF I'
            elif filtered_df.loc[row_index, "Kernel"] == 'RQ':
                if filtered_df.loc[row_index, "Flag"] == False:
                    filtered_df.loc[row_index, "kernel_ls"] = 'RQ I'
                else:
                    filtered_df.loc[row_index, "kernel_ls"] = 'RQ A'

        # Set the positions and width for the bars
        bar_width = 0.15
        #index = np.arange(len(models))
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
        ratio_threshold =  20  # You can adjust this threshold as needed


        if ratio > ratio_threshold:  # Use broken axis when the ratio is large
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [1, 1]}, figsize = (figsize_length,figsize_width))
            fig.subplots_adjust(hspace=0.05)  # Adjust space between plots

            # Set limits for the broken Y-axis
            if max_LML and min_LML < 0:
                if np.abs(min_LML) < 50 * np.abs(max_LML):
                    ax1.set_ylim(max_LML * 3, 0)  # Zoomed-in upper part
                    ax2.set_ylim(1.75 * min_LML, 0.5 * min_LML)  # Zoomed-out lower part
                else:
                    ax1.set_ylim(max_LML * 2.5, 0)  # Zoomed-in upper part
                    ax2.set_ylim(1.85 * min_LML, 6.3 * max_LML)  # Zoomed-out lower part
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
            fig, ax1 = plt.subplots(figsize = (figsize_length,figsize_width))
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


        labels_ = [""] * len(kern_anisotropy)
        for i in range(len(kern_anisotropy)):
            if kern_anisotropy[i] == 'Matern12 I':
                labels_[i] = r'Matérn, $\nu = 1/2$, I'
            elif kern_anisotropy[i] == 'Matern32 I':
                labels_[i] = r'Matérn, $\nu = 3/2$, I'
            elif kern_anisotropy[i] == 'Matern52 I':
                labels_[i] = r'Matérn, $\nu = 5/2$, I'
            elif kern_anisotropy[i] == 'RBF I':
                labels_[i] = r'RBF, I'
            elif kern_anisotropy[i] == 'RQ I':
                labels_[i] = r'RQ, I'
            elif kern_anisotropy[i] == 'RQ A':
                labels_[i] = r'RQ, A'           

        # Loop over each kernel
        for i, kernel_ in enumerate(kern_anisotropy):
            # Filter data for the specific kernel
            kernel_df = filtered_df[filtered_df['kernel_ls'] == kernel_]
            # Extract LML values and corresponding model numbers
            lml_values = kernel_df['LML'].values
            model_indices = [np.where(models == model)[0][0] for model in kernel_df['Model']]
            # Offset the bar positions by multiplying i by bar_width
            bars1 = ax1.bar(index + i * bar_width, 
                            lml_values,
                            bar_width, 
                            label=labels_[i], 
                            color=selected_colors[i], 
                            hatch=hatches[i], 
                            edgecolor='black')

            bars2 = ax2.bar(index + i * bar_width, 
                            lml_values,
                            bar_width, 
                            label=labels_[i], 
                            color=selected_colors[i], 
                            hatch=hatches[i], 
                            edgecolor='black')

            # Add LML value callouts above or below each bar, depending on the value
            for j, (bar1, bar2, value) in enumerate(zip(bars1, bars2, lml_values)):
                # Check if this bar corresponds to the maximum LML value
                weight = 'normal' if (kernel_ == max_kernel and model_indices[j] == np.where(models == max_model)[0][0]) else 'normal'
                fontsize_ = 20
                rotation_ = 90
                # Set vertical alignment based on the value's sign
                va = 'bottom' if value > 0 else 'top'
                scale = bar1.get_height() * 0.05 if value > 0 else 0

                if property == 'Tc':
                    formatted_value = f'{value:.1f}'
                else:
                    formatted_value = f'{value:.0f}'
                if value > ax2.get_ylim()[1]:  # Only add to ax1
                    ax1.text(bar1.get_x() + 0.5 * bar1.get_width(), 
                             bar1.get_height() + scale,
                             formatted_value, 
                             ha='center', 
                             va=va, 
                             rotation=rotation_, 
                             fontsize=fontsize_ , 
                             fontweight=weight)
                elif value < ax1.get_ylim()[0]:  # Only add to ax2
                    ax2.text(bar2.get_x() + 0.5 * bar2.get_width(), 
                             bar2.get_height() + scale,
                             formatted_value, 
                             ha='center', 
                             va=va, 
                             rotation=rotation_, 
                             fontsize=fontsize_, 
                             fontweight=weight)
                else:  # Value falls within both ranges, decide based on closeness to limits
                    if value >= ax1.get_ylim()[0]:
                        ax1.text(bar1.get_x() + 0.5 * bar1.get_width(), 
                                 bar1.get_height() + scale,
                                 formatted_value, 
                                 ha='center', 
                                 va=va, 
                                 rotation=rotation_, 
                                 fontsize=fontsize_, 
                                 fontweight=weight)
                    else:
                        ax2.text(bar2.get_x() + 0.5 * bar2.get_width(), 
                                 bar2.get_height() + scale,
                                 formatted_value, 
                                 ha='center', 
                                 va=va, 
                                 rotation=rotation_, 
                                 fontsize=fontsize_, 
                                 fontweight=weight)

        # Add labels and title
        xtick_size = 24
        ytick_size = 24
        font_size_ = 36
        label_size_ = 20
        ax1.tick_params(axis='y', labelsize=ytick_size)
        ax2.tick_params(axis='y', labelsize=ytick_size)

        ax1.set_ylabel('LML', fontsize=font_size_, fontweight='bold') 
        if ax1 != ax2:
            ax1.yaxis.set_label_coords(-0.1, -0.25) 

        #plt.ylabel('LML', fontsize=ytick_size)
        if property == 'Vc':
            plt.xlabel('Model', fontsize=font_size_, fontweight='bold')
            plt.xticks(index + bar_width * (len(kern_anisotropy) - 1) / 2, models, fontsize=xtick_size)
        else:
            plt.xticks(index + bar_width * (len(kern_anisotropy) - 1) / 2, ['','','',''], fontsize=xtick_size)

        if property == 'Hvap':
            ax1.legend(ncol=3, loc=(0.04, 1.02), fontsize=label_size_, handles=[Patch(facecolor=selected_colors[i], hatch=hatches[i], edgecolor='black', label   
            =labels_[i]) for i in range(len(labels_))])
        plt.tight_layout()
        if property == 'Vc':
            ax2.text(-0.25, min_LML + 1.85*offset, labels[count], fontweight='bold', fontsize=font_size_)
        else:
            ax2.text(-0.25, min_LML - 0.85*offset, labels[count], fontweight='bold', fontsize=font_size_)
            
        # Save the plot as a PNG file
        save_path = "LML_plots/" + f'LML_plot_{property}'
        save_fig(save_path)
        count += 1

# Call the function with the desired flag value (True or False)
create_bar_charts(df, flag_value=False)
