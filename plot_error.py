############################

# Code written by Kyla Jones

############################


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import global_plot_settings as rcParams

# Set global plot settings
rcParams.set_plot_settings()

# Define custom colors
mysalmon = (235/255, 134/255, 100/255)
myteal   = (103/255, 185/255, 155/255)

# Define labels for the x-axis
xlabels = [
    '$\Delta H_{vap}$' + '\n[kJ $\cdot$' + ' mol' + '$^{-1}$' + ']',
    '$P_c$' + ' [bar]',
    '$T_b$' + ' [K]',
    '$T_c$' + ' [K]',
    '$T_m$' + ' [K]',
    '$V_c$' + '\n [cm' + '$^{3}$' + '$\cdot$' + ' mol' + '$^{-1}$' + ']'
]

def plot_error_bars(ax, train_file, test_file, xlabels, width=0.2, ylabel=True):
    """Plots error bars for the train and test datasets on a given axis."""
    # Load data
    train = pd.read_csv(train_file, index_col=0)
    test = pd.read_csv(test_file, index_col=0)
    
    no_cats = len(xlabels)
    x = np.arange(no_cats)
    
    # Plot MAE bars
    bars1 = ax.bar(x - 1.5*width, train.loc['MAE'], width, color=mysalmon, edgecolor='k')
    bars2 = ax.bar(x - width/2, test.loc['MAE'], width, color=mysalmon, edgecolor='k', hatch='//')
    
    # Plot RMSE bars
    bars3 = ax.bar(x + width/2, train.loc['RMSE'], width, color=myteal, edgecolor='k')
    bars4 = ax.bar(x + 1.5*width, test.loc['RMSE'], width, color=myteal, edgecolor='k', hatch='//')
    
    # Add error values on top of bars with rotation and consistent formatting
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 3, "{:.3f}".format(yval), 
                    ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Set labels and titles
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Thermophysical Property")
    if ylabel:
        ax.set_ylabel("Error")
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', labelrotation=60)

def create_custom_legend(ax):
    """Creates a custom legend for the plot."""
    # Custom legend handles
    legend_patches = [
        mpatches.Patch(color=mysalmon, label='MAE'),
        mpatches.Patch(color=myteal, label='RMSE'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Test'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Train')
    ]
    
    # Adding custom legend to the plot
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

def create_shared_yaxis_plots(train_file_1, test_file_1, train_file_2, test_file_2, xlabels, output_file, width=0.2):
    """Creates two subplots with a shared y-axis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    
    # Plot the Joback model results on the left axis with legend
    plot_error_bars(ax1, train_file_2, test_file_2, xlabels, width)
    ax1.text(-0.35, 155, "(a) JR", fontsize = 14, fontweight = 'bold')
    
    # Plot the model with discrepancy correction on the right axis with the legend
    plot_error_bars(ax2, train_file_1, test_file_1, xlabels, width, ylabel=False)
    ax2.text(-0.35, 155, "(b) GC-GP", fontsize = 14, fontweight = 'bold')
    
    # Create and add custom legend
    create_custom_legend(ax2)
    
    ax1.set_ylim(0, 170)
    # Save the figure to a file
    plt.tight_layout()
    plt.savefig(output_file)

# Create the shared y-axis plots
create_shared_yaxis_plots(
    train_file_1=os.path.join('Final_Results', 'train_error'),
    test_file_1=os.path.join('Final_Results', 'test_error'),
    train_file_2=os.path.join('Final_Results', 'train_error_jb'),
    test_file_2=os.path.join('Final_Results', 'test_error_jb'),
    xlabels=xlabels,
    output_file=os.path.join('Final_Results', 'error_bar_chart_shared_yaxis')
)
