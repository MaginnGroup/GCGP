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

plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14

 # Pyplot Configuration
#plt.rcParams['figure.dpi']=300
#plt.rcParams['savefig.dpi']=300
##plt.rcParams['text.usetex']=False
##plt.rcParams['font.family']='serif'
## plt.rcParams['font.serif']='Times New Roman'
#plt.rcParams['font.weight']='bold'
##plt.rcParams['mathtext.rm']='serif'
##plt.rcParams['mathtext.it']='serif:italic'
##plt.rcParams['mathtext.bf']='serif:bold'
##plt.rcParams['mathtext.fontset']='custom'
#plt.rcParams['axes.titlesize']=9
#plt.rcParams['axes.labelsize']=9
#plt.rcParams['xtick.labelsize']=9
#plt.rcParams['ytick.labelsize']=9
#plt.rcParams['font.size']=8.5
#plt.rcParams["savefig.pad_inches"]=0.02



# Define custom colors
mysalmon = (235/255, 134/255, 100/255)
myteal   = (103/255, 185/255, 155/255)

# Define labels for the x-axis
#xlabels = [
#    '$\Delta H_{vap}$' + '\n[kJ $\cdot$' + ' mol' + '$^{-1}$' + ']',
#    '$P_c$' + ' [bar]',
#    '$T_b$' + ' [K]',
#    '$T_c$' + ' [K]',
#    '$T_m$' + ' [K]',
#    '$V_c$' + '\n [cm' + '$^{3}$' + '$\cdot$' + ' mol' + '$^{-1}$' + ']'
#]


xlabels = [
    "$\Delta$H$_{vap}$ \n/kJmol$^{-1}$",
    'P$_{c}$ /bar',
    'T$_{b}$ /K',
    'T$_{c}$ /K',
    'T$_{m}$ /K',
    'V$_{c}$ \n/cm$^3$mol$^{-1}$'
]

def plot_error_bars(ax, train_file, test_file, xlabels, width=0.2, ylabel=True):
    """Plots error bars for the train and test datasets on a given axis."""
    # Load data
    train = pd.read_csv(train_file, index_col=0)
    test = pd.read_csv(test_file, index_col=0)
    
    no_cats = len(xlabels)
    x = np.arange(no_cats)
    
    # Plot MAE bars
    #bars1 = ax.bar(x - 1.5*width, train.loc['MAE'], width, color='#0057b7', edgecolor='k')
    bars1 = ax.bar(x - 1.5*width, train.loc['MAE'], width, color='#6699ff', edgecolor='k')
    bars2 = ax.bar(x - width/2, test.loc['MAE'], width, color='#6699ff', edgecolor='k', hatch='//')
    
    # Plot RMSE bars
    bars3 = ax.bar(x + width/2, train.loc['RMSE'], width, color='red', edgecolor='k')
    bars4 = ax.bar(x + 1.5*width, test.loc['RMSE'], width, color='red', edgecolor='k', hatch='//')
    
    # Add error values on top of bars with rotation and consistent formatting
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 3, "{:.2f}".format(yval), 
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
        mpatches.Patch(color='#6699ff', label='MAE'),
        mpatches.Patch(color='red', label='RMSE'),
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
    ax1.text(-0.35, 155, "(a) JR GC", fontsize = 14, fontweight = 'bold')
    
    # Plot the model with discrepancy correction on the right axis with the legend
    plot_error_bars(ax2, train_file_1, test_file_1, xlabels, width, ylabel=False)
    ax2.text(-0.35, 155, "(b) GCGP", fontsize = 14, fontweight = 'bold')
    
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





