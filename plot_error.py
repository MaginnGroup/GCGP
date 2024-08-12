import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import global_plot_settings as rcParams

# Set global plot settings
rcParams.set_plot_settings()

# Define custom colors
mysalmon = (235/255, 134/255, 100/255)
myteal   = (103/255, 185/255, 155/255)

# Define labels for the x-axis
xlabels = [
    '$H_{vap}$' + '\n [kJ ' + '$\cdot$' + ' mol' + '$^{-1}$' + ']',
    '$P_c$' + ' [bar]',
    '$T_b$' + ' [K]',
    '$T_c$' + ' [K]',
    '$T_m$' + ' [K]',
    '$V_c$' + '\n [cm' + '$^{3}$' + '$\cdot$' + ' mol' + '$^{-1}$' + ']'
]

def plot_error_bars(train_file, test_file, output_file, xlabels, width=0.2):
    """Plots error bars for the train and test datasets."""
    # Load data
    train = pd.read_csv(train_file, index_col=0)
    test = pd.read_csv(test_file, index_col=0)
    
    no_cats = len(xlabels)
    x = np.arange(no_cats)
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    
    # Plot MAE bars
    plt.bar(x - 1.5 * width, train.loc['MAE'], width, color = mysalmon, edgecolor='k')
    plt.bar(x - 0.5 * width, test.loc['MAE'], width, color = mysalmon, edgecolor='k', hatch='//')
    
    # Plot RMSE bars
    plt.bar(x + 0.5 * width, train.loc['RMSE'], width, color = myteal, edgecolor = 'k')
    plt.bar(x + 1.5 * width, test.loc['RMSE'], width, color = myteal, edgecolor = 'k', hatch = '//')
    
    # Add legends
    plt.bar(x - 1.5 * width, 0, width, color=mysalmon, edgecolor='k', label='MAE')
    plt.bar(x - 1.5 * width, 0, width, color=myteal, edgecolor='k', label='RMSE')
    plt.bar(x + 0.5 * width, 0, width, color='w', edgecolor='k', label='Train')
    plt.bar(x + 0.5 * width, 0, width, color='w', edgecolor='k', hatch='//', label='Test')
    
    # Set labels and titles
    plt.xticks(x, labels = xlabels)
    plt.xlabel("Thermophysical Property")
    plt.ylabel("Error")
    plt.legend(loc = 'upper left')
    
    # Save the plot to a file
    plt.savefig(output_file)

# Plot the first set of results
plot_error_bars(
    train_file=os.path.join('Final_Results', 'train_error'),
    test_file=os.path.join('Final_Results', 'test_error'),
    output_file=os.path.join('Final_Results', 'error_bar_chart'),
    xlabels=xlabels
)

# Plot the Joback model results
plot_error_bars(
    train_file=os.path.join('Final_Results', 'train_error_jb'),
    test_file=os.path.join('Final_Results', 'test_error_jb'),
    output_file=os.path.join('Final_Results', 'error_bar_chart_jb'),
    xlabels=xlabels
)
