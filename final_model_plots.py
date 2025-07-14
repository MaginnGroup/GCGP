"""
Script to make plots comparing GC predictions with GCGP predictions for GCGP project.


Contributors: Dinis Abranches, Montana Carlozo, Barnabas Agbodekhe
"""  

import os
import warnings
import time

# Specific
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics


# =============================================================================
# Plots
# =============================================================================
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

        
        
def parity_plot(code, Y_exp, Y_gp_Pred, Y_gc_Pred, Y_gp_CI, method_number=3, split='Train', save_plot = False):
    """
    parity_plot() generates a parity plot comparing the experimental and predicted property values

    Parameters:
    code : string
        Property code
    Y_Train : numpy array
        Training set property values
    Y_Test : numpy array
        Testing set property values
    Y_Train_Pred : numpy array
        Training set predicted property values
    Y_Test_Pred : numpy array   
        Testing set predicted property values
    gpConfigs : dictionary   
        Dictionary of GP Configuration
    save_plot : bool
        Whether to save the plot or not
    disc : bool
        Whether to plot the discrepancy of the property or not
    """
    
    if method_number == 1:
        SaveName='model_1'
    elif method_number == 2:
        SaveName='model_2'
    elif method_number == 3:
        SaveName='model_3'
    elif method_number == 4:
        SaveName='model_4'
        
        
    # Pyplot Configuration
    plt.rcParams['figure.dpi']=300
    plt.rcParams['savefig.dpi']=300
    #plt.rcParams['text.usetex']=False
    #plt.rcParams['font.family']='serif'
    # plt.rcParams['font.serif']='Times New Roman'
    plt.rcParams['font.weight']='bold'
    #plt.rcParams['mathtext.rm']='serif'
    #plt.rcParams['mathtext.it']='serif:italic'
    #plt.rcParams['mathtext.bf']='serif:bold'
    #plt.rcParams['mathtext.fontset']='custom'
    plt.rcParams['axes.titlesize']=9
    plt.rcParams['axes.labelsize']=9
    plt.rcParams['xtick.labelsize']=9
    plt.rcParams['ytick.labelsize']=9
    plt.rcParams['font.size']=8.5
    plt.rcParams["savefig.pad_inches"]=0.02

 
    if code=='Tb':
        varName='T$_{b}$ /K'
    elif code=='Tm':
        varName='T$_{m}$ /K'
    elif code=='Hvap':
        varName = "$\Delta$H$_{vap}$ /kJmol$^{-1}$"
    elif code == "Vc":
        varName = 'V$_{c}$ /cm$^3$mol$^{-1}$'
    elif code == "Tc":
        varName = 'T$_{c}$ /K'
    elif code == "Pc":
        varName = 'P$_{c}$ /bar'
    else:
        varName = 'Property'
    # Predictions Scatter Plot
    # Compute metrics
    try:
        R2_gp=metrics.r2_score(Y_exp,Y_gp_Pred)
        R2_gc=metrics.r2_score(Y_exp,Y_gc_Pred)
    except:
        R2_gp = None
        R2_gc = None
    try:
        MAE_gp=metrics.mean_absolute_error(Y_exp,Y_gp_Pred)
        MAE_gc=metrics.mean_absolute_error(Y_exp,Y_gc_Pred)
    except:
        MAE_gp = None
        MAE_gc = None

    try:
        MAPE_gp=metrics.mean_absolute_percentage_error(Y_exp,Y_gp_Pred)*100
        MAPE_gc=metrics.mean_absolute_percentage_error(Y_exp,Y_gc_Pred)*100
    except:
        MAPE_gp = None
        MAPE_gc = None

    # Plot
    plt.figure(figsize=(3.0,2))
    plt.scatter(Y_exp, Y_gc_Pred, c='r', marker='x', alpha=0.7, s=10) #markersize=3,
    #plt.plot(Y_exp.flatten(), Y_gp_Pred.flatten(),'ow',markersize=3,  mec ='r', mew=0.5, zorder = 2)
    gp_err = plt.errorbar(Y_exp.flatten(), Y_gp_Pred.flatten(), yerr=Y_gp_CI.flatten(), \
             fmt='D',ecolor = 'black',color='blue', markersize=2, linewidth=0.5, zorder = 10, alpha = 0.5)
    (plotline, _, _) = gp_err
    gp_err[-1][0].set_linestyle('--') 
    plotline.set_markerfacecolor('none')
    
    lims=[np.min([plt.gca().get_xlim(),plt.gca().get_ylim()]),
        np.max([plt.gca().get_xlim(),plt.gca().get_ylim()])]
    plt.axline((lims[0],lims[0]),(lims[1],lims[1]),color='k',
            linestyle='--',linewidth=1)

    plt.xlabel('Exp. '+varName ,weight='bold')
    plt.ylabel('Pred. '+varName ,weight='bold')
    
    #plt.title(gpConfigs["Name"])
    
    if MAPE_gc != None:
        plt.text(0.99,0.13,
                'MAPE = '+'{:.2f} '.format(MAPE_gc)+"%",
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='r')
    if MAPE_gp != None:
        plt.text(0.99,0.04,
            'MAPE = '+'{:.2f} '.format(MAPE_gp)+"%",
            horizontalalignment='right',
            transform=plt.gca().transAxes,c='b') 
    if R2_gc != None:
        plt.text(0.99,0.35,'$R^2$ = '+'{:.2f}'.format(R2_gc),
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='r')
    if R2_gp != None:
        plt.text(0.99,0.26,'$R^2$ = '+'{:.2f}'.format(R2_gp),
                horizontalalignment='right',
                transform=plt.gca().transAxes,c='b')
    if save_plot == True:
        save_path = "Final_Results/" + code + "/" + SaveName +"/" + split + "Parity_Plot"
        save_fig(save_path)
    else:
        plt.show()

######################################################################################### 
# Plots generation
#########################################################################################         

code = 'Pc'    
method_number_ = 3
save_plot_ = True


dir_root = "Final_Results/" + code


Y_Train_plt = np.loadtxt(dir_root+f"/{code}_Y_train_true.txt")
Y_Test_plt = np.loadtxt(dir_root+f"/{code}_Y_test_true.txt")
Y_Train_Pred_plt = np.loadtxt(dir_root+f"/{code}_Y_train_pred.txt")
Y_Test_Pred_plt = np.loadtxt(dir_root+f"/{code}_Y_test_pred.txt")

Y_gc_Train = np.loadtxt(dir_root+f"/{code}_Y_gc_train.txt")
Y_gc_Test = np.loadtxt(dir_root+f"/{code}_Y_gc_test.txt")

Y_Train_CI_plt = np.loadtxt(dir_root+f"/{code}_Y_train_pred_95CI.txt")
Y_Test_CI_plt = np.loadtxt(dir_root+f"/{code}_Y_test_pred_95CI.txt")


parity_plot(code, Y_Train_plt, Y_Train_Pred_plt, Y_gc_Train, Y_Train_CI_plt, \
            method_number=method_number_, split='Train', save_plot=save_plot_)

parity_plot(code, Y_Test_plt, Y_Test_Pred_plt, Y_gc_Test, Y_Test_CI_plt, \
            method_number=method_number_, split='Test', save_plot=save_plot_)


######################################################################################### 

#########################################################################################  

