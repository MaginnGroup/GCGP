# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt



def plot_perfect_parity(X, Y, num_std, plot_error_bounds=True):
    max_x = np.max(X)
    max_y = np.max(Y)
    max_data = np.max([max_x, max_y])
    min_x = np.min(X)
    min_y = np.min(Y)
    min_data = np.min([min_x, min_y])
    z_lo = min_data - 0.1 * (np.abs(min_data))
    z_hi = max_data + 0.1 * (np.abs(max_data))
    z_data = np.linspace(z_lo, z_hi, 100)
    a_data = z_data
    plt.plot(a_data, z_data, 'r--')
    if plot_error_bounds:
        mean_y = np.mean(Y)
        std_y = np.std(Y)
        error_bound =num_std*std_y
    return std_y
    



code_list=['Tb', 'Tm','Hvap', 'Vc', 'Pc', 'Tc']
save_plot = True
num_std = 2
for i, prop in enumerate(code_list):
    dbPath=""
    # Property Code
    code=prop 
    if save_plot:
        dir_root = f"./Data_Viz_and_Outlier_Figs/{code}/"
        os.makedirs(dir_root, exist_ok=True)
    # Load data
    db=pd.read_csv(os.path.join(dbPath,code+'_prediction_data.csv'))
    db=db.dropna()
    col_names = db.columns.tolist()[2:]
    X=db.iloc[:,2:-1].copy().to_numpy('float')
    Y_exp=db.iloc[:,-1].copy().to_numpy('float')
    Y_gc = X[:,-1].reshape(-1,1)
    Y_disc = X[:,-1]-Y_exp
    mol_wt = X[:,0].reshape(-1,1)

    if code == 'Tm':
        ax_lab = 'T$_m$'
        units = '/K'
    elif code == 'Tb':
        ax_lab = 'T$_b$'
        units = '/K'
    elif code == 'Tc':
        ax_lab = 'T$_c$'
        units = '/K'
    elif code == 'Vc':
        ax_lab = 'V$_c$'
        units = '/cm$^3$mol$^{-1}$'
    elif code == 'Pc':
        ax_lab = 'P$_c$'
        units = '/bar'
    elif code == 'Hvap':
        ax_lab = '$\Delta$H$_{vap}$'
        units = '/kJmol$^{-1}$'   

    MW_units = '/gmol$^{-1}$'


    plt.rcParams['figure.dpi'] = 300

    fig, ax = plt.subplots()
    twoD_scatter = ax.scatter(Y_gc, Y_exp, marker='o', edgecolors='blue', facecolors='none', alpha=0.9, \
              c=X[:,0], cmap='viridis', s=100, edgecolor='k')
    plot_perfect_parity(X[:,1], Y_exp, num_std, True)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    ax.minorticks_on()
    ax.tick_params(axis='x', labelrotation=45)
    plt.xlabel('GC '+ax_lab+' '+units, fontsize=20, weight='bold')
    plt.ylabel('Exp. '+ax_lab+' '+units, fontsize=20, weight='bold')

    cbar = plt.colorbar(twoD_scatter)
    cbar.set_label('MW '+MW_units, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    if save_plot:
        fig.savefig(dir_root+"2D_Exp_vs_GC_plot.png", bbox_inches='tight')
    plt.show()


    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    threeD_scatter = ax.scatter(Y_gc, mol_wt, Y_disc, c=Y_disc, cmap='viridis', s=100)
    cbar = plt.colorbar(threeD_scatter, shrink=0.5, aspect=20, pad=0.07)
    cbar.set_label(ax_lab+' '+'Disc. '+units, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax.set_xlabel('GC '+ax_lab+' '+units, fontsize=20, weight='bold', labelpad=20)
    ax.set_ylabel('MW '+MW_units, fontsize=20, weight='bold', labelpad=16)
    ax.set_zlabel(ax_lab+' '+'Disc. '+units, fontsize=20, weight='bold', labelpad=16)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    ax.grid(False)
    ax.tick_params(axis='x', labelrotation=15)
    ax.tick_params(axis='y', labelrotation=-15)

    plt.tight_layout()
    if save_plot:
        fig.savefig(dir_root+"3D_GC_vs_MW_vs_Disc_plot.png", bbox_inches='tight')
    plt.show()


def z_score_analysis(X, num_std):
    mean_x = np.mean(X)
    std_x = np.std(X)
    X_z_score = np.zeros(len(X))
    high_zscore_index = []
    high_zscore = []
    acceptable_zscore_index = []
    for i in range(len(X)):
        X_z_score[i] = (np.abs(X[i] - mean_x))/std_x
        if X_z_score[i] >= num_std:
            high_zscore_index.append(i)
            high_zscore.append(X_z_score[i])
        else:
            acceptable_zscore_index.append(i)        
    num_high_zscore = len(high_zscore)
    return high_zscore_index, high_zscore, num_high_zscore, acceptable_zscore_index


def power_law(x, a1, b1, c1):
    return a1 * np.power(x, b1) + c1

def linear_func(x, a1, b1):
    return a1 * x + b1


def detect_outliers(code, fit_type, num_std, make_plot=True, save_plot=True):
    dbPath=""
    db=pd.read_csv(os.path.join(dbPath,code+'_prediction_data.csv'))
    db=db.dropna()
    col_names = db.columns.tolist()[2:]
    X=db.iloc[:,2:-1].copy().to_numpy('float')
    Y_exp=db.iloc[:,-1].copy().to_numpy('float')
    Y_gc = X[:,-1].reshape(-1,1)
    Y_disc = X[:,-1]-Y_exp
    mol_wt = X[:,0].reshape(-1,1)
    X = Y_gc.flatten()
    Y = Y_exp.flatten()
    X_fit = np.linspace((np.min(X)-np.min(X)*0.1), (np.max(X)+np.max(X)*0.2), 1000)  
    if fit_type == "linear":
        params, covariance = curve_fit(linear_func, X, Y, maxfev=10000)
        a1, b1 = params
        Y_fit = linear_func((X_fit), a1, b1)
        Y_fit_all = linear_func((X), a1, b1)
    elif fit_type == "power_law":
        params, covariance = curve_fit(power_law, X, Y, maxfev=10000)
        a1, b1, c1 = params
        Y_fit = power_law((X_fit), a1, b1, c1)
        Y_fit_all = power_law((X), a1, b1, c1)
    mean_x = np.mean(X)
    std_x = np.std(X)
    error_bound =num_std*std_x
    outlier_score = np.zeros(len(X))
    outlier_index = []
    inlier_index = []
    for i in range(len(X)):
        y_fit = Y_fit_all[i]
        outlier_score[i] = ((Y[i] - y_fit)/error_bound)*num_std
        if np.abs(outlier_score[i]) > num_std:
            outlier_index.append(i)
        else:
            inlier_index.append(i)
    outlier_id = db.iloc[(outlier_index),0:]
    inlier_id = db.iloc[(inlier_index),0:]
    if make_plot:
        fig, ax = plt.subplots()
        ax.tick_params(axis='x', labelrotation=45)
        if code == 'Tm':
            ax_lab = 'T$_m$'
            units = '/K'
        elif code == 'Tb':
            ax_lab = 'T$_b$'
            units = '/K'
        elif code == 'Tc':
            ax_lab = 'T$_c$'
            units = '/K'
        elif code == 'Vc':
            ax_lab = 'V$_c$'
            units = '/cm$^3$mol$^{-1}$'
        elif code == 'Pc':
            ax_lab = 'P$_c$'
            units = '/bar'
        elif code == 'Hvap' or code == 'Hvap_orig':
            ax_lab = '$\Delta$H$_{vap}$'
            units = '/kJmol$^{-1}$'
        plt.rcParams['figure.dpi'] = 300
        plt.scatter(X[inlier_index], Y[inlier_index], marker='o', facecolors='none', edgecolors='blue')
        plt.scatter(X[outlier_index], Y[outlier_index], marker='o', facecolors='none', edgecolors='red')
        plt.plot(X_fit, Y_fit, "g-")
        plt.plot(X_fit, Y_fit+error_bound, 'r-')
        plt.plot(X_fit, Y_fit-error_bound, 'r-')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        ax.minorticks_on()
        plt.xlabel('GC '+ax_lab+' '+units, fontsize=20, weight='bold')
        plt.ylabel('Exp. '+ax_lab+' '+units, fontsize=20, weight='bold')
        if save_plot:
            dir_root = f"./Data_Viz_and_Outlier_Figs/{code}/"
            os.makedirs(dir_root, exist_ok=True)
            fig.savefig(dir_root+"outlier_viz_parity.png", bbox_inches='tight')
        plt.show()
        fig, ax = plt.subplots()
        ax.tick_params(axis='x', labelrotation=45)
        outlier_lo_limit = np.ones(len(X)) * num_std
        outlier_zero = np.zeros(len(X))
        outlier_hi_limit = (np.ones(len(X))) * -1 * num_std
        plt.plot(X, outlier_lo_limit, 'r-')
        plt.plot(X, outlier_zero, color='green')
        plt.plot(X, outlier_hi_limit, 'r-')
        plt.scatter(X[inlier_index], outlier_score[inlier_index], marker='x', color='black')
        plt.scatter(X[outlier_index], outlier_score[outlier_index], marker='x', color='red')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        ax.minorticks_on()
        plt.xlabel('GC '+ax_lab+' '+units, fontsize=20, weight='bold')
        plt.ylabel("Outlier score", fontsize=20, weight='bold')
        if save_plot:
            fig.savefig(dir_root+"outlier_score.png", bbox_inches='tight')
        plt.show()        
    return outlier_index, inlier_index, outlier_score, std_x, outlier_id, inlier_id


def show_outlier(code, num_std=2.0, make_plot=True, save_plot=True):
    dir_root = f"./Data_Viz_and_Outlier_Figs/{code}/"
    os.makedirs(dir_root, exist_ok=True)
    if code=="Hvap" or code=="Hvap_orig" or code=="Pc" or code=="Vc":
        fit_type_="linear"
    elif code=="Tb" or code=="Tc" or code=="Tm":
        fit_type_= "power_law"
    outlier_res = detect_outliers(code, fit_type_, num_std, make_plot, save_plot)
    outlier_id = outlier_res[4]
    inlier_id = outlier_res[5]
    outlier_id.to_csv(dir_root+f'{code}_prediction_data_outlier.csv', index=False)
    return inlier_id, outlier_id, outlier_res[2]


code_array = np.array(['Hvap', 'Hvap_orig', 'Pc', 'Vc', 'Tc', 'Tb', 'Tm'])
num_std = 2.0
outlier_res_all = []
for i in range(len(code_array)):
    show_outlier_ = show_outlier(code_array[i], num_std=2.0, make_plot=True, save_plot=True)

# end
   
    

