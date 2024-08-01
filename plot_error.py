import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import global_plot_settings as rcParams

rcParams.set_plot_settings()

mysalmon = (235/255, 134/255, 100/255)
myteal = (103/225, 185/255, 155/255)

xlabels = ['$H_{vap}$'+'\n [kJ '+'$\cdot$'+' mol' + '$^{-1}$'+']',
           '$P_c$'+' [bar]',
           '$T_b$'+' [K]',
           '$T_c$'+' [K]',
           '$T_m$'+' [K]',
           '$V_c$'+'\n [cm'+'$^{3}$'+'$\cdot$'+' mol' + '$^{-1}$'+']']

# Set path to results
train = pd.read_csv(os.path.join('Final_Results', 'train_error'), index_col = 0)
test  = pd.read_csv(os.path.join('Final_Results', 'test_error'),  index_col = 0)
width = 0.2
no_cats = len(xlabels)
x = np.arange(no_cats)

plt.figure(figsize=(10,6))
plt.bar(x - 1.5*width, train.loc['MAE'],  width,  color = mysalmon, edgecolor = 'k')
plt.bar(x - width/2,   test.loc['MAE'],   width,  color = mysalmon, edgecolor = 'k',  hatch = '//')

plt.bar(x + width/2,   train.loc['RMSE'],  width,  color = myteal, edgecolor = 'k')
plt.bar(x + 1.5*width, test.loc['RMSE'],  width, color = myteal, edgecolor = 'k', hatch = '//')

# For the legends
plt.bar(x - 1.5*width, 0*train.loc['MAE'],  width,  color = mysalmon, edgecolor = 'k', label = 'MAE')
plt.bar(x - 1.5*width, 0*train.loc['MAE'],  width,  color = myteal, edgecolor = 'k', label = 'RMSE')
plt.bar(x + width/2,   0*test.loc['MAE'],   width,  color = 'w', edgecolor = 'k',  label = 'Train')
plt.bar(x + width/2,   0*test.loc['MAE'],   width,  color = 'w', edgecolor = 'k',  hatch = '//', label = 'Test')

plt.xticks(x, labels = xlabels)
plt.xlabel("Thermophysical Property")
plt.ylabel("Error")
plt.legend(loc = 'upper left')
plt.savefig(os.path.join('Final_Results', 'error_bar_chart'))
