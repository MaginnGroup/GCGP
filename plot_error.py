import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

xlabels = ['$H_{vap}$'+'\n [kJ '+r'$\cdot$'+' mol' + '$^{-1}$'+']',
           '$P_c$'+'\n [bar]',
           '$T_b$'+'\n [K]',
           '$T_c$'+'\n [K]',
           '$T_m$'+'\n [K]',
           '$V_c$'+'\n [cm'+'$^{3}$'+']']

# Set path to results
train = pd.read_csv(os.path.join('Final_Results', 'train_error'), index_col = 0)
test  = pd.read_csv(os.path.join('Final_Results', 'test_error'),  index_col = 0)
width = 0.35
no_cats = len(xlabels)
x = np.arange(no_cats)

plt.figure()
plt.bar(x-width/2, train.loc['MAE'], width, color = 'b', label = 'Train')
plt.bar(x+width/2, test.loc['MAE'],  width, color = 'r', label = 'Test')
plt.xticks(x, labels = xlabels)
plt.xlabel("Thermophysical Property")
plt.ylabel("Mean Absolute Error (MAE)")
plt.legend()
plt.savefig(os.path.join('Final_Results', 'mae_bar_chart'))

plt.figure()
plt.bar(x-width/2, train.loc['RMSE'], width, color = 'b', label = 'Train')
plt.bar(x+width/2, test.loc['RMSE'],  width, color = 'r', label = 'Test')
plt.xticks(x, labels = xlabels)
plt.xlabel("Thermophysical Property")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.legend()
plt.savefig(os.path.join('Final_Results', 'rmse_bar_chart'))
