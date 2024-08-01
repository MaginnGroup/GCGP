import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

mysalmon = (235/255, 134/255, 100/255)
myteal = (103/225, 185/255, 155/255)

xlabels = ['$H_{vap}$'+'\n [kJ '+r'$\cdot$'+' mol' + '$^{-1}$'+']',
           '$P_c$'+'\n [bar]',
           '$T_b$'+'\n [K]',
           '$T_c$'+'\n [K]',
           '$T_m$'+'\n [K]',
           '$V_c$'+'\n [cm'+'$^{3}$'+']']

# Set path to results
train = pd.read_csv(os.path.join('Final_Results', 'train_error'), index_col = 0)
test  = pd.read_csv(os.path.join('Final_Results', 'test_error'),  index_col = 0)
width = 0.2
no_cats = len(xlabels)
x = np.arange(no_cats)

plt.figure()
plt.bar(x - 1.5*width, train.loc['MAE'],  width,  color = mysalmon, edgecolor = 'k')
plt.bar(x - width/2,   test.loc['MAE'],   width,  color = mysalmon, edgecolor = 'k',  hatch = '/')

plt.bar(x + width/2,   train.loc['RMSE'],  width,  color = myteal, edgecolor = 'k')
plt.bar(x + 1.5*width, test.loc['RMSE'],  width, color = myteal, edgecolor = 'k', hatch = '/')

# For the legends
plt.bar(x - 1.5*width, 0*train.loc['MAE'],  width,  color = mysalmon, edgecolor = 'k', label = 'MAE')
plt.bar(x - 1.5*width, 0*train.loc['MAE'],  width,  color = myteal, edgecolor = 'k', label = 'RMSE')
plt.bar(x + width/2,   0*test.loc['MAE'],   width,  color = 'w', edgecolor = 'k',  label = 'Train')
plt.bar(x + width/2,   0*test.loc['MAE'],   width,  color = 'w', edgecolor = 'k',  hatch = '/', label = 'Test')

plt.xticks(x, labels = xlabels)
plt.xlabel("Thermophysical Property")
plt.ylabel("Error")
plt.legend()
plt.savefig(os.path.join('Final_Results', 'error_bar_chart'))

# plt.figure()

# plt.xticks(x, labels = xlabels)
# plt.xlabel("Thermophysical Property")
# plt.ylabel("RMSE")
# plt.legend()
# plt.savefig(os.path.join('Final_Results', 'rmse_bar_chart'))
