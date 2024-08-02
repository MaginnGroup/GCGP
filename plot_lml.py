import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import global_plot_settings as rcParams


rcParams.set_plot_settings()
phys_property = ['Hvap', 'Pc', 'Tb', 'Tc', 'Tm', 'Vc']
xlabels = ['$H_{vap}$'+'\n [kJ '+'$\cdot$'+' mol' + '$^{-1}$'+']',
           '$P_c$'+' [bar]',
           '$T_b$'+' [K]',
           '$T_c$'+' [K]',
           '$T_m$'+' [K]',
           '$V_c$'+'\n [cm'+'$^{3}$'+'$\cdot$'+' mol' + '$^{-1}$'+']']

# Collect log marginal likelihood for each thermophysical property
lml = []
for loopA, property in enumerate(phys_property):
    lml.append(np.loadtxt(os.path.join(os.getcwd(), 'Final_Results', property, 'model_4', 'lml')).tolist())
width = 0.35
no_cats = len(xlabels)
x = np.arange(no_cats)

lml = np.array(lml)
max_lml = max(lml)
dist = max_lml - min(lml)
norm_lml = (max_lml - lml)/dist

plt.figure(figsize=(8,6))
plt.bar(x, norm_lml, width, edgecolor = 'k')
plt.xticks(x, labels = xlabels)
plt.xlabel("Thermophysical Property")
plt.ylabel("Normalized Log Evidence")
plt.savefig(os.path.join('Final_Results', 'lml_bar_chart'))
