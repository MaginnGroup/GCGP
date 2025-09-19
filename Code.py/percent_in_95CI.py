# -*- coding: utf-8 -*-
"""
Script to plot % inside CI
Contributors: Barnabas Agbodekhe
"""



import matplotlib.pyplot as plt
import numpy as np
import os
import time

                                                                                       
prop_id = ['Hvap', 'Pc', 'Tb', 'Tc', 'Tm', 'Vc']
Train_CI_out = []
Test_CI_out = []
for i, prop_ in enumerate(prop_id):
    count_ci_data = np.loadtxt(f'./Final_Results/{prop_}/{prop_}_count_CI.txt')
    Train_CI_out.append(count_ci_data[1])
    Test_CI_out.append(count_ci_data[3])
    
Train_CI = 100 - np.array([Train_CI_out]).flatten() * 100
Test_CI = 100 - np.array([Test_CI_out]).flatten() * 100


Props = [
    r'$\mathbf{\Delta H_{vap}}$',
    r'$\mathbf{P_c}$',
    r'$\mathbf{T_b}$',
    r'$\mathbf{T_c}$',
    r'$\mathbf{T_m}$',
    r'$\mathbf{V_c}$'
]

#Props = ['$\Delta$$H_{vap}$', '$P_c$', '$T_b$', '$T_c$', '$T_m$', '$V_c$'] # In the order Hvap, Pc, Tb, Tc, Tm, Vc
                                                                    # Same order applies for collated Train_CI and Test_CI above

x = np.arange(len(Props))  
width = 0.28 

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5), 
                               gridspec_kw={'height_ratios': [2.4, 1]})

hatches = [None, '//']
bars1_ax1 = ax1.bar(x - width/2, Train_CI, width, label='Train', color='#6699ff', edgecolor='k', hatch=hatches[0])
bars2_ax1 = ax1.bar(x + width/2, Test_CI, width, label='Test', color='red', edgecolor='k', hatch=hatches[1])

bars1_ax2 = ax2.bar(x - width/2, Train_CI, width, label='Train', color='#6699ff', edgecolor='k', hatch=hatches[0])
bars2_ax2 = ax2.bar(x + width/2, Test_CI, width, label='Test', color='red', edgecolor='k', hatch=hatches[1])

# Set y-axis limits for break effect
ax1.set_ylim(80, 107)  # Upper part
ax2.set_ylim(0, 10)    # Lower part
ax2.set_yticks(np.arange(0, 11, 5))

# Hide spines between broken axis
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # No labels on top subplot
ax2.xaxis.tick_bottom()

# Set x-ticks and make them bold
ax2.set_xticks(x)
ax2.set_xticklabels(Props)
for label in ax2.get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(14)

# Grid lines
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Add diagonal break markers
d = 0.010  # Break size
kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)  # Top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

kwargs.update(transform=ax2.transAxes)  # Update transform for bottom plot
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

# Annotate bar values
for bar in bars1_ax1:
    ax1.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 1, 
             f"{bar.get_height():.2f}",  
             ha='center', va='bottom', rotation=90, 
             fontsize=14, fontweight='normal')

for bar in bars2_ax1:
    ax1.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 1, 
             f"{bar.get_height():.2f}", 
             ha='center', va='bottom', rotation=90, 
             fontsize=14, fontweight='normal')

# Y-axis label
fig.text(-0.032, 0.5, '% predicted within 95% \n confidence intervals', 
         va='center', rotation='vertical', fontsize=14, fontweight='bold')

# Tick sizes
ax1.tick_params(axis='x', labelsize=14)  
ax1.tick_params(axis='y', labelsize=14)  
ax2.tick_params(axis='x', labelsize=14)  
ax2.tick_params(axis='y', labelsize=14)

# Legend
ax1.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.15))



# Save plot
dir_root = "Final_Results/"
os.makedirs(dir_root, exist_ok=True)
plt.savefig(dir_root+'percent_95percent_CI.png', dpi=500, bbox_inches='tight')

plt.tight_layout()
# plt.show()







