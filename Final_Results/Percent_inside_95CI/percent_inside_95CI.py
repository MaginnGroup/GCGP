# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

Train_CI = 100 - np.array([0.04604, 0.06193, 0.04530, 0.04371, 0.05314, 0.02496])* 100 # values from file *_count_CI.txt 2nd row 
Test_CI = 100 - np.array([0.07143, 0.05839, 0.04730, 0.07692, 0.06039, 0.11429]) * 100 # values from file *_count_CI.txt 4th row
                                                                                        # where * is the property code

Props = ['$\Delta$$H_{vap}$', '$P_c$', '$T_b$', '$T_c$', '$T_m$', '$V_c$'] # In the order Hvap, Pc, Tb, Tc, Tm, Vc
                                                                    # Same order applies for collated Train_CI and Test_CI above

x = np.arange(len(Props))  
width = 0.28 

z = np.arange(-0.5, len(Props)+0.55, 1)
line_95 = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]) * 100
line_90 = np.array([0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]) * 100


fig, ax = plt.subplots()

bars1 = ax.bar(x - width/2, Train_CI, width, label='Train', color='blue')
bars2 = ax.bar(x + width/2, Test_CI, width, label='Test', color='red')

plt.plot(z, line_95, ls='--', color='green')
plt.plot(z, line_90, ls='--', color='gray')


for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width()/2, 
        bar.get_height() + 1, 
        f"{bar.get_height():.2f}",  
        ha='center', 
        va='bottom', 
        rotation=90, 
        fontsize=13.5, 
        fontweight='normal' 
    )

for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width()/2, 
        bar.get_height() + 1, 
        f"{bar.get_height():.2f}", 
        ha='center', 
        va='bottom', 
        rotation=90, 
        fontsize=13.5,  
        fontweight='normal'
    )


ax.set_ylabel('% predicted within 95% \n confidence intervals', size=15)
ax.set_xticks(x)
ax.set_xticklabels(Props)


ax.tick_params(axis='x', labelsize=14)  
ax.tick_params(axis='y', labelsize=14)  


for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('normal')  

plt.ylim(0, 117)
ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.1))

plt.savefig('percent_95%_CI.png', dpi=350, bbox_inches='tight')

plt.tight_layout()
plt.show()
