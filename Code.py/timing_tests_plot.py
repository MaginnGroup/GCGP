import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Base directories
base_dir = "/scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_v01/Final_Results"
out_dir = "/scratch365/bagbodek/GCGP_clone_05_02_2024/GCGP_v01/Timing_Tests"
os.makedirs(out_dir, exist_ok=True)

# Fixed order of properties
properties = ["Hvap", "Pc", "Vc", "Tc", "Tb", "Tm", "Tm_Hfus"]

# -------------------------
# Training Data Processing
# -------------------------
train_means = []
train_stds = []
train_labels = []



# Pyplot Configuration
plt.rcParams['figure.dpi']=300
plt.rcParams['savefig.dpi']=300
#plt.rcParams['font.weight']='bold'
plt.rcParams['axes.titlesize']=21
plt.rcParams['axes.labelsize']=21
plt.rcParams['xtick.labelsize']=21
plt.rcParams['ytick.labelsize']=21
plt.rcParams['font.size']=12
plt.rcParams["savefig.pad_inches"]=0.02




for prop in properties:
    if prop=='Tb':
        varName='T$_{b}$'
    elif prop=='Tm':
        varName=r'T$_{m, 2}$'
    elif prop=='Tm_Hfus':
        varName=r'T$_{m, 3}$'
    elif prop=='Hvap':
        varName = "$\Delta$H$_{vap}$"
    elif prop == "Vc":
        varName = 'V$_{c}$'
    elif prop == "Tc":
        varName = 'T$_{c}$'
    elif prop == "Pc":
        varName = 'P$_{c}$'
    train_file = os.path.join(base_dir, prop, f"{prop}_GP_training_time.csv")
    if os.path.exists(train_file):
        try:
            df = pd.read_csv(train_file)
            times = df.iloc[0, 1:].astype(float)
            mean_val = times.mean()
            std_val = times.std()
            size = int(df.iloc[0, 0])
            train_means.append(mean_val)
            train_stds.append(std_val)
            train_labels.append(f"{varName}\n({size})")
        except (pd.errors.EmptyDataError, IndexError, ValueError) as e:
            print(f"Error processing training file {train_file}: {e}")
            continue
    else:
        print(f"Training file not found: {train_file}")

# Plot Training Times with broken y-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

x = np.arange(len(train_means))

# Find the maximum value to set y-limit for the top plot
max_val = np.ceil(max(train_means) / 100) * 100
top_ylim = max_val + 50 # Add a buffer for the text label

# Top (large values)
ax1.bar(x, train_means, yerr=train_stds, capsize=5, color="blue",  edgecolor='k')
# Bottom (small values)
ax2.bar(x, train_means, yerr=train_stds, capsize=5, color="blue",  edgecolor='k')

# Add numbers above bars
for i, v in enumerate(train_means):
    if v > 100:
        ax1.text(i, v + 20, f"{v:.1f}", ha="center", va="bottom", fontsize=18)
    else:
        ax2.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=18)


# Broken axis limits
ax1.set_ylim(250, top_ylim)  # Adjusted for better visibility
ax2.set_ylim(0, 20)  # Adjusted for better visibility

# Hide the spines and tick labels between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(bottom=False, labelbottom=False)  # Hide x-ticks and labels on top plot
ax2.xaxis.tick_bottom()

# Diagonal slashes
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax2.set_xticks(x)
#ax2.set_xticklabels(train_labels, rotation=30, ha="right")
ax2.set_xticklabels(train_labels)

fig.supxlabel("Property", fontsize=24, fontweight="bold") # Added x-axis label for bar plot
fig.supylabel("Training Time (s)", fontsize=24, fontweight="bold")
fig.suptitle("") # Removed plot title
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "training_times.png"), dpi=300)
plt.close()

# -------------------------
# Testing Data Processing
# -------------------------
color_cycle = ["#000080", "#FF0000", "#0000FF", "#404040", "#FF8C00", "#008000", "#800080"]
markers = ["o", "s", "D", "^", "v", ">", "<"]  # different open markers

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

# Ensure y-axis ticks have no more than 2 decimal places
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

for i, prop in enumerate(properties):
    if prop=='Tb':
        varName='T$_{b}$'
    elif prop=='Tm':
        varName=r'T$_{m, 2}$'
    elif prop=='Tm_Hfus':
        varName=r'T$_{m, 3}$'
    elif prop=='Hvap':
        varName = "$\Delta$H$_{vap}$"
    elif prop == "Vc":
        varName = 'V$_{c}$'
    elif prop == "Tc":
        varName = 'T$_{c}$'
    elif prop == "Pc":
        varName = 'P$_{c}$'
    test_file = os.path.join(base_dir, prop, f"{prop}_prediction_time_tests.csv")
    if os.path.exists(test_file):
        try:
            df = pd.read_csv(test_file)
            batch_sizes = df.iloc[:, 0].astype(int)
            times = df.iloc[:, 1:].astype(float)
            means = times.mean(axis=1)
            stds = times.std(axis=1)

            for ax in (ax1, ax2):
                ax.errorbar(batch_sizes, means, yerr=stds,
                            label=varName if ax is ax1 else "",
                            color=color_cycle[i % len(color_cycle)],
                            marker=markers[i % len(markers)], markersize=6,
                            mfc="white", mec=color_cycle[i % len(color_cycle)],
                            linestyle="-", capsize=3)
        except (pd.errors.EmptyDataError, IndexError, ValueError) as e:
            print(f"Error processing testing file {test_file}: {e}")
            continue
    else:
        print(f"Testing file not found: {test_file}")

# Broken y-axis limits
ax1.set_ylim(1, 4.5)   # zoomed in on larger values
ax2.set_ylim(0, 0.1)   # zoomed in on very small values

# Hide the spines and tick labels between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(bottom=False, labelbottom=False) # Hide x-ticks and labels on top plot
ax2.xaxis.tick_bottom()

# Diagonal slashes
d = .010
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)





fig.supxlabel("Batch Size", fontsize=24, fontweight="bold")
fig.supylabel("Prediction Time (s)", fontsize=24, fontweight="bold")
fig.suptitle("") # Removed plot title

# Adjusting legend placement and size
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.50), ncol=4, fontsize=20, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "prediction_times.png"), dpi=300)
plt.close()

print(f"Plots saved in {out_dir}")


