import matplotlib.pyplot as plt

def set_plot_settings():
    """
    Set global plot settings for Matplotlib.
    """
    plt.rcParams['figure.figsize']      = (6, 6)
    plt.rcParams['figure.dpi']          = 300
    plt.rcParams['axes.labelsize']      = 16
    plt.rcParams['xtick.labelsize']     = 15
    plt.rcParams['ytick.labelsize']     = 15
    plt.rcParams['legend.fontsize']     = 12
    plt.rcParams['lines.linewidth']     = 3
    plt.rcParams['lines.markersize']    = 8
    plt.rcParams['axes.labelweight']    = 'bold'
    plt.rcParams['xtick.direction']     = 'in'
    plt.rcParams['ytick.direction']     = 'in'
    plt.rcParams['xtick.top']           = True
    plt.rcParams['ytick.right']         = True
    plt.rcParams['axes.labelpad']       = 10
    plt.rcParams['savefig.bbox']        = 'tight'

if __name__ == "__main__":
    set_plot_settings()
    print("Global plot settings have been applied.")
