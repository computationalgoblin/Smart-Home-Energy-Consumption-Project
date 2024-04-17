import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmaps(data1, data2, title1="Heatmap 1", title2="Heatmap 2", figsize=(12, 10), cmap='viridis'):
    """Plot dual heatmaps with customized titles, color maps, and shared colorbar.

    Args:
        data1: First dataset for heatmap.
        data2: Second dataset for heatmap.
        figsize (tuple, optional): Figure size. Defaults to (12, 10).
        cmap: Colormap for the heatmap
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    sns.heatmap(data1.corr(), annot=True, vmin=-1.0, vmax=1.0, center=0, cmap=cmap, ax=axes[0])
    axes[0].set_title(title1)
    
    sns.heatmap(data2.corr(), annot=True, vmin=-1.0, vmax=1.0, center=0, cmap=cmap, ax=axes[1])
    axes[1].set_title(title2)

    plt.tight_layout()
    plt.show()

