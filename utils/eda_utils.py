import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(data, title="Heatmap 1", figsize=(20,20), cmap='BrBG'):
    """
    Args:
        data: dataset for heatmap.
        figsize: figure size for the heatmap.
        cmap: colormap for the heatmap.
    """
    
    plt.figure(figsize=figsize)
    sns.heatmap(data.select_dtypes("number").corr(), vmin=-1, vmax=1, center=0,annot= True, fmt="0.2f", cmap=cmap)
    plt.title(title)

    plt.show()

