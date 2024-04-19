import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px


def plot_heatmaps(data1, data2, data3, title1="Heatmap 1", title2="Heatmap 2", title3="Heatmap 3", figsize=(14, 12), cmap='BrBG', heatmap3_figsize=(30,20)):
    """
    Args:
        data1: First dataset for heatmap.
        data2: Second dataset for heatmap.
        data3: Third dataset for heatmap.
        figsize: Figure size for the first two heatmaps.
        cmap: Colormap for the heatmaps.
        figsize3: Figure size for the third heatmap.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    sns.heatmap(data1.corr(), annot=True, vmin=-1.0, vmax=1.0, center=0,fmt="0.2f", cmap=cmap, ax=axes[0])
    axes[0].set_title(title1)
    
    sns.heatmap(data2.corr(), annot=True, vmin=-1.0, vmax=1.0, center=0,fmt="0.2f", cmap=cmap, ax=axes[1])
    axes[1].set_title(title2)
    
    plt.figure(figsize=heatmap3_figsize)
    sns.heatmap(data3.select_dtypes("number").corr(), vmin=-1, vmax=1, center=0,annot= True, fmt="0.2f", cmap=cmap)
    plt.title(title3)

    plt.tight_layout()
    plt.show()

def plot_histogram(data, variable1, variable2):
    """
    Args:
    data: The input dataset containing the variables.
    variable1: The name of the first variable to plot.
    variable2: The name of the second variable to plot.
    """
    # Create a figure with subplots
    fig = go.Figure()

    # Add subplot for variable1
    fig.add_trace(go.Histogram(x=data[variable1], marker=dict(color='orange'), name="General Use", yaxis="y", xaxis="x"))

    # Add subplot for variable2
    fig.add_trace(go.Histogram(x=data[variable2], marker=dict(color='green'), name="Generation", yaxis="y2", xaxis="x2"))

    # Update layout of the plot
    fig.update_layout(
        title="Energy Generation and Consumption Distribution",
        xaxis_title="Value",
        yaxis_title="Frequency",
        xaxis=dict(domain=[0, 0.45]),
        xaxis2=dict(domain=[0.55, 1]),
        yaxis=dict(title="Frequency"),
        yaxis2=dict(
            title="Frequency",
            overlaying="y",
            side="right"
        ),
        bargap=0.1,
        grid=dict(rows=1, columns=2),
        bargroupgap=0.2
    )

    # Print distribution metrics for variable1
    skewness1 = data[variable1].skew()
    kurtosis1 = data[variable1].kurt()
    print(f"{variable1} distribution metrics:")
    print(f"Skewness: {skewness1}")
    print(f"Kurtosis: {kurtosis1}")

    # Print distribution metrics for variable2
    skewness2 = data[variable2].skew()
    kurtosis2 = data[variable2].kurt()
    print(f"\n{variable2} distribution metrics:")
    print(f"Skewness: {skewness2}")
    print(f"Kurtosis: {kurtosis2}")

    fig.show()



