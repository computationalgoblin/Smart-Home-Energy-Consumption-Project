import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_dist_subplots(data, columns):
    num_columns = len(columns)
    num_rows = math.ceil(num_columns / 2)  # calculate the number of rows needed for subplots
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4*num_rows))
    axes = axes.flatten()  # flatten the axes array in case num_columns = 1
    
    for i, column in enumerate(columns):
        sns.distplot(data[column], ax=axes[i])
        axes[i].set_title(f'{column.capitalize()} Distribution')
    
    plt.tight_layout()
    plt.show()


def plot_subplots(data, cols: list, freq: str = "D"):
    """
    Args:
        data: the df containing the data.
        cols: a list of column names to plot.
        freq: the resampling frequence
    """
    # Resample the data at a daily frequency and calculate the mean for each day
    resampled_data = data[cols].resample(freq).mean()
    
    # Plot the resampled data in subplots
    resampled_data.plot(subplots=True, layout=(-1, 2), figsize=(20, 15), grid=True)


def time_series_subplots(data, cols):
    for freq in ["D", "W", "M"]:
        if freq == "D":
            title = "DAILY"
        elif freq == "W":
            title = "WEEKLY"
        elif freq == "M":
            title = "MONTHLY"
        
        print(f"---------------------- {title} ----------------------------")
        plot_subplots(data, cols, freq=freq)
        plt.show()
