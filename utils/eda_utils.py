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
    # Resample the data at a daily frequency and calculate the mean for each day
    resampled_data = data[cols].resample(freq).mean()
    
    # Plot the resampled data in subplots
    resampled_data.plot(subplots=True, layout=(-1, 2), figsize=(20, 15), grid=True)

def scatterplot_subplots(x, y_cols, data):
    num_plots = len(y_cols)
    num_cols = 3  # Number of columns in the subplots layout
    num_rows = (num_plots - 1) // num_cols + 1  # Number of rows in the subplots layout
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*5))
    axes = axes.flatten()  # Convert the axes array into a 1D array
    
    for i, col in enumerate(y_cols):
        ax = axes[i]
        ax.scatter(data[x], data[col])
        ax.set_title(f'{col} vs {x}')
        ax.set_xlabel(x)
        ax.set_ylabel(col)
    
    # Remove unused subplots
    for i in range(num_plots, num_cols * num_rows):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

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

