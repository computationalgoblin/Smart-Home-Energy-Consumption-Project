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
