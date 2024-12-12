import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import os
import csv
import logging


class AccuracyLogger:
    def __init__(self, file_path, config_class):
        self.file_path = file_path
        # Derive header dynamically from config class attributes
        self.header = list(vars(config_class()).keys()) + ["accuracy", "time"]
        # Ensure file exists and header is in place
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.header)

    def log(self, config, accuracy, time):
        # Prepare the row data from config attributes and accuracy
        row = [getattr(config, attr)
               for attr in vars(config)] + [round(accuracy, 4)] + [round(time, 0)]
        # Append to the file
        with open(self.file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
        logging.info(f"Logged: {row}")


class HyperparameterResultsPlotter:
    def __init__(self, csv_path):
        """
        Initialize the plotter by loading the data.
        :param csv_path: Path to the CSV file containing the results.
        """
        self.df = pd.read_csv(csv_path)

    def scatter_plot(self, x_param, y_param, hue_param=None, size_param=None, title=None):
        """
        Create a scatter plot of the results.
        :param x_param: Column name for x-axis.
        :param y_param: Column name for y-axis.
        :param hue_param: Column name for color grouping.
        :param size_param: Column name for size scaling.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df, x=x_param, y=y_param, hue=hue_param,
                        size=size_param, palette='viridis', sizes=(20, 200))
        plt.title(title or f'{x_param} vs {y_param}')
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.legend(title=hue_param)
        plt.show()

    def heatmap(self, index_param, column_param, value_param, title=None):
        """
        Create a heatmap of the results.
        :param index_param: Column name for heatmap rows.
        :param column_param: Column name for heatmap columns.
        :param value_param: Column name for heatmap values.
        :param title: Title of the plot.
        """
        heatmap_data = self.df.pivot_table(
            values=value_param, index=index_param, columns=column_param)
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(title or f'{value_param} Heatmap')
        plt.xlabel(column_param)
        plt.ylabel(index_param)
        plt.show()

    def line_plot(self, x_param, y_param, hue_param=None, style_param=None, title=None):
        """
        Create a line plot of the results.
        :param x_param: Column name for x-axis.
        :param y_param: Column name for y-axis.
        :param hue_param: Column name for color grouping.
        :param style_param: Column name for line style differentiation.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=self.df, x=x_param, y=y_param,
                     hue=hue_param, style=style_param, markers=True)
        plt.title(title or f'{x_param} vs {y_param}')
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.show()

    def three_param_plot(self, x_param, y_param, z_param, color_param=None, title=None):
        """
        Create a 3D scatter plot for visualizing three parameters.
        :param x_param: Column name for x-axis.
        :param y_param: Column name for y-axis.
        :param z_param: Column name for z-axis.
        :param color_param: Column name for color grouping.
        :param title: Title of the plot.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Map color
        colors = None
        if color_param:
            cmap = plt.get_cmap('plasma')
            norm = LogNorm(vmin=0.7, vmax=self.df[color_param].max())
            colors = cmap(norm(self.df[color_param]))

        # Scatter plot
        scatter = ax.scatter(
            self.df[x_param], self.df[y_param], self.df[z_param], c=colors, marker='o')

        # Color bar if applicable
        if color_param:
            mappable = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
            mappable.set_array([])
            fig.colorbar(mappable, ax=ax, label=color_param)

        # Set labels
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_zlabel(z_param)
        ax.set_title(title or f'{x_param}, {y_param}, and {z_param}')
        plt.show()

    def plot_all_features_vs_accuracy(self, features, accuracy_column='accuracy', hue_param='accuracy'):
        """
        Plot all features against accuracy in a grid of subplots with two columns.
        :param features: List of feature names to plot against accuracy.
        :param accuracy_column: Name of the accuracy column.
        :param hue_param: Optional column name for color grouping.
        """
        num_features = len(features)
        num_columns = 2  # Set number of columns for subplots

        # Determine number of rows needed
        num_rows = (num_features + num_columns -
                    1) // num_columns  # Ceil division

        fig, axes = plt.subplots(
            num_rows, num_columns, figsize=(20, 6 * num_rows))

        # Flatten axes array in case of multi-dimensional subplot grid
        axes = axes.flatten()

        for i, feature in enumerate(features):
            sns.scatterplot(data=self.df, x=feature, y=accuracy_column,
                            hue=hue_param, ax=axes[i], palette='plasma')
            axes[i].set_title(f'{feature} vs {accuracy_column}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(accuracy_column)

        # Remove any unused axes (in case there are fewer features than subplot spaces)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
