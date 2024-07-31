import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import json
from matplotlib.cm import get_cmap
import numpy as np  # For linspace
import pandas as pd
import os

class DataVisualizer:
    def __init__(self, config_path):
        """
        Initialize the DataVisualizer with configuration settings.

        Parameters:
        config_path (str): Path to the JSON configuration file.
        """
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        visualization_config = config['visualization']
        self.grid_size = visualization_config['grid_size']
        self.chunk_size = visualization_config['chunk_size']
        self.plot_settings = visualization_config['plot_settings']

    def plot_kde(self, df, x_col, y_col, output_file):
        """
        Plot a kernel density estimate (KDE) plot and save it to a file.

        Parameters:
        df (DataFrame): DataFrame containing the data to plot.
        x_col (str): Column name for the x-axis data.
        y_col (str): Column name for the y-axis data.
        output_file (str): Path to save the plot.
        """
        fig, ax = plt.subplots()
        ax.set_title(os.path.basename(output_file), fontsize=self.plot_settings['title_fontsize'], fontweight=self.plot_settings['title_fontweight'])

        sns.kdeplot(data=df, x=x_col, y=y_col, cmap=self.plot_settings['cmap'], fill=True, ax=ax)

        norm = mpl.colors.Normalize(vmin=df[y_col].min(), vmax=df[y_col].max())
        sm = plt.cm.ScalarMappable(cmap=self.plot_settings['cmap'], norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(self.plot_settings['kde']['colorbar_label'], fontsize=self.plot_settings['kde']['fontsize'], fontweight=self.plot_settings['kde']['fontweight'])

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

def visualize_csv_files(input_dir, output_dir, config_path):
    """
    Visualize data from CSV files in the input directory and save the plots to the output directory.

    Parameters:
    input_dir (str): Directory containing the input CSV files.
    output_dir (str): Directory to save the output plots.
    config_path (str): Path to the JSON configuration file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualizer = DataVisualizer(config_path)

    for csv_file in os.listdir(input_dir):
        if csv_file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, csv_file))
            event_name = os.path.splitext(csv_file)[0].split('_')[0]
            x_col = f"{event_name}_dnn"
            y_col = f"{event_name}_xgb"

            if x_col in df.columns and y_col in df.columns:
                output_file = os.path.join(output_dir, f"{event_name}_kde.pdf")
                visualizer.plot_kde(df, x_col, y_col, output_file)
            else:
                print(f"Columns {x_col} or {y_col} not found in {csv_file}")

if __name__ == "__main__":
    input_directory = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Filtered_Set\Event_Data'
    output_directory = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Filtered_Set\Event_Visualization'
    config_path = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\config.json'
    
    visualize_csv_files(input_directory, output_directory, config_path)
