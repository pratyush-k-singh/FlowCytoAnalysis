import os
import pandas as pd
import numpy as np
from scipy.stats import skew, moment
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def calculate_statistics(data, columns):
    """
    Calculate mean, median, skew, and moment for each column.
    
    Args:
        data (DataFrame): The data from which to calculate statistics.
        columns (list): The list of columns to calculate statistics for.
    
    Returns:
        dict: A dictionary of statistics.
    """
    stats = {
        'mean': [],
        'median': [],
        'skew': [],
        'moment': []
    }
    
    for column in columns:
        stats['mean'].append(data[column].mean())
        stats['median'].append(data[column].median())
        stats['skew'].append(skew(data[column]))
        stats['moment'].append(moment(data[column]))
    
    return stats

def normalize_statistics(statistics):
    """
    Normalize statistics to a scale of 0 to 100.
    
    Args:
        statistics (list of dict): A list of dictionaries of statistics.
    
    Returns:
        dict: A dictionary of normalized statistics.
    """
    normalized_stats = {}
    
    for key in statistics[0].keys():
        values = np.array([stat[key] for stat in statistics]).flatten()
        max_val = np.nanmax(values)  # Use np.nanmax to handle NaN values
        normalized_values = 100 * values / max_val
        
        # Ensure that NaN values in normalized_values are set to 0
        normalized_values[np.isnan(normalized_values)] = 0
        
        normalized_stats[key] = normalized_values.reshape(len(statistics), -1)
    
    return normalized_stats

def plot_heatmap(data, metric, columns, file_names, pdf, output_folder_path):
    """
    Plot heatmap for a given metric and save it to a PDF and PNG.
    
    Args:
        data (np.array): The data to plot.
        metric (str): The metric to plot.
        columns (list): The list of columns.
        file_names (list): The list of file names.
        pdf (PdfPages): The PDF file to save the heatmap.
        output_folder_path (str): The path to the folder to save the PNG files.
    """
    num_columns = len(file_names)
    num_rows = len(columns)
    fig_height = max(4, num_rows / 4.5)
    
    plt.figure(figsize=(20, fig_height))
    sns.heatmap(data.T, yticklabels=columns, xticklabels=file_names, cmap="Reds", annot=False, vmin=0, vmax=100)
    plt.title(f'{metric.capitalize()} Heatmap', fontsize=20)
    plt.ylabel('Channels', fontsize=15)
    plt.xlabel('Files', fontsize=15)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=8)  # Adjust the fontsize for y-axis labels
    pdf.savefig()
    plt.savefig(os.path.join(output_folder_path, f'{metric}_heatmap.png'), bbox_inches='tight')
    plt.close()

def main(input_folder_path, output_folder_path):
    """
    Main function to process the directory and create heatmaps.
    
    Args:
        input_folder_path (str): The path to the folder containing the CSV files.
        output_folder_path (str): The path to the folder to save the output PDF and PNGs.
    """
    statistics = []
    file_names = []
    columns = None
    
    # Calculate statistics for each file
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file_name)
            data = pd.read_csv(file_path)
            if columns is None:
                columns = data.columns[1:]  # Assuming the first column is 'Time'
            file_statistics = calculate_statistics(data, columns)
            statistics.append(file_statistics)
            file_names.append(file_name)
    
    # Normalize statistics
    normalized_stats = normalize_statistics(statistics)
    
    # Create the PDF file
    pdf_path = os.path.join(output_folder_path, 'Heatmaps.pdf')
    with PdfPages(pdf_path) as pdf:
        # Plot heatmaps for each metric and save them to the PDF and PNG
        for metric in normalized_stats.keys():
            heatmap_data = np.array(normalized_stats[metric])
            plot_heatmap(heatmap_data, metric, columns, file_names, pdf, output_folder_path)
    
    print(f'Heatmaps saved to {pdf_path}')

if __name__ == "__main__":
    input_folder_path = r'C:\Users\praty\Downloads\Research\NormalizedCytometryFiles'
    output_folder_path = r'C:\Users\praty\Downloads\Research\HeatmapOutput'
    main(input_folder_path, output_folder_path)
