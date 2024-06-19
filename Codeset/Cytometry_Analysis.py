import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: The data loaded from the CSV file.
    """
    return pd.read_csv(file_path)

def calculate_summary_statistics(data):
    """
    Calculate summary statistics for the data.

    Args:
        data (DataFrame): The data for which to calculate summary statistics.

    Returns:
        DataFrame: A DataFrame containing summary statistics.
    """
    return data.describe()

def calculate_correlation_matrix(data):
    """
    Calculate the correlation matrix for the data.

    Args:
        data (DataFrame): The data for which to calculate the correlation matrix.

    Returns:
        DataFrame: A DataFrame containing the correlation matrix.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    return data[numeric_cols].corr()

def plot_heatmap(correlation_matrix, pdf):
    """
    Plot a heatmap of the correlation matrix and save it to a PDF.

    Args:
        correlation_matrix (DataFrame): The correlation matrix to plot.
        pdf (PdfPages): The PdfPages object to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    pdf.savefig()
    plt.close()

def apply_agglomerative_clustering(correlation_matrix, n_clusters=5):
    """
    Apply Agglomerative Clustering to the transposed correlation matrix.

    Args:
        correlation_matrix (DataFrame): The correlation matrix to cluster.
        n_clusters (int): The number of clusters to form.

    Returns:
        ndarray: Cluster labels for each column.
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clustering.fit(correlation_matrix.T)
    return clustering.labels_

def plot_cluster_distribution(cluster_labels, pdf):
    """
    Plot the distribution of columns across clusters and save it to a PDF.

    Args:
        cluster_labels (ndarray): Cluster labels for each column.
        pdf (PdfPages): The PdfPages object to save the plot.
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_counts.keys(), cluster_counts.values())
    plt.title('Distribution of Columns Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Columns')
    plt.xticks(list(cluster_counts.keys()))
    pdf.savefig()
    plt.close()

def plot_colored_scatter(data, labels, title, pdf):
    """
    Plot a scatter plot with points colored by cluster labels and save it to a PDF.

    Args:
        data (ndarray): The data to plot.
        labels (ndarray): Cluster labels for each data point.
        title (str): The title of the plot.
        pdf (PdfPages): The PdfPages object to save the plot.
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(labels))))
    label_colors = dict(zip(np.unique(labels), colors))

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        plt.scatter(data[label_indices, 0], data[label_indices, 1], c=label_colors[label], alpha=0.5, label=f'Cluster {label}')
    plt.title(title)
    plt.legend()
    pdf.savefig()
    plt.close()

def plot_pca(data, numeric_cols, labels, pdf):
    """
    Plot a PCA scatter plot with points colored by cluster labels and save it to a PDF.

    Args:
        data (DataFrame): The data to perform PCA on.
        numeric_cols (list): List of numeric columns to include in PCA.
        labels (ndarray): Cluster labels for each data point.
        pdf (PdfPages): The PdfPages object to save the plot.
    """
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data[numeric_cols])
    plot_colored_scatter(pca_results, labels, 'PCA Scatter Plot', pdf)

def generate_summary_statistics_table(data, pdf):
    """
    Generate a table of summary statistics and save it to a PDF.

    Args:
        data (DataFrame): The data for which to generate summary statistics.
        pdf (PdfPages): The PdfPages object to save the table.
    """
    summary_stats = calculate_summary_statistics(data)
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    plt.table(cellText=summary_stats.values, colLabels=summary_stats.columns, loc='center', cellLoc='center')
    pdf.savefig()
    plt.close()

def plot_peak_points(data, pdf):
    """
    Plot the six highest peak points for each '-H' column and save it to a PDF.

    Args:
        data (DataFrame): The data to find and plot peaks.
        pdf (PdfPages): The PdfPages object to save the plot.
    """
    min_distance = 1000  # Minimum distance between peaks
    plt.figure(figsize=(14, 7))

    h_columns = [col for col in data.columns if col.endswith('-H')]

    for col in h_columns:
        peaks, _ = find_peaks(data[col], distance=min_distance)
        peak_values = data[col].iloc[peaks].nlargest(6)
        peak_times = data['Time'].iloc[peak_values.index]
        plt.scatter(peak_times, peak_values, label=col)

    plt.xlabel('Time')
    plt.ylabel('Detected Concentration')
    plt.title('Six Highest Peak Points for Each Detection Type')
    plt.legend()
    pdf.savefig()
    plt.close()

def plot_cluster_columns(cluster_columns, pdf):
    """
    Plot a bar chart listing columns in each cluster and save it to a PDF.

    Args:
        cluster_columns (dict): Dictionary with cluster numbers as keys and lists of column names as values.
        pdf (PdfPages): The PdfPages object to save the plot.
    """
    plt.figure(figsize=(12, 8))
    for i in range(1, 6):
        plt.subplot(2, 3, i)
        plt.bar(range(len(cluster_columns[i])), cluster_columns[i])
        plt.title(f'Cluster {i}')
        plt.xlabel('Column Index')
        plt.ylabel('Column Name')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

def analyze_file(file_path, output_folder, cluster_number):
    """
    Analyze a CSV file and generate a PDF report.

    Args:
        file_path (str): The path to the CSV file.
        output_folder (str): The folder to save the PDF report.
        cluster_number (int): The cluster number for naming the PDF file.
    """
    data = load_data(file_path)
    correlation_matrix = calculate_correlation_matrix(data)
    
    cluster_labels = apply_agglomerative_clustering(correlation_matrix.values[:, 1:], n_clusters=5)
    
    cluster_columns = {i: [] for i in range(1, 6)}
    for col, label in zip(data.columns[1:], cluster_labels):
        cluster_columns[label + 1].append(col)

    pdf_path = os.path.join(output_folder, f'Cluster_{cluster_number}_Analysis.pdf')
    with PdfPages(pdf_path) as pdf:
        plot_heatmap(correlation_matrix, pdf)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()[1:]
        plot_pca(data, numeric_cols, cluster_labels, pdf)
        plot_peak_points(data, pdf)
        generate_summary_statistics_table(data, pdf)
        plot_cluster_distribution(cluster_labels, pdf)
        plot_cluster_columns(cluster_columns, pdf)

def main(input_folder_path, output_folder_path):
    """
    Main function to process and analyze CSV files from each cluster directory.

    Args:
        input_folder_path (str): The main directory containing cluster subdirectories.
        output_folder_path (str): The directory to save the analysis reports.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for cluster_dir in os.listdir(input_folder_path):
        cluster_path = os.path.join(input_folder_path, cluster_dir)
        if os.path.isdir(cluster_path) and cluster_dir.startswith('Cluster'):
            cluster_number = int(cluster_dir.split()[-1])
            file_path = os.path.join(cluster_path, 'Cluster_Combined_Data.csv')
            if os.path.exists(file_path):
                analyze_file(file_path, output_folder_path, cluster_number)

if __name__ == "__main__":
    input_folder_path = r'C:\Users\praty\Downloads\Research\ClusteredSamples'
    output_folder_path = r'C:\Users\praty\Downloads\Research\AnalysisOutput'
    main(input_folder_path, output_folder_path)
