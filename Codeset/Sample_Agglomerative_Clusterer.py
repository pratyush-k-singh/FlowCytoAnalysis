import os
import shutil
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: The data loaded from the CSV file.
    """
    return pd.read_csv(file_path)

def extract_features(data):
    """
    Extract features from the data.

    Args:
        data (DataFrame): The data from which to extract features.

    Returns:
        np.array: An array of extracted features.
    """
    features = []
    for column in data.columns[1:]:  # Assuming the first column is 'Time'
        column_data = data[column].dropna()
        mean = column_data.mean()
        std = column_data.std()
        peaks = np.sum((column_data.shift(1) < column_data) & (column_data.shift(-1) < column_data))
        features.extend([mean, std, peaks])
    return np.array(features)

def create_distance_matrix(feature_vectors):
    """
    Create a distance matrix from feature vectors.

    Args:
        feature_vectors (np.array): An array of feature vectors.

    Returns:
        np.array: A distance matrix.
    """
    pairwise_dist = pdist(feature_vectors, metric='euclidean')
    distance_matrix = squareform(pairwise_dist)
    return pairwise_dist, distance_matrix

def cluster_datasets(distance_matrix, n_clusters):
    """
    Cluster datasets based on a distance matrix.

    Args:
        distance_matrix (np.array): A condensed distance matrix.
        n_clusters (int): The number of clusters to form.

    Returns:
        np.array: An array of cluster labels.
    """
    Z = linkage(distance_matrix, method='ward')
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
    return cluster_labels

def merge_cluster_data(cluster_dir):
    """
    Merge data from all CSV files in a cluster into a single CSV file within the cluster directory.

    Args:
        cluster_dir (str): The directory containing the CSV files of the cluster.
    """
    # Initialize an empty list to store DataFrames
    data_frames = []
    
    # Iterate over all files in the cluster directory
    for file_name in os.listdir(cluster_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(cluster_dir, file_name)
            # Load data from CSV file
            data = pd.read_csv(file_path)
            
            # Append the DataFrame to the list, skipping the header for subsequent files
            if not data_frames:
                data_frames.append(data)
            else:
                data_frames.append(data.iloc[1:])

    # Concatenate all DataFrames in the list
    merged_data = pd.concat(data_frames, ignore_index=True)
    
    # Sort the merged data based on the 'Time' column
    merged_data.sort_values(by=merged_data.columns[0], inplace=True)
    
    # Save the merged and sorted data to a CSV file within the cluster directory
    merged_file_path = os.path.join(cluster_dir, 'Cluster_Combined_Data.csv')
    merged_data.to_csv(merged_file_path, index=False)

def main(input_folder_path, output_folder_path, n_clusters):
    """
    Main function to process and cluster datasets.

    Args:
        input_folder_path (str): The path to the folder containing the CSV files.
        output_folder_path (str): The path to the folder where clustered files will be saved.
        n_clusters (int): The number of clusters to form.
    """
    feature_vectors = []
    file_names = []

    # Load data and extract features for each file
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file_name)
            data = load_data(file_path)
            features = extract_features(data)
            feature_vectors.append(features)
            file_names.append(file_name)

    # Create distance matrix and cluster datasets
    condensed_dist_matrix, _ = create_distance_matrix(feature_vectors)
    cluster_labels = cluster_datasets(condensed_dist_matrix, n_clusters)

    # Create directories for each cluster and copy files
    cluster_directories = {}
    for cluster_label in np.unique(cluster_labels):
        cluster_dir = os.path.join(output_folder_path, f'Cluster {cluster_label}')
        os.makedirs(cluster_dir, exist_ok=True)
        cluster_directories[cluster_label] = cluster_dir

    # Copy files to their respective cluster directories
    for file_name, cluster_label in zip(file_names, cluster_labels):
        src_file_path = os.path.join(input_folder_path, file_name)
        dst_file_path = os.path.join(cluster_directories[cluster_label], file_name)
        shutil.copy(src_file_path, dst_file_path)

    # Merge data within each cluster directory
    for cluster_dir in cluster_directories.values():
        merge_cluster_data(cluster_dir)

    # Generate PDF report with cluster information
    pdf_path = os.path.join(output_folder_path, 'cluster_report.pdf')
    with PdfPages(pdf_path) as pdf:
        for cluster_label, cluster_dir in cluster_directories.items():
            cluster_files = os.listdir(cluster_dir)
            plt.figure()
            plt.text(0.5, 0.5, f'Cluster {cluster_label}\nFiles:\n' + '\n'.join(cluster_files),
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            pdf.savefig()
            plt.close()

if __name__ == "__main__":
    input_folder_path = r'C:\Users\praty\Downloads\Research\NormalizedCytometryFiles'
    output_folder_path = r'C:\Users\praty\Downloads\Research\ClusteredAgglomerativeSamples'
    n_clusters = 4  # Adjust the number of clusters as needed
    main(input_folder_path, output_folder_path, n_clusters)
