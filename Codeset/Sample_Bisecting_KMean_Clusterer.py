import os
import shutil
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import skew

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
        skewness = skew(column_data)  # Calculate skewness instead of peaks
        features.extend([mean, std, skewness])
    return np.array(features)

def bisecting_kmeans(data, n_clusters):
    """
    Perform bisecting k-means clustering on the data.

    Args:
        data (np.array): Feature vectors.
        n_clusters (int): The number of clusters to form.

    Returns:
        np.array: An array of cluster labels.
    """    
    clusters = [data]
    labels = np.zeros(len(data), dtype=int)

    while len(clusters) < n_clusters:
        max_size_cluster_index = np.argmax([len(cluster) for cluster in clusters])
        cluster_to_split = clusters.pop(max_size_cluster_index)
        if len(cluster_to_split) <= 1:
            continue
        
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(cluster_to_split)
        split_labels = kmeans.labels_
        
        new_clusters = [
            cluster_to_split[split_labels == i] for i in range(2)
        ]
        clusters.extend(new_clusters)
        
        for i, new_cluster in enumerate(new_clusters):
            for point in new_cluster:
                index = np.where(np.all(data == point, axis=1))[0][0]
                labels[index] = len(clusters) - len(new_clusters) + i

    return labels

def merge_cluster_data(cluster_dir):
    """
    Merge data from all CSV files in a cluster into a single CSV file within the cluster directory.

    Args:
        cluster_dir (str): The directory containing the CSV files of the cluster.
    """
    data_frames = []
    for file_name in os.listdir(cluster_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(cluster_dir, file_name)
            data = pd.read_csv(file_path)
            if not data_frames:
                data_frames.append(data)
            else:
                data_frames.append(data.iloc[1:])

    merged_data = pd.concat(data_frames, ignore_index=True)
    merged_data.sort_values(by=merged_data.columns[0], inplace=True)
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

    feature_vectors = np.array(feature_vectors)
    cluster_labels = bisecting_kmeans(feature_vectors, n_clusters)

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
    os.environ['OMP_NUM_THREADS'] = '1'
    input_folder_path = r'C:\Users\praty\cytoflow\Codeset\NormalizedCytometryFiles'
    output_folder_path = r'C:\Users\praty\cytoflow\Codeset\ClusteredKMeanSamplesSkewBased'
    n_clusters = 5  # Adjust the number of clusters as needed
    main(input_folder_path, output_folder_path, n_clusters)
