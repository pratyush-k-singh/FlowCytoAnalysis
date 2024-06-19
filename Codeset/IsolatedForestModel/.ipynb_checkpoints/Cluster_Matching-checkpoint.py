import os
import pandas as pd
from FlowCytometryTools import FCMeasurement
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import shutil

def process_folder(input_folder_path, output_folder_path):
    """
    Process FCS files in the input folder and save them as CSV files in the output folder.

    Args:
        input_folder_path (str): Path to the folder containing FCS files.
        output_folder_path (str): Path to the output folder.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.fcs'):
            file_path = os.path.join(input_folder_path, file_name)
            sample = FCMeasurement(ID='TestSample', datafile=file_path)
            data = sample.data
            
            # Save the preprocessed data to a new CSV file in RawData folder
            output_file_path = os.path.join(output_folder_path, 'RawData', file_name.replace('.fcs', '.csv'))
            data.to_csv(output_file_path, index=False)

def normalize_time(data, new_min=0, new_max=10):
    """
    Normalize the time column in the data to a new scale.

    Args:
        data (pd.DataFrame): DataFrame containing the time data.
        new_min (int, optional): The minimum value of the new scale. Defaults to 0.
        new_max (int, optional): The maximum value of the new scale. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with the normalized time data.
    """
    time_col = data.columns[0]  # Assuming the first column is 'Time'
    old_min = data[time_col].min()
    old_max = data[time_col].max()
    # Linear transformation to scale time data
    data[time_col] = new_min + (data[time_col] - old_min) * (new_max - new_min) / (old_max - old_min)
    return data

def process_and_normalize(input_folder_path, output_folder_path):
    """
    Normalize the time data in CSV files in the input folder and save them to the output folder.

    Args:
        input_folder_path (str): Path to the folder containing CSV files.
        output_folder_path (str): Path to the output folder.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file_name)
            data = pd.read_csv(file_path)
            
            # Normalize the time scale
            normalized_data = normalize_time(data)
            
            # Save the normalized data to a new CSV file in NormalizedData folder
            output_file_path = os.path.join(output_folder_path, file_name)
            normalized_data.to_csv(output_file_path, index=False)

def calculate_distance_matrix(data):
    """
    Calculate the pairwise Euclidean distance matrix.

    Args:
        data (pd.DataFrame): DataFrame containing the data.

    Returns:
        np.ndarray: Pairwise Euclidean distance matrix.
    """
    pairwise_dist = pdist(data.values, metric='euclidean')
    distance_matrix = squareform(pairwise_dist)
    return distance_matrix

def assign_to_cluster(file_path, cluster_data_paths, output_folder_path):
    """
    Assign a file to the closest cluster based on the distance matrix.

    Args:
        file_path (str): Path to the file.
        cluster_data_paths (dict): Dictionary containing cluster labels as keys and file paths as values.
        output_folder_path (str): Path to the output folder.
    """
    data = pd.read_csv(file_path)
    file_distance_matrix = calculate_distance_matrix(data)
    min_distance = float('inf')
    closest_cluster = None
    
    for cluster_label, cluster_path in cluster_data_paths.items():
        cluster_data = pd.read_csv(cluster_path)
        cluster_distance_matrix = calculate_distance_matrix(cluster_data)
        distance = ((file_distance_matrix - cluster_distance_matrix) ** 2).sum()
        
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster_label
    
    output_file_name = os.path.basename(file_path).replace('_normalized.csv', f'_cluster_{closest_cluster}.csv')
    output_file_path = os.path.join(output_folder_path, output_file_name)
    data.to_csv(output_file_path, index=False)

def main():
    input_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\InputFiles'
    raw_data_output_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\RawData'
    process_folder(input_folder_path, raw_data_output_path)
    
    input_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\RawData'
    output_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\NormalizedData'
    process_and_normalize(input_folder_path, output_folder_path)
    
    cluster_data_paths = {
        'Cluster_1': r'C:\Users\praty\Downloads\Research\ClusteredSamples\Cluster 1\Cluster_Combined_Data',
        'Cluster_2': r'C:\Users\praty\Downloads\Research\ClusteredSamples\Cluster 2\Cluster_Combined_Data',
        'Cluster_3': r'C:\Users\praty\Downloads\Research\ClusteredSamples\Cluster 3\Cluster_Combined_Data',
        'Cluster_4': r'C:\Users\praty\Downloads\Research\ClusteredSamples\Cluster 4\Cluster_Combined_Data',
        'Cluster_5': r'C:\Users\praty\Downloads\Research\ClusteredSamples\Cluster 5\Cluster_Combined_Data',
    }

    input_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\NormalizedData'
    output_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\MatchedData'
    for file_name in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file_name)
        assign_to_cluster(file_path, cluster_data_paths, output_folder_path)

if __name__ == "__main__":
    main()
