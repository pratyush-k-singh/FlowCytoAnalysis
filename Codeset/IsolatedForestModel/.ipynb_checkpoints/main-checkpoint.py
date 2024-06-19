import os
import pandas as pd
from data_processing import load_data, preprocess_data
from clustering import assign_to_cluster
from anomaly_detection import train_isolation_forest, detect_anomalies
from user_feedback import user_feedback, retrain_model

def main(input_folder_path, output_folder_path, cluster_data_paths):
    """
    Main function to process and analyze CSV files from each cluster directory.

    Args:
        input_folder_path (str): The main directory containing new data files.
        output_folder_path (str): The directory to save the analysis reports.
        cluster_data_paths (dict): Dictionary mapping cluster numbers to data file paths for training IsolationForest models.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Train IsolationForest models for each cluster
    cluster_models = {}
    for cluster_num, folder_path in cluster_data_paths.items():
        data_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        data = pd.concat([preprocess_data(load_data(f)) for f in data_files], ignore_index=True)
        model = train_isolation_forest(data)
        cluster_models[cluster_num] = model

    # Process new data files
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file_name)
            new_data = preprocess_data(load_data(file_path))
            reference_data = pd.concat([preprocess_data(load_data(os.path.join(cluster_data_paths[cluster_num], f))) for cluster_num in cluster_data_paths for f in os.listdir(cluster_data_paths[cluster_num]) if f.endswith('.csv')], ignore_index=True)
            assigned_cluster = assign_to_cluster(new_data, reference_data)
            model = cluster_models[assigned_cluster]
            scores, predictions = detect_anomalies(model, new_data)

            # Save analysis report
            pdf_path = os.path.join(output_folder_path, f'Analysis_{file_name}.pdf')
            with PdfPages(pdf_path) as pdf:
                feedback_data = user_feedback(new_data, scores, predictions)
                retrained_model = retrain_model(model, new_data, feedback_data)
                cluster_models[assigned_cluster] = retrained_model  # Update the model for future use

                # Plot and save results (omitted for brevity, similar to previous plotting functions)

if __name__ == "__main__":
    input_folder_path = r'C:\Users\praty\Downloads\NewDataFiles'
    output_folder_path = r'C:\Users\praty\Downloads\AnalysisOutput'
    cluster_data_paths = {
        1: r'C:\Users\praty\Downloads\ClusteredSamples\Cluster 1',
        2: r'C:\Users\praty\Downloads\ClusteredSamples\Cluster 2',
        3: r'C:\Users\praty\Downloads\ClusteredSamples\Cluster 3',
        4: r'C:\Users\praty\Downloads\ClusteredSamples\Cluster 4',
        5: r'C:\Users\praty\Downloads\ClusteredSamples\Cluster 5'
    }
    main(input_folder_path, output_folder_path, cluster_data_paths)
