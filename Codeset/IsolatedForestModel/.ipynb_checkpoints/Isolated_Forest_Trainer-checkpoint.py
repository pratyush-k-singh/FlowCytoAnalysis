import os
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

def train_isolation_forest_models(input_folder_path, model_folder_path):
    """
    Train Isolation Forest models for each cluster based on individual CSV files in the input folder.

    Args:
        input_folder_path (str): Path to the folder containing CSV files with clustered data.
        model_folder_path (str): Path to the folder to save the trained models.
    """
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    for cluster_folder in os.listdir(input_folder_path):
        cluster_folder_path = os.path.join(input_folder_path, cluster_folder)
        if os.path.isdir(cluster_folder_path) and cluster_folder.startswith('Cluster_'):
            csv_files = [f for f in os.listdir(cluster_folder_path) if f.endswith('.csv') and f != 'Cluster_Combined_Data.csv']
            for csv_file in csv_files:
                data = pd.read_csv(os.path.join(cluster_folder_path, csv_file))
                model = IsolationForest(contamination=0.05, random_state=42)
                model.fit(data)
                model_file_path = os.path.join(model_folder_path, f'{csv_file.replace(".csv", "_model.pkl")}')
                joblib.dump(model, model_file_path)

def evaluate_anomalies(input_folder_path, model_folder_path):
    """
    Evaluate anomalies using trained Isolation Forest models and gather user feedback.

    Args:
        input_folder_path (str): Path to the folder containing CSV files with clustered data.
        model_folder_path (str): Path to the folder containing trained Isolation Forest models.
    """
    for cluster_folder in os.listdir(input_folder_path):
        cluster_folder_path = os.path.join(input_folder_path, cluster_folder)
        if os.path.isdir(cluster_folder_path) and cluster_folder.startswith('Cluster_'):
            csv_files = [f for f in os.listdir(cluster_folder_path) if f.endswith('.csv') and f != 'Cluster_Combined_Data.csv']
            for csv_file in csv_files:
                data = pd.read_csv(os.path.join(cluster_folder_path, csv_file))
                model_file_path = os.path.join(model_folder_path, f'{csv_file.replace(".csv", "_model.pkl")}')
                model = joblib.load(model_file_path)
                anomalies = model.predict(data)
                significant_anomalies = data[anomalies == -1]
                print(f"Significant anomalies found in {csv_file}:")
                print(significant_anomalies)
                for idx, row in significant_anomalies.iterrows():
                    feedback = input("Is this point a valid anomaly? (yes/no): ")
                    if feedback.lower() == 'no':
                        data.drop(index=idx, inplace=True)
                # Update model with feedback
                model.fit(data)
                # Save updated model
                joblib.dump(model, model_file_path)

def main():
    input_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\ClusteredFiles'
    model_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\Models'
    train_isolation_forest_models(input_folder_path, model_folder_path)
    evaluate_anomalies(input_folder_path, model_folder_path)

if __name__ == "__main__":
    main()
