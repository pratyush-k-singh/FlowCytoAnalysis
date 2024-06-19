import os
import pandas as pd
import joblib

def predict_with_isolated_forest(input_folder_path, output_folder_path, model_folder_path):
    """
    Predict labels for files in the input folder using pre-trained Isolation Forest models and save the results.

    Args:
        input_folder_path (str): Path to the folder containing CSV files with clustered data.
        output_folder_path (str): Path to the output folder to save the predicted labels.
        model_folder_path (str): Path to the folder containing pre-trained Isolation Forest models.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Load pre-trained models
    models = {}
    for model_file in os.listdir(model_folder_path):
        if model_file.endswith('.pkl'):
            file_name = model_file.replace('_model.pkl', '')
            models[file_name] = joblib.load(os.path.join(model_folder_path, model_file))

    # Predict labels for files using corresponding models
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file_name)
            model_key = file_name.replace('.csv', '')
            if model_key in models:
                model = models[model_key]
                data = pd.read_csv(file_path)
                predictions = model.predict(data)
                output_file_path = os.path.join(output_folder_path, file_name.replace('.csv', '_predicted.csv'))
                pd.DataFrame(predictions, columns=['Prediction']).to_csv(output_file_path, index=False)

def main():
    input_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\ClusteredFiles'
    output_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\PredictedData'
    model_folder_path = r'C:\Users\praty\Downloads\Research\IsolatedForestModel\Models'
    predict_with_isolated_forest(input_folder_path, output_folder_path, model_folder_path)

if __name__ == "__main__":
    main()
