import os
import pandas as pd

def normalize_time(data, new_min=0, new_max=10000):
    """
    Normalize the time column in the data to a new scale.

    Args:
        data (pd.DataFrame): The DataFrame containing the time data.
        new_min (int, optional): The minimum value of the new scale. Defaults to 0.
        new_max (int, optional): The maximum value of the new scale. Defaults to 10000.

    Returns:
        pd.DataFrame: The DataFrame with the normalized time data.
    """
    time_col = data.columns[0]  # Assuming the first column is 'Time'
    old_min = data[time_col].min()
    old_max = data[time_col].max()
    # Linear transformation to scale time data
    data[time_col] = new_min + (data[time_col] - old_min) * (new_max - new_min) / (old_max - old_min)
    return data

def process_folder(input_folder_path, output_folder_path):
    """
    Process all CSV files in the input folder, normalize their time data, and save to the output folder.

    Args:
        input_folder_path (str): The path to the folder containing the original CSV files.
        output_folder_path (str): The path to the folder where the normalized CSV files will be saved.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output directory: {output_folder_path}")
    
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file_name)
            print(f"Processing file: {file_path}")
            data = pd.read_csv(file_path)
            
            # Normalize the time scale
            normalized_data = normalize_time(data)
            
            # Save the normalized data to a new CSV file
            output_file_name = file_name.replace('.csv', '_NORMALIZED.csv')
            output_file_path = os.path.join(output_folder_path, output_file_name)
            normalized_data.to_csv(output_file_path, index=False)
            print(f"Saved normalized file: {output_file_path}")

    print("Processing complete.")

# Example usage
input_folder_path = r'C:\Users\praty\Downloads\Research\CSVCytometryFiles'
output_folder_path = r'C:\Users\praty\Downloads\Research\NormalizedCytometryFiles'
process_folder(input_folder_path, output_folder_path)
