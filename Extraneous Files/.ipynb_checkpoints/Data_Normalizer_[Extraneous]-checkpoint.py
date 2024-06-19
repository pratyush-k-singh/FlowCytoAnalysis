import os
import pandas as pd
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def resample_data(df, target_rows, time_units):
    resampled_series_list = []
    # Resample the time column to match the target time units
    time_column_resampled = np.linspace(df.iloc[0, 0], time_units, target_rows)
    resampled_series_list.append(time_column_resampled)
    
    for column in df.columns[1:]:  # Exclude the first column (time column)
        # Resample the data using linear interpolation
        resampled_series = np.interp(
            np.linspace(0, len(df[column])-1, target_rows), 
            np.arange(len(df[column])), 
            df[column]
        )
        resampled_series_list.append(resampled_series)
    
    resampled_df = pd.DataFrame(resampled_series_list).transpose()
    resampled_df.columns = df.columns  # Assign the correct column names
    return resampled_df

def normalize_data_frames(data_frames, output_folder):
    # Determine the minimum number of rows and corresponding time units
    min_rows = min(df.shape[0] for df in data_frames)
    min_time_units = min(df.iloc[-1, 0] for df in data_frames)  # Last value of the time column
    
    # Normalize each data frame to have the same number of rows and time units
    normalized_dfs = []
    for df in data_frames:
        normalized_df = resample_data(df, min_rows, min_time_units)
        normalized_dfs.append(normalized_df)
        
        # Save the normalized DataFrame to the specified output folder
        file_name = os.path.basename(df.file_path).replace('_CLEAN.csv', '_NORMALIZED.csv')
        normalized_file_path = os.path.join(output_folder, file_name)
        normalized_df.to_csv(normalized_file_path, index=False)
    
    print(f"All files have been normalized to {min_rows} rows and {min_time_units} time units.")

def main(input_folder_path, output_folder_path):
    data_frames = []
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('_CLEAN.csv'):  # Process only preprocessed CSV files
            file_path = os.path.join(input_folder_path, file_name)
            df = load_data(file_path)
            df.file_path = file_path  # Set the file_path attribute
            data_frames.append(df)

    normalize_data_frames(data_frames, output_folder_path)

if __name__ == "__main__":
    input_folder_path = r'C:\Users\praty\Downloads\Research\ProcessedCytometryFiles'
    output_folder_path = r'C:\Users\praty\Downloads\Research\NormalizedCytometryFiles'
    main(input_folder_path, output_folder_path)
