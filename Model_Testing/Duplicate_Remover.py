import pandas as pd
import os

def process_csv_files(input_dir):
    """
    Process all CSV files in the input directory, check for duplicate IDs,
    remove all occurrences of duplicates, and save the duplicates in a CSV file in the input directory.

    Parameters:
    input_dir (str): Directory containing the CSV files.
    """
    duplicates_file = os.path.join(input_dir, 'duplicate_events.csv')
    all_duplicates = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)
            
            duplicates = df[df.duplicated(subset=df.columns[0], keep=False)]
            all_duplicates.append(duplicates)
            
            unique_records = df.drop_duplicates(subset=df.columns[0], keep=False)
            
            unique_records.to_csv(file_path, index=False)
    
    if all_duplicates:
        all_duplicates_df = pd.concat(all_duplicates, ignore_index=True)
        all_duplicates_df.to_csv(duplicates_file, index=False)

if __name__ == "__main__":
    input_directory = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Filtered_Set\Event_Data'
    process_csv_files(input_directory)
