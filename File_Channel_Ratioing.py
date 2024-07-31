import pandas as pd
import os

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    columns = df.columns.tolist()
    
    for index, row in df.iterrows():
        basis_value = row[columns[2]]
        
        if pd.api.types.is_numeric_dtype(basis_value) and basis_value != 0:
            for col in columns[1:]:
                df.at[index, col] = row[col] / basis_value
        else:
            print(f"Warning: Skipping row {index} due to invalid basis value: {basis_value}")
    
    df.to_csv(output_file, index=False)

def process_all_csv_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            process_csv(input_file, output_file)
            print(f"Processed {filename}")

# Example usage
input_directory = r'C:\Users\praty\cytoflow\Codeset\NormalizedCytometryFiles'
output_directory = r'C:\Users\praty\cytoflow\Codeset\RatioedFiles'
process_all_csv_files(input_directory, output_directory)
