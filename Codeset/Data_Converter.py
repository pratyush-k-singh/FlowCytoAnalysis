import os
from FlowCytometryTools import FCMeasurement
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def process_folder(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.fcs'):
            file_path = os.path.join(input_folder_path, file_name)
            sample = FCMeasurement(ID='TestSample', datafile=file_path)
            data = sample.data
            
            # Save the preprocessed data to a new CSV file
            output_file_name = file_name.replace('.fcs', '_TEST.csv')
            output_file_path = os.path.join(output_folder_path, output_file_name)
            data.to_csv(output_file_path, index=False)

input_folder_path = r'C:\Users\praty\Downloads\Research\CytometryFiles'
output_folder_path = r'C:\Users\praty\Downloads\Research\CSVCytometryFiles'
process_folder(input_folder_path, output_folder_path)
