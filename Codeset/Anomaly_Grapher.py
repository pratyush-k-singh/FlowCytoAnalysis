import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def find_anomalies(data, std_multiplier=3.0, neighbor_limit=20):
    avg = np.mean(data)
    std_dev = np.std(data)
    threshold = std_multiplier * std_dev
    anomalies = np.where(np.abs(data - avg) > threshold)[0]
    
    # Group anomalies that are within `neighbor_limit` of each other
    groups = []
    current_group = []
    
    for i in anomalies:
        if not current_group or i - current_group[-1] <= neighbor_limit:
            current_group.append(i)
        else:
            if len(current_group) >= 3:
                groups.append(current_group)
            current_group = [i]
    
    if len(current_group) >= 3:
        groups.append(current_group)
    
    return groups

def plot_and_save(data, time, groups, pdf, col_name):
    plt.figure()
    
    for group in groups:
        start = max(group[0] - 20, 0)
        end = min(group[-1] + 20, len(data))
        plt.plot(time[start:end], data[start:end], marker='o', linestyle='-', markersize=5)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Anomalies in {col_name}')
    plt.grid(True)
    plt.tight_layout()
    
    pdf.savefig()
    plt.close()

def process_file(file_path, output_dir):
    df = pd.read_csv(file_path)
    time = df.iloc[:, 0]
    base_filename = os.path.basename(file_path)
    pdf_filename = os.path.join(output_dir, base_filename.replace('.csv', '.pdf'))
    
    with PdfPages(pdf_filename) as pdf:
        for col in df.columns[1:]:
            data = df[col]
            groups = find_anomalies(data)
            
            if not groups:
                continue
            
            plot_and_save(data, time, groups, pdf, col)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            process_file(file_path, output_dir)

input_directory = r'C:\Users\praty\Downloads\Research\NormalizedCytometryFiles'
output_directory = r'C:\Users\praty\Downloads\Research\Anomaly_Graphs_3.0'
process_directory(input_directory, output_directory)
