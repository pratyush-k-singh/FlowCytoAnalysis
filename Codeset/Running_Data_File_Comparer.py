import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def load_csv_files(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df.attrs['file_path'] = file  # Store the file path in the DataFrame's attributes
        dfs.append(df)
    return dfs

def calculate_running_statistics(dfs, method='mean'):
    running_dfs = []
    for df in dfs:
        running_df = pd.DataFrame(index=df.index)
        running_df['time'] = df['time']
        for column in df.columns:
            if column == 'time':
                continue
            if method == 'mean':
                running_df[column] = df[column].expanding().mean()
            elif method == 'median':
                running_df[column] = df[column].expanding().median()
        running_dfs.append(running_df)
    return running_dfs

def plot_columns(dfs, pdf_path, exclude_column='time', exclude_points=100):
    columns = dfs[0].columns.difference([exclude_column])
    
    with PdfPages(pdf_path) as pdf_pages:
        for column in columns:
            plt.figure(figsize=(10, 6))
            for i, df in enumerate(dfs):
                subdir_name = os.path.basename(os.path.dirname(df.attrs['file_path']))
                plt.plot(df['time'][exclude_points:], df[column][exclude_points:], label=f'{subdir_name} File {i+1}')
            plt.title(f'Comparison of {column}')
            plt.xlabel('Time')
            plt.ylabel(column)
            plt.legend()
            pdf_pages.savefig()
            plt.close()

def process_files(file_list, pdf_path):
    dfs = load_csv_files(file_list)
    running_mean_dfs = calculate_running_statistics(dfs, method='mean')
    plot_columns(running_mean_dfs, pdf_path, exclude_points=100)
    print(f'PDF saved to {pdf_path}')

if __name__ == "__main__":
    # List of specific files to process
    files_to_process = [
        r'C:\Users\praty\Downloads\Research\ClusteredKMeanSamples\FCSC_file1.csv',
        r'C:\Users\praty\Downloads\Research\ClusteredKMeanSamples\FCSC_file2.csv',
        r'C:\Users\praty\Downloads\Research\ClusteredAgglomerativeSamples\FCSC_file3.csv',
        r'C:\Users\praty\Downloads\Research\ClusteredAgglomerativeSamples\FCSC_file4.csv',
    ]
    
    pdf_path = r'C:\Users\praty\Downloads\Research\Comparison_Plots_Running_Mean.pdf'
    process_files(files_to_process, pdf_path)
