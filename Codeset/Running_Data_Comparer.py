import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def load_csv_files(files):
    dfs = [pd.read_csv(file) for file in files]
    return dfs

def calculate_running_statistics(dfs, time_column='Time', method='mean'):
    running_dfs = []
    for df in dfs:
        if time_column not in df.columns:
            raise KeyError(f"Column '{time_column}' not found in DataFrame")
        running_df = pd.DataFrame(index=df.index)
        running_df[time_column] = df[time_column]
        for column in df.columns:
            if column == time_column:
                continue
            if method == 'mean':
                running_df[column] = df[column].expanding().mean()
            elif method == 'median':
                running_df[column] = df[column].expanding().median()
        running_dfs.append(running_df)
    return running_dfs

def plot_columns(dfs, pdf_path, method, time_column='Time', exclude_points=100):
    columns = dfs[0].columns.difference([time_column])
    subdir_colors = {}

    with PdfPages(pdf_path) as pdf_pages:
        for column in columns:
            plt.figure(figsize=(10, 6))
            for i, df in enumerate(dfs):
                subdir_name = os.path.basename(os.path.dirname(df.attrs['file_path']))
                if subdir_name not in subdir_colors:
                    subdir_colors[subdir_name] = plt.get_cmap('tab10')(len(subdir_colors) % 10)
                plt.plot(df[time_column][exclude_points:], df[column][exclude_points:], 
                         label=f'{subdir_name} File {i+1}', color=subdir_colors[subdir_name])
            plt.title(f'Comparison of {method.capitalize()} {column}')
            plt.xlabel('Time')
            plt.ylabel(column)
            plt.legend()
            pdf_pages.savefig()
            plt.close()

def process_directory(directory, pdf_base_path):
    methods = ['mean', 'median']
    all_dfs = []
    
    for subdir, _, files in os.walk(directory):
        csv_files = [os.path.join(subdir, file) for file in files if file.startswith('FCSC') and file.endswith('.csv')]
        if len(csv_files) >= 2:
            for method in methods:
                subdir_name = os.path.basename(subdir)
                output_dir = os.path.join(pdf_base_path, subdir_name)
                os.makedirs(output_dir, exist_ok=True)
                pdf_path = os.path.join(output_dir, f'Comparison_Plots_Running_{method.capitalize()}.pdf')
                dfs = load_csv_files(csv_files)
                running_dfs = calculate_running_statistics(dfs, method=method)
                for df, file in zip(running_dfs, csv_files):
                    df.attrs['file_path'] = file  # Store the file path in the DataFrame's attributes
                plot_columns(running_dfs, pdf_path, method, time_column='Time', exclude_points=100)
                print(f'PDF saved to {pdf_path}')
            all_dfs.extend(running_dfs)
    
    # Create overall PDFs for all data
    if all_dfs:
        overall_pdf_mean = os.path.join(pdf_base_path, 'Comparison_Plots_Overall_Mean.pdf')
        overall_pdf_median = os.path.join(pdf_base_path, 'Comparison_Plots_Overall_Median.pdf')
        for method, pdf_path in zip(methods, [overall_pdf_mean, overall_pdf_median]):
            plot_columns(all_dfs, pdf_path, method, time_column='Time', exclude_points=100)
            print(f'Overall PDF ({method.capitalize()}) saved to {pdf_path}')

if __name__ == "__main__":
    directory = r'C:\Users\praty\cytoflow\Codeset\ClusteredKMeanSamplesSkewBased'
    pdf_base_path = directory
    process_directory(directory, pdf_base_path)
