import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ks_2samp
from scipy.special import rel_entr
import numpy as np

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

def plot_columns(dfs, pdf_path, method, time_column='Time', exclude_points=100, colors=None):
    columns = dfs[0].columns.difference([time_column])
    
    with PdfPages(pdf_path) as pdf_pages:
        for column in columns:
            plt.figure(figsize=(10, 6))
            for i, df in enumerate(dfs):
                subdir_name = os.path.basename(os.path.dirname(df.attrs['file_path']))
                color = colors[subdir_name] if colors else None
                plt.plot(df[time_column][exclude_points:], df[column][exclude_points:], label=f'{subdir_name} File {i+1}', color=color)
            plt.title(f'Comparison of {method.capitalize()} {column}')
            plt.xlabel('Time')
            plt.ylabel(column)
            plt.legend()
            pdf_pages.savefig()
            plt.close()

def compute_statistical_tests(dfs, time_column='Time'):
    columns = dfs[0].columns.difference([time_column])
    results = {}

    for column in columns:
        column_results = {}
        for i in range(len(dfs)):
            for j in range(i + 1, len(dfs)):
                data1 = dfs[i][column]
                data2 = dfs[j][column]
                ks_stat, ks_p_value = ks_2samp(data1, data2)
                
                # Jensen-Shannon Divergence
                data1_hist, _ = np.histogram(data1, bins=50, density=True)
                data2_hist, _ = np.histogram(data2, bins=50, density=True)
                js_divergence = np.sum(rel_entr(data1_hist, data2_hist) + rel_entr(data2_hist, data1_hist)) / 2
                
                column_results[f'File {i+1} vs File {j+1}'] = {
                    'KS Test': (ks_stat, ks_p_value),
                    'Jensen-Shannon Divergence': js_divergence
                }
        results[column] = column_results
    return results

def save_statistical_tests_to_pdf(results, pdf_path):
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    
    with PdfPages(pdf_path) as pdf_pages:
        for column, tests in results.items():
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            plt.text(0.5, 0.5, f'Statistical Tests for {column}', fontsize=14, ha='center', va='center')
            pdf_pages.savefig()
            plt.close()

            for pair, stats in tests.items():
                plt.figure(figsize=(10, 6))
                plt.axis('off')
                text = f'{pair}\n\n'
                for test, result in stats.items():
                    text += f'{test}: {result}\n'
                plt.text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
                pdf_pages.savefig()
                plt.close()

def process_directory(directory, pdf_base_path):
    methods = ['mean', 'median']
    all_dfs = []
    all_tests_results = {}
    subdir_colors = {}
    color_palette = plt.get_cmap('tab10')
    
    for idx, (subdir, _, files) in enumerate(os.walk(directory)):
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

            # Assign colors to subdirectories
            subdir_colors[subdir_name] = color_palette(idx % 10)

            # Compute statistical tests for raw data
            tests_results = compute_statistical_tests(dfs, time_column='Time')
            all_tests_results[subdir_name] = tests_results

            # Save statistical tests to PDF
            stats_pdf_path = os.path.join(output_dir, 'Statistical_Tests.pdf')
            save_statistical_tests_to_pdf(tests_results, stats_pdf_path)
            print(f'Statistical Tests PDF saved to {stats_pdf_path}')
    
    # Create overall PDFs for all data
    if all_dfs:
        overall_pdf_mean = os.path.join(pdf_base_path, 'Comparison_Plots_Overall_Mean.pdf')
        overall_pdf_median = os.path.join(pdf_base_path, 'Comparison_Plots_Overall_Median.pdf')
        for method, pdf_path in zip(methods, [overall_pdf_mean, overall_pdf_median]):
            plot_columns(all_dfs, pdf_path, method, time_column='Time', exclude_points=100, colors=subdir_colors)
            print(f'Overall PDF ({method.capitalize()}) saved to {pdf_path}')
    
    return all_tests_results

if __name__ == "__main__":
    directory = r'C:\Users\praty\cytoflow\Codeset\ClusteredKMeanSamples'
    pdf_base_path = directory
    kmean_results = process_directory(directory, pdf_base_path)
    
    directory = r'C:\Users\praty\cytoflow\Codeset\ClusteredAgglomerativeSamples'
    pdf_base_path = directory
    agglomerative_results = process_directory(directory, pdf_base_path)
