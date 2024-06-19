import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse

def load_csv_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return df1, df2

def calculate_reduced_data(df, method='mean'):
    reduced_df = pd.DataFrame()
    for column in df.columns:
        if column == 'time':
            continue
        if method == 'mean':
            reduced_df[column] = df[column].groupby(df.index // 10).mean()
        elif method == 'median':
            reduced_df[column] = df[column].groupby(df.index // 10).median()
    return reduced_df

def plot_columns(df1, df2, output_folder, exclude_column='time', filename='comparison_plots.pdf'):
    columns = df1.columns.difference([exclude_column])
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_path = os.path.join(output_folder, filename)
    pdf_pages = PdfPages(pdf_path)
    
    for column in columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df1.index, df1[column], label='File 1', color='blue')
        plt.plot(df2.index, df2[column], label='File 2', color='orange')
        plt.title(f'Comparison of {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.legend()
        pdf_pages.savefig()
        plt.close()
    
    pdf_pages.close()
    print(f'PDF saved to {pdf_path}')

if __name__ == "__main__":
    file1 = r'C:\Users\praty\Downloads\Research\CSVCytometryFiles\FCSC_WG1-001_AZGBBIO_Aurora_V500C-lyoPBMC-cell_SOP-p1_e2_1_3_1_FCS_TEST.csv'
    file2 = r'C:\Users\praty\Downloads\Research\CSVCytometryFiles\FCSC_WG1-001_AZGBBIO_Aurora_V450-lyoPBMC-cell_SOP-p1_e2_1_3_1_FCS_TEST.csv'
    output_folder = 'TestComparison'
    
    df1, df2 = load_csv_files(file1, file2)

    # Plot original data
    plot_columns(df1, df2, output_folder, filename='Comparison_Plots.pdf')

    # Plot reduced data using mean
    reduced_df1_mean = calculate_reduced_data(df1, method='mean')
    reduced_df2_mean = calculate_reduced_data(df2, method='mean')
    plot_columns(reduced_df1_mean, reduced_df2_mean, output_folder, filename='Comparison_Plots_Mean.pdf')

    # Plot reduced data using median
    reduced_df1_median = calculate_reduced_data(df1, method='median')
    reduced_df2_median = calculate_reduced_data(df2, method='median')
    plot_columns(reduced_df1_median, reduced_df2_median, output_folder, filename='Comparison_Plots_Median.pdf')
