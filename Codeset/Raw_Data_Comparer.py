import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def load_csv_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return df1, df2

def plot_columns(df1, df2, output_folder, exclude_column='time'):
    columns = df1.columns.difference([exclude_column])
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_path = os.path.join(output_folder, 'Comparison_Plots.pdf')
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

# Run code
file1 = r'C:\Users\praty\Downloads\Research\CSVCytometryFiles\FCSC_WG1-001_AZGBBIO_Aurora_V500C-lyoPBMC-cell_SOP-p1_e2_1_3_1_FCS_TEST.csv'
file2 = r'C:\Users\praty\Downloads\Research\CSVCytometryFiles\FCSC_WG1-001_AZGBBIO_Aurora_V450-lyoPBMC-cell_SOP-p1_e2_1_3_1_FCS_TEST.csv'
output_folder = 'TestComparison'

df1, df2 = load_csv_files(file1, file2)
plot_columns(df1, df2, output_folder)
