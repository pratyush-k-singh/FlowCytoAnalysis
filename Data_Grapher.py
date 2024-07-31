import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_plots_for_csv(file_path, output_dir):
    df = pd.read_csv(file_path)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the PDF file
    pdf_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '.pdf'))
    with PdfPages(pdf_file_path) as pdf:
        time_column = df.columns[0]
        time_data = df[time_column]

        for column in df.columns[1:]:
            plt.figure()
            plt.plot(time_data, df[column])
            plt.xlabel(time_column)
            plt.ylabel(column)
            plt.title(f'{column} vs {time_column}')
            pdf.savefig()
            plt.close()

def process_directory(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            create_plots_for_csv(file_path, output_dir)

# Define your input and output directories
input_directory = r'C:\Users\praty\cytoflow\Codeset\CSVCytometryFiles'
output_directory = r'C:\Users\praty\cytoflow\Codeset\RawGraphs'

process_directory(input_directory, output_directory)
