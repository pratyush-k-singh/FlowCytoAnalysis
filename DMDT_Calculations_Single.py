import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def process_column(time, data, chunk_size=1000):
    num_chunks = len(data) // chunk_size

    mean_data = []
    mean_time = []
    for i in range(num_chunks):
        chunk = data[i*chunk_size : (i+1)*chunk_size]
        time_chunk = time[i*chunk_size : (i+1)*chunk_size]
        mean_data.append(chunk.mean())
        mean_time.append(time_chunk.mean())

    return np.array(mean_time), np.array(mean_data)

def calculate_dmdt(mean_time, mean_data):
    dm_values = []
    dt_values = []

    for i in range(len(mean_time)):
        for j in range(i + 1, len(mean_time)):
            dm = mean_data[j] - mean_data[i]
            dt = mean_time[j] - mean_time[i]
            dm_values.append(dm)
            dt_values.append(dt)

    return np.array(dm_values), np.array(dt_values)

def plot_dmdt(dm, dt, title):
    fig, ax = plt.subplots()
    ax.scatter(dt, dm, alpha=0.5)
    ax.set_xlabel('dt')
    ax.set_ylabel('dm')
    ax.set_title(title)
    ax.grid(True)
    return fig

def save_to_pdf(output_dir, filename, figures):
    pdf_path = os.path.join(output_dir, filename)
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)  # Close the figure after saving it to the PDF

def main(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    
    time = df['Time']
    
    # Get the first data column (excluding 'Time')
    data_columns = [col for col in df.columns if col != 'Time']
    if not data_columns:
        print("No data columns found in the CSV.")
        return

    first_data_column = data_columns[0]
    data = df[first_data_column]
    mean_time, mean_data = process_column(time, data)
    dm, dt = calculate_dmdt(mean_time, mean_data)
    fig = plot_dmdt(dm, dt, f'DMDT for {first_data_column}')
    
    base_filename = os.path.splitext(os.path.basename(input_csv))[0]
    output_filename = f'{base_filename}_{first_data_column}.pdf'
    save_to_pdf(output_dir, output_filename, [fig])

if __name__ == "__main__":
    input_file = r'C:\Users\praty\cytoflow\Codeset\NormalizedCytometryFiles\FCSC_WG1-001_AZGBBIO_Aurora_APC-Cy7-lyoPBMC-cell_SOP-p1_e2_1_1_1_FCS_TEST_NORMALIZED.csv'
    output_directory = r'C:\Users\praty\cytoflow\Codeset\DMDTGraphs'
    main(input_file, output_directory)
