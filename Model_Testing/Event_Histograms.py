import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def create_histograms(input_csv, output_pdf, output_dir):
    """
    Create histograms for each event type and save them into a single PDF.

    Parameters:
    input_csv (str): Path to the input CSV file.
    output_pdf (str): Name of the output PDF file.
    output_dir (str): Directory to save the output PDF file.
    """
    df = pd.read_csv(input_csv)
    event_columns = [col for col in df.columns if col.endswith('_dnn') or col.endswith('_xgb')]
    event_names = set(col.split('_')[0] for col in event_columns)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_path = os.path.join(output_dir, output_pdf)
    
    with PdfPages(pdf_path) as pdf:
        for event_name in event_names:
            dnn_col = f"{event_name}_dnn"
            xgb_col = f"{event_name}_xgb"

            if dnn_col in df.columns and xgb_col in df.columns:
                plt.figure()
                df[dnn_col].plot(kind='hist', alpha=0.5, label=f'{event_name}_dnn')
                df[xgb_col].plot(kind='hist', alpha=0.5, label=f'{event_name}_xgb')
                plt.yscale('log')
                plt.xlabel('Confidence Level')
                plt.ylabel('Frequency (log scale)')
                plt.title(f'Histogram for {event_name}')
                plt.legend()
                pdf.savefig()
                plt.close()

if __name__ == "__main__":
    input_csv = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Model_Testing_Data\field_776.csv'
    output_directory = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Event_Histograms'
    output_pdf = os.path.basename(input_csv).replace('.csv', '_histograms.pdf')

    create_histograms(input_csv, output_pdf, output_directory)
