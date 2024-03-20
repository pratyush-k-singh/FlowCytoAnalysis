import os
import pandas as pd
from FlowCytometryTools import FCMeasurement
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages

def load_data(file_path):
    sample = FCMeasurement(ID='TestSample', datafile=file_path)
    return sample.data, sample.meta

def calculate_summary_statistics(data):
    return data.describe()

def calculate_correlation_matrix(data):
    return data.iloc[:, 1:].corr()

def plot_heatmap(correlation_matrix, pdf):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    pdf.savefig()
    plt.close()

def plot_pca(data, pdf):
    time_column = data.iloc[:, 0]
    data_to_scale = data.iloc[:, 1:]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns[1:], index=data.index)
    final_data = pd.concat([time_column, scaled_data_df], axis=1)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_data)
    plt.scatter(pca_results[:, 0], pca_results[:, 1])
    plt.title('PCA - First Two Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    pdf.savefig()
    plt.close()

def generate_summary_statistics_table(data, pdf):
    summary_stats = calculate_summary_statistics(data)
    pdf.savefig(pd.DataFrame(summary_stats).to_html())
    plt.close()

def analyze_file(file_path, output_folder):
    data, _ = load_data(file_path)
    correlation_matrix = calculate_correlation_matrix(data)

    # Create a PDF file
    pdf_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + '.pdf')
    with PdfPages(pdf_path) as pdf:
        plot_heatmap(correlation_matrix, pdf)
        plot_pca(data, pdf)
        generate_summary_statistics_table(data, pdf)

def main(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.fcs'):
            file_path = os.path.join(input_folder_path, file_name)
            analyze_file(file_path, output_folder_path)

if __name__ == "__main__":
    input_folder_path = r'C:\Users\praty\Downloads\CytometryFiles'
    output_folder_path = r'C:\Users\praty\Downloads\AnalysisOutput'
    main(input_folder_path, output_folder_path)
