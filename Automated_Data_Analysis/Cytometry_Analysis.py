import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_summary_statistics(data):
    return data.describe()

def calculate_correlation_matrix(data):
    # Exclude the 'Time' column when calculating the correlation matrix
    return data.drop('Time', axis=1).corr()

def plot_heatmap(correlation_matrix, pdf):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    pdf.savefig()
    plt.close()

def plot_pca(data, pdf):
    # Exclude the 'Time' column from PCA
    data_to_scale = data.drop('Time', axis=1)
    
    # Apply PCA directly since the data is already scaled
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data_to_scale)
    
    # Create a scatter plot of the two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
    plt.xlabel(f'Principal Component 1 - {pca.explained_variance_ratio_[0]:.2f}% variance')
    plt.ylabel(f'Principal Component 2 - {pca.explained_variance_ratio_[1]:.2f}% variance')
    plt.title('PCA Scatter Plot')
    pdf.savefig()
    plt.close()

def generate_summary_statistics_table(data, pdf):
    summary_stats = calculate_summary_statistics(data)
    pdf.savefig(pd.DataFrame(summary_stats).to_html())
    plt.close()

def analyze_file(file_path, output_folder):
    data = load_data(file_path)
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
        if file_name.endswith('_CLEAN.csv'):  # Process only preprocessed CSV files
            file_path = os.path.join(input_folder_path, file_name)
            analyze_file(file_path, output_folder_path)

if __name__ == "__main__":
    input_folder_path = r'C:\Users\praty\Downloads\ProcessedCytometryFiles'
    output_folder_path = r'C:\Users\praty\Downloads\AnalysisOutput'
    main(input_folder_path, output_folder_path)
