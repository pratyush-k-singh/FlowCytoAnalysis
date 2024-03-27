import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_summary_statistics(data):
    return data.describe()

def calculate_correlation_matrix(data):
    # Exclude non-numeric columns when calculating the correlation matrix
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    return data[numeric_cols].corr()

def plot_heatmap(correlation_matrix, pdf):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    pdf.savefig()
    plt.close()

def plot_pca(data, pdf):
    # Apply PCA to the numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data[numeric_cols])
    
    # Create a scatter plot of the two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
    plt.xlabel(f'Principal Component 1 - {pca.explained_variance_ratio_[0]:.2f}% variance')
    plt.ylabel(f'Principal Component 2 - {pca.explained_variance_ratio_[1]:.2f}% variance')
    plt.title('PCA Scatter Plot')
    pdf.savefig()
    plt.close()

def plot_tsne(data, pdf):
    # Apply t-SNE to the numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data[numeric_cols])
    
    # Create a scatter plot of the two t-SNE components
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Scatter Plot')
    pdf.savefig()
    plt.close()

def generate_summary_statistics_table(data, pdf):
    summary_stats = calculate_summary_statistics(data)
    pdf.savefig(pd.DataFrame(summary_stats).to_html())
    plt.close()

def plot_peak_points(data, pdf):
    # Define a minimum distance between peaks
    min_distance = 1000  # 1000 data points apart

    plt.figure(figsize=(14, 7))

    # Identify columns that end with '-H'
    h_columns = [col for col in data.columns if col.endswith('-H')]

    # Find and plot peaks for each '-H' column
    for col in h_columns:
        # Find peaks
        peaks, _ = find_peaks(data[col], distance=min_distance)
        # Get the six highest peaks
        peak_values = data[col].iloc[peaks].nlargest(6)
        peak_times = data['Time'].iloc[peak_values.index]
        # Scatter plot
        plt.scatter(peak_times, peak_values, label=col)

    plt.xlabel('Time')
    plt.ylabel('Detected Concentration')
    plt.title('Six Highest Peak Points for Each Detection Type')
    plt.legend()
    pdf.savefig()
    plt.close()

def analyze_file(file_path, output_folder):
    data = load_data(file_path)
    correlation_matrix = calculate_correlation_matrix(data)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data[numeric_cols])

    # Create a PDF file
    pdf_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + '.pdf')
    with PdfPages(pdf_path) as pdf:
        plot_heatmap(correlation_matrix, pdf)
        plot_pca(data, pdf)
        plot_tsne(data, pdf)
        plot_peak_points(data, pdf)  # Add this line to include the peak points plot
        generate_summary_statistics_table(data, pdf)
        plt.close()
    
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
