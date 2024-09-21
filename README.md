# Irregular Data Analysis Repository

## Overview
This repository contains comprehensive data analysis for various forms of irregular data, along with various test models developed throughout the project. Due to security concerns, the final models and sensitive genetic and astronomical data have been excluded from this repository.

## Repository Contents

### Directories
- **AnalysisOutput**: Contains output files from various analyses performed on the flow cytometric data.
- **Anomaly_Graphs_2.0**: Graphs and plots generated to identify anomalies that devaiate by 2.0 standards.
- **Anomaly_Graphs_2.5**: Graphs and plots generated to identify anomalies that devaiate by 2.5 standards.
- **BinnedDMDT**: Binned Data Mining and Time (DMDT) results, providing insights into specific data segments.
- **ClusteredAgglomerativeSamples**: Samples clustered using agglomerative clustering techniques.
- **ClusteredIsolatedForestModel**: Data samples processed using the isolated forest model for anomaly detection.
- **ClusteredKMeanSamples**: Samples clustered using K-means clustering techniques.
- **DMDTGraphs**: Graphs and visualizations from DMDT analyses.
- **Extraneous Files**: Contains additional files used during data analysis that are not directly part of the core analysis.
- **Heatmaps**: Heatmap visualizations generated from the data analysis.
- **IsolatedForestModel**: Results and visualizations from the isolated forest model used for anomaly detection.
- **LogarithmicHeatmaps**: Heatmap visualizations using logarithmic scales.
- **Model_Testing**: Various test models and their results created during the project.
- **OptimizedDMDT**: Optimized DMDT analyses and their corresponding visualizations.
- **RawGraphs**: Raw graph outputs from initial data analysis steps.

### Python Scripts
- **Anomaly_Grapher.py**: Generates graphs and visualizations to identify anomalies in the dataset.
- **Channel_Trend_Heatmapper.py**: Creates heatmaps to visualize trends across different data channels.
- **Channel_Trend_Heatmapper_Logarithmic.py**: Similar to `Channel_Trend_Heatmapper.py`, but uses logarithmic scales for better visualization of trends.
- **Cytometry_Analysis.py**: Main script for performing flow cytometry data analysis.
- **DMDT_Calculations.py**: Performs DMDT calculations and generates related visualizations.
- **DMDT_Calculations_Single.py**: Executes DMDT calculations for individual data samples.
- **Data_Converter.py**: Converts raw data into a format suitable for analysis.
- **Data_Grapher.py**: Generates various graphs and plots from the data.
- **File_Channel_Ratioing.py**: Calculates and analyzes channel ratios in the data files.
- **Raw_Data_Comparer.py**: Compares raw data across different samples.
- **Running_Data_Comparer.py**: Compares running data sets for trend analysis.
- **Running_Data_File_Comparer.py**: Compares running data files for consistency checks.
- **Sample_Agglomerative_Clusterer.py**: Applies agglomerative clustering to sample data.
- **Sample_Bisecting_KMean_Clusterer.py**: Uses bisecting K-means clustering on sample data.
- **Time_Normalizer.py**: Normalizes time data for consistency in analysis.
