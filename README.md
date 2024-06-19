# Cyto Flow

03/31/2024:
- Set up basics for the training of Isolation Forest model to detect rare events in Flow Cytometry Data, including a feedback system and a looping training system.
- Modified the CytometryAnalysis file to cluster data based on correlation values via agglomerative clustering. Removed t-SNE modelling for now, pending further review and study.

05/07/2024:
- Implemented the clustering of samples based on PCA dimensionality reduction of samples
- Modified data analysis too analyize clusters of samples, merging the data together before completing the analysis process

Order of file usage for large scale data analysis:
- Data_Converter.py
- Time_Normalizer.py
- Sample_Clusterer.py
- Cytometry_Analysis.py

05/12/2024:
- Began implementation of new Isolated Forest Model design, meant to operate by clustering the initial training data, and then matchin inputted data to one of the resulting clusters based on a distance matrix. Then, the file will be fed into a model built of that clustered data _only_. Resultingly, there will be one model for each cluster.

05/20/2024:
- Completed initial implementation of Isolated Forest Model design outlined above; testing pending
- Fixed issues with the analysis of the clustered data caused by poor logic when merging the data in each cluster

06/15/2024:
- Implemented the use of bisecting k-mean clustering into the code
- Implemented code visualization scripts to determine efficacy of potential clustering algorithms

06/19/2024:
- Implemented heatmap visualization of the mean, median, and skew of flow cytometry data
- Implemented rudimentary anomaly detection within flow cytometry files to the 2.0, 2.5, and 3.0 standard deviations degrees
