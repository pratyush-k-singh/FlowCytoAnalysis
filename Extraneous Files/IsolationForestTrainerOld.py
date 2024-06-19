import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import os

class OutlierDetector:
    def __init__(self, contamination=0.1, random_state=None):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def train_model(self, X):
        self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.model.fit(X)

    def detect_outliers(self, X):
        if self.model is None:
            raise Exception("Model not trained. Call train_model() first.")
        # The lower the score, the more abnormal the data point
        scores = self.model.decision_function(X)
        return self.model.predict(X), scores

    def feedback(self, X, identified_outliers):
        if self.model is None:
            raise Exception("Model not trained. Call train_model() first.")
        # Ask for feedback on identified outliers
        print("Identified outliers:")
        print(identified_outliers)
        feedback = input("Enter feedback (1 for correct, 0 for incorrect): ")
        feedback_array = np.array([int(x) for x in feedback.split()])
        if len(feedback_array) != len(identified_outliers):
            raise ValueError("Feedback length does not match identified outliers length.")
        return feedback_array

    def update_model(self, X, feedback_array):
        # Update the model based on feedback
        X_feedback = X[feedback_array == 1]  # Get the instances marked as correct outliers
        if len(X_feedback) > 0:
            self.model.fit(X_feedback)  # Refit the model with the updated data

if __name__ == "__main__":
    input_folder_path = r'C:\Users\praty\Downloads\ProcessedCytometryFiles'
    csv_files = [file for file in os.listdir(input_folder_path) if file.endswith('.csv')]
    outlier_detector = OutlierDetector(contamination=0.1, random_state=42)

    for csv_file in csv_files:
        file_path = os.path.join(input_folder_path, csv_file)
        data = pd.read_csv(file_path)
        X = data.values
        outlier_detector.train_model(X)
        outliers, scores = outlier_detector.detect_outliers(X)
    
        # Sort the scores and get the indices of the most anomalous data points
        sorted_idx = np.argsort(scores)[:10]  # Get indices of top 10 anomalies
        print(f"Top 10 anomaly scores and their data points:")
        for idx in sorted_idx:
            print(f"Score: {scores[idx]:.2f}, Data Point: {X[idx]}")

        # Provide feedback interactively on identified outliers
        feedback_array = outlier_detector.feedback(X, outliers)

        # Update the model based on feedback
        outlier_detector.update_model(X, feedback_array)

        # Check if any incorrect outliers were marked
        all_correct = np.all(feedback_array == 1)
        if all_correct:
            print("All outliers correctly identified.")
            break
        else:
            print("Incorrect outliers detected, updating the model...")
