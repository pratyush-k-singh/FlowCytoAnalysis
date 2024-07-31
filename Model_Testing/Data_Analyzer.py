import pandas as pd
import os

class DataAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the DataAnalyzer with a CSV file.

        Parameters:
        file_path (str): Path to the CSV file containing the data.
        """
        self.df = pd.read_csv(file_path)
        
    def subset_by_values(self, col1, col2, threshold1, threshold2=None):
        """
        Subset the DataFrame based on threshold values for two columns.

        Parameters:
        col1 (str): Column name for the first criterion.
        col2 (str): Column name for the second criterion.
        threshold1 (float): Threshold for the first criterion.
        threshold2 (float): Threshold for the second criterion (default is same as threshold1).

        Returns:
        DataFrame: Subset of the original DataFrame.
        """
        if threshold2 is None:
            threshold2 = threshold1
        mask = (self.df[col1] >= threshold1) | (self.df[col2] >= threshold2)
        return self.df[mask]
    
    def subset_specific_condition(self, col1, col2, threshold1, threshold2):
        """
        Subset the DataFrame based on a specific condition involving two columns.

        Parameters:
        col1 (str): Column name for the first criterion.
        col2 (str): Column name for the second criterion.
        threshold1 (float): Threshold for the first criterion.
        threshold2 (float): Threshold for the second criterion.

        Returns:
        DataFrame: Subset of the original DataFrame.
        """
        mask = ((self.df[col1] >= threshold1) & (self.df[col2] >= threshold2)) | \
               ((self.df[col2] >= threshold1) & (self.df[col1] >= threshold2))
        return self.df[mask]
    
    def extract_events(self, output_dir):
        """
        Extract events from the DataFrame and save them as separate CSV files.

        Parameters:
        output_dir (str): Directory to save the extracted event CSV files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        event_columns = [col for col in self.df.columns if col.endswith('_dnn') or col.endswith('_xgb')]
        event_names = set(col.split('_')[0] for col in event_columns)
        
        for event_name in event_names:
            dnn_col = f"{event_name}_dnn"
            xgb_col = f"{event_name}_xgb"
            
            if dnn_col in self.df.columns and xgb_col in self.df.columns:
                subset = self.subset_specific_condition(dnn_col, xgb_col, 0.8, 0.7)
                
                if len(subset) > 50:
                    cols_to_keep = self.df.columns[:11].tolist()
                    event_specific_cols = [col for col in self.df.columns if col.startswith(event_name)]
                    cols_to_keep.extend(event_specific_cols)
                    
                    subset = subset[cols_to_keep]
                    
                    output_path = os.path.join(output_dir, f"{event_name}_events.csv")
                    subset.to_csv(output_path, index=False)

if __name__ == "__main__":
    csv_file = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Model_Testing_Data\field_776.csv'
    output_directory = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Filtered_Set\Event_Data'

    analyzer = DataAnalyzer(csv_file)
    analyzer.extract_events(output_directory)
