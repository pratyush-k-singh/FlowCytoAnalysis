import pandas as pd

def normalize_confidences(df: pd.DataFrame, column_groups: list) -> pd.DataFrame:
    """
    Normalizes the confidence values in specified column groups so that they add up to 1.

    :param df: The DataFrame containing the confidence values.
    :param column_groups: A list of lists, where each sublist contains column names to normalize together.
    :return: A DataFrame with normalized confidence values.
    """
    for columns in column_groups:
        # Calculate the sum of the columns row-wise
        sums = df[columns].sum(axis=1)

        # Check which rows do not sum to 1
        needs_normalization = sums != 1

        # Normalize those rows
        df.loc[needs_normalization, columns] = df.loc[needs_normalization, columns].div(
            sums[needs_normalization], axis=0
        )

    return df

def process_file(file_path: str, output_path: str, model_prefix: str, suffixes: list):
    """
    Processes a CSV file to normalize the confidence values in specified groups of columns.

    :param file_path: Path to the input CSV file.
    :param output_path: Path to save the output CSV file.
    :param model_prefix: The prefix that identifies the first group of columns to normalize.
    :param suffixes: A list of suffixes that identify the second and third groups of columns to normalize.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Identify columns for the prefix group
    prefix_columns = [col for col in df.columns if col.startswith(model_prefix)]

    # Identify columns for the suffix groups
    suffix_groups = [
        [col for col in df.columns if col.endswith(suffix)]
        for suffix in suffixes
    ]

    # Combine prefix and suffix column groups
    all_groups = [prefix_columns] + suffix_groups

    # Normalize the confidences
    normalized_df = normalize_confidences(df, all_groups)

    # Save the normalized DataFrame to a new CSV file
    normalized_df.to_csv(output_path, index=False)
    print(f"Normalized file saved to {output_path}")

# Example usage
file_paths = [
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\GRU\Predictions\final_merged_predictions.csv",
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\TCN\Predictions\final_merged_predictions.csv",
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\LSTM\Predictions\final_merged_predictions.csv"
]

output_paths = [
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\GRU\Predictions\normalized_predictions.csv",
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\TCN\Predictions\normalized_predictions.csv",
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\LSTM\Predictions\normalized_predictions.csv"
]

model_prefixes = ['gru_', 'tcn_', 'lstm_']
suffixes = ['_xgb', '_dnn']

for i, file_path in enumerate(file_paths):
    process_file(file_path, output_paths[i], model_prefixes[i], suffixes)
