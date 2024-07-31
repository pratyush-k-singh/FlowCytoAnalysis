import pandas as pd
import os

# Define the event names to process
event_names = ['ew', 'fla', 'i', 'lpv', 'puls']

# Define input file names (update these as necessary)
input_files = [
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\GRU\Predictions\normalized_predictions.csv",
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\TCN\Predictions\normalized_predictions.csv",
    r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\LSTM\Predictions\normalized_predictions.csv"
]

# Define summary output directory
summary_output_dir = r"C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\Model_Performance"

def process_file(input_file, event_names):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Initialize an empty DataFrame for results
    result_df = pd.DataFrame()
    
    for event in event_names:
        # Find the column that ends with _eventname
        event_col = [col for col in df.columns if col.endswith(f'_{event}')]
        
        if len(event_col) != 1:
            raise ValueError(f"Expected exactly one column ending with '_{event}', but found {len(event_col)}")
        
        event_col = event_col[0]
        
        # Find columns that start with event_
        related_cols = [col for col in df.columns if col.startswith(f'{event}_')]
        
        # Ensure we have at least two columns to compare against
        if len(related_cols) < 2:
            raise ValueError(f"Expected at least two columns starting with '{event}_' for event '{event}', but found {len(related_cols)}")

        # Calculate the absolute differences
        diff_cols = {}
        for col in related_cols:
            diff_col_name = f'diff_{event}_{col[len(event)+1:]}'
            diff_cols[diff_col_name] = (df[event_col] - df[col]).abs()
        
        # Calculate the average of the absolute differences
        avg_diff = pd.DataFrame(diff_cols).mean(axis=1)
        diff_cols[f'avg_diff_{event}'] = avg_diff
        
        # Create a DataFrame with the required columns
        trio_df = pd.DataFrame({
            'id': df['id'],
            **diff_cols
        })
        
        # Append the new DataFrame to the result DataFrame
        result_df = pd.concat([result_df, trio_df], axis=1)
    
    # Remove duplicate 'id' columns if any (in case 'id' column was duplicated during concat)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    
    # Define the output file name based on the input file
    output_file = os.path.join(os.path.dirname(input_file), f'compared_{os.path.basename(input_file)}')
    
    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)
    print(f'Saved results to {output_file}')
    
    # Correct extraction of model name
    model_name = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
    summary_file = os.path.join(summary_output_dir, f"{model_name}_Predictions_Results.csv")
    
    # Ensure the summary output directory exists
    os.makedirs(summary_output_dir, exist_ok=True)
    
    # Calculate means for each column except 'id'
    summary_df = result_df.drop(columns=['id']).mean().reset_index()
    summary_df.columns = ['Column', 'Mean']
    
    # Save the summary DataFrame to a new CSV file
    summary_df.to_csv(summary_file, index=False)
    print(f'Saved summary results to {summary_file}')

# Process each input file
for input_file in input_files:
    process_file(input_file, event_names)
