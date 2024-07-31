import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence

# Define the Totem model (use the same definition as in training)
class TotemModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TotemModel, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
field_path = os.path.join(script_dir, '..', '..', 'Model_Training_Fields', 'field_776.csv')
model_save_path = os.path.join(script_dir, 'Totem_model.pth')
predictions_save_path = os.path.join(script_dir, 'Predictions', 'final_predictions.csv')

def load_and_preprocess_data_for_ids(event_path, ids, max_len=None):
    """
    Load and preprocess light curve data for specific IDs from a given directory.
    """
    print(f"Loading data from folder: {event_path}")
    df_list = []
    ids_set = set(ids)
    for file_name in os.listdir(event_path):
        if file_name.endswith('.csv'):
            parts = file_name.split('_')
            if len(parts) == 3 and parts[0] == 'lc':
                color = parts[2].replace('.csv', '')
                if color in ['r', 'g', 'i']:
                    file_id = parts[1].replace('.0', '')
                    if file_id in ids_set:
                        file_path = os.path.join(event_path, file_name)
                        df = pd.read_csv(file_path)
                        df['id'] = file_id
                        df['color'] = color
                        df_list.append(df)
    
    if not df_list:
        print(f"No data available for IDs in folder: {event_path}")
        return None, None
    
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by=['id', 'mjd'])
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.nunique() > 1]

    scaler = StandardScaler()
    columns_to_scale = ['mag', 'magerr', 'ra', 'dec', 'chi', 'sharp', 'filefracday', 'limitmag', 
                        'magzp', 'magzprms', 'clrcoeff', 'clrcounc', 'exptime', 'airmass']
    
    missing_cols = set(columns_to_scale) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # Ensure that the columns to scale are numeric
    df[columns_to_scale] = df[columns_to_scale].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=columns_to_scale)  # Drop rows where scaling columns are NaN

    scaled_features = scaler.fit_transform(df[columns_to_scale])
    
    scaled_df = pd.DataFrame(scaled_features, columns=columns_to_scale)
    scaled_df['id'] = df['id'].values
    scaled_df['color'] = df['color'].values
    
    color_dummies = pd.get_dummies(scaled_df['color'], prefix='color')
    scaled_df = pd.concat([scaled_df.drop(columns=['color']), color_dummies], axis=1)
    
    grouped = scaled_df.groupby('id')
    sequences = []
    for id, group in grouped:
        sequence = group.drop(columns=['id']).values
        if sequence.size > 0:
            sequences.append(torch.tensor(sequence, dtype=torch.float32))
    
    if not sequences:
        print(f"No valid sequences found for IDs in folder: {event_path}")
        return None, None
    
    sequences = pad_sequence(sequences, batch_first=True, padding_value=float('nan'))

    if max_len is not None:
        if sequences.size(1) < max_len:
            padded_sequences = torch.full((sequences.size(0), max_len, sequences.size(2)), float('nan'))
            padded_sequences[:, :sequences.size(1), :] = sequences
            sequences = padded_sequences
        elif sequences.size(1) > max_len:
            sequences = sequences[:, :max_len, :]
    
    return sequences.numpy(), df['id'].unique()

def load_ids_from_split_folder(split_folder):
    """
    Load IDs from the split folder.
    """
    print(f"Loading IDs from split folder: {split_folder}")
    test_file = os.path.join(split_folder, 'testing.csv')
    
    if os.path.exists(test_file):
        test_ids = pd.read_csv(test_file)['id'].astype(str).values
    else:
        print(f"Testing file not found in folder: {split_folder}")
        return []
    
    return test_ids

def main():
    """
    Main function to load data, preprocess, predict using the Totem model, and merge with field data.
    """
    events_folder = os.path.join(script_dir, '..', '..', 'Filtered_Set', 'Light_Curves')
    split_data = os.path.join(script_dir, '..', '..', 'Filtered_Set', 'Model_Split')
    event_types = [d for d in os.listdir(split_data) if os.path.isdir(os.path.join(split_data, d))]
    max_len = 0
    
    # First pass to compute max_len
    for event_folder in event_types:
        print(f"Processing event folder for max_len: {event_folder}")  # Debug: Print current event folder
        split_folder = os.path.join(split_data, event_folder)
        test_ids = load_ids_from_split_folder(split_folder)
        event_path = os.path.join(events_folder, event_folder)
        X, _ = load_and_preprocess_data_for_ids(event_path, test_ids)
        if X is not None:
            max_len = max(max_len, X.shape[1])

    X_test_all, test_ids_all = [], []
    
    # Second pass to process data and apply max_len
    for event_folder in event_types:
        print(f"Processing event folder: {event_folder}")  # Debug: Print current event folder
        split_folder = os.path.join(split_data, event_folder)
        test_ids = load_ids_from_split_folder(split_folder)
        event_path = os.path.join(events_folder, event_folder)
        
        X_test, ids = load_and_preprocess_data_for_ids(event_path, test_ids, max_len)
        
        if X_test is not None:
            X_test_all.append(X_test)
            test_ids_all.extend(ids)
            
            # Debug: Check lengths
            print(f"Event: {event_folder} - Testing samples: {len(X_test)}")

    if X_test_all:
        X_test = np.concatenate(X_test_all, axis=0)
        test_ids = np.array(test_ids_all)
    else:
        print("No data available for testing.")
        return
    
    # Load the trained Totem model
    model = TotemModel(input_dim=X_test.shape[2], hidden_dim=128, output_dim=len(event_types))
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Create DataLoader for the test data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    y_pred_all = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0]
            outputs = model(inputs)
            y_pred_all.append(outputs.cpu().numpy())
    
    y_pred = np.concatenate(y_pred_all, axis=0)
    
    # Prepare predictions DataFrame
    predictions = pd.DataFrame(y_pred, columns=[f'totem_{event}' for event in event_types])
    
    # Ensure correct IDs length
    print(f"Number of test IDs: {len(test_ids)}")
    if len(test_ids) != len(y_pred):
        print(f"Length mismatch: test_ids ({len(test_ids)}) vs. predictions ({len(y_pred)})")
        return

    # Assign test_ids directly if lengths match
    predictions['id'] = test_ids
    predictions.to_csv(predictions_save_path, index=False)
    print(f"Predictions saved to: {predictions_save_path}")

    # Load field CSV
    field_df = pd.read_csv(field_path)
    
    # Rename _id to id for merging
    field_df.rename(columns={'_id': 'id'}, inplace=True)
    
    # Ensure 'id' columns have the same data type
    predictions['id'] = predictions['id'].astype(str)
    field_df['id'] = field_df['id'].astype(str)
    
    # Merge with field data
    merged_df = pd.merge(predictions, field_df, on='id')
    
    # Prepare the final DataFrame with Totem predictions and placeholders for other models
    final_columns = ['id'] + [f'totem_{event}' for event in event_types] + \
                    [f'{event}_xgb' for event in event_types] + \
                    [f'{event}_dnn' for event in event_types]
    final_df = merged_df[final_columns]
    
    # Save final merged predictions
    final_predictions_path = os.path.join(script_dir, 'final_merged_predictions.csv')
    final_df.to_csv(final_predictions_path, index=False)
    print(f"Final merged predictions saved to: {final_predictions_path}")

if __name__ == "__main__":
    main()
