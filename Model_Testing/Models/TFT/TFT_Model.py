import os
import warnings
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from torch.optim import Adam

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Relative paths
events_folder = os.path.join(script_dir, '..', '..', 'Filtered_Set', 'Light_Curves')
split_data = os.path.join(script_dir, '..', '..', 'Filtered_Set', 'Model_Split')
model_save_path = os.path.join(script_dir, 'TFT_model.pth')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define columns to scale globally so they are accessible in multiple functions
columns_to_scale = ['mag', 'magerr', 'ra', 'dec', 'chi', 'sharp', 'filefracday', 'limitmag',
                    'magzp', 'magzprms', 'clrcoeff', 'clrcounc', 'exptime', 'airmass']

def load_and_preprocess_data_for_ids(event_path, ids, max_len=None):
    """
    Load and preprocess light curve data for specific IDs from a given directory.

    Parameters:
        event_path (str): The path to the directory containing the light curve data files.
        ids (list of str): List of IDs to filter the light curve data.
        max_len (int, optional): The maximum length to pad the sequences. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - df (DataFrame): Preprocessed DataFrame suitable for TFT.
            - unique_ids (ndarray): Array of unique IDs.
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
    
    missing_cols = set(columns_to_scale) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    scaled_features = scaler.fit_transform(df[columns_to_scale])

    scaled_df = pd.DataFrame(scaled_features, columns=columns_to_scale)
    scaled_df['id'] = df['id'].values
    scaled_df['color'] = df['color'].values
    scaled_df['mjd'] = df['mjd'].values  # Time index

    # Convert mjd to integer
    scaled_df['mjd'] = scaled_df['mjd'].astype(int)

    color_encoder = LabelEncoder()
    scaled_df['color_encoded'] = color_encoder.fit_transform(scaled_df['color']).astype(str)
    
    grouped = scaled_df.groupby('id')
    
    return scaled_df, df['id'].unique()

def load_ids_from_split_folder(split_folder):
    """
    Load training and testing IDs from the split folder.

    Parameters:
        split_folder (str): The path to the directory containing the split data files.

    Returns:
        tuple: A tuple containing:
            - train_ids (list of str): List of training IDs.
            - test_ids (list of str): List of testing IDs.
    """
    print(f"Loading IDs from split folder: {split_folder}")
    train_file = os.path.join(split_folder, 'training.csv')
    test_file = os.path.join(split_folder, 'testing.csv')
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        train_ids = pd.read_csv(train_file)['id'].astype(str).values
        test_ids = pd.read_csv(test_file)['id'].astype(str).values
    else:
        print(f"Training or testing files not found in folder: {split_folder}")
        return [], []
    
    return train_ids, test_ids

def main():
    """
    Main function to execute the data loading, model training, and saving.
    """
    event_types = [d for d in os.listdir(split_data) if os.path.isdir(os.path.join(split_data, d))]
    max_len = 0
    
    # First pass to compute max_len
    for event_folder in event_types:
        print(f"Processing event folder for max_len: {event_folder}")
        split_folder = os.path.join(split_data, event_folder)
        train_ids, _ = load_ids_from_split_folder(split_folder)
        event_path = os.path.join(events_folder, event_folder)
        X, _ = load_and_preprocess_data_for_ids(event_path, train_ids)
        if X is not None:
            max_len = max(max_len, X.shape[0])
    
    data_all = []
    
    # Second pass to process data and apply max_len
    for event_folder in event_types:
        print(f"Processing event folder: {event_folder}")
        split_folder = os.path.join(split_data, event_folder)
        train_ids, test_ids = load_ids_from_split_folder(split_folder)
        event_path = os.path.join(events_folder, event_folder)
        
        # Load and preprocess data
        df_train, _ = load_and_preprocess_data_for_ids(event_path, train_ids, max_len)
        df_test, _ = load_and_preprocess_data_for_ids(event_path, test_ids, max_len)
        
        # Add event type as label
        df_train['event_type'] = event_folder
        df_test['event_type'] = event_folder
        
        if df_train is not None and df_test is not None:
            data_all.append((df_train, 'train'))
            data_all.append((df_test, 'test'))
    
    if not data_all:
        print("No data available for training and testing.")
        return
    
    # Concatenate all data for each split
    train_data = pd.concat([df for df, split in data_all if split == 'train'], ignore_index=True)
    test_data = pd.concat([df for df, split in data_all if split == 'test'], ignore_index=True)

    # Prepare TFT TimeSeriesDataSet
    training = TimeSeriesDataSet(
        train_data,
        time_idx='mjd',
        target='event_type',
        group_ids=['id'],
        min_encoder_length=1,  # Minimal historical length
        max_encoder_length=max_len,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=['id'],
        time_varying_known_categoricals=['color_encoded'],
        time_varying_unknown_reals=columns_to_scale,
        target_normalizer=GroupNormalizer(groups=['id']),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True
    )
    
    # Create DataLoaders
    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=4)
    test_dataloader = training.to_dataloader(train=False, batch_size=64, num_workers=4)

    # Define and train the TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=len(event_types),
        loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=4
    ).to(device)

    # Use PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        gpus=1 if torch.cuda.is_available() else 0,
        gradient_clip_val=0.1
    )
    
    trainer.fit(tft, train_dataloader)
    
    # Save the model
    torch.save(tft.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
