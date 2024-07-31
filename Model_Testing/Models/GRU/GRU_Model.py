"""
GRU Model Trainer for Light Curve Data

This script loads and preprocesses light curve data for various event types, trains a GRU model on the data, 
and saves the trained model to a specified path. It also handles padding of sequences and standardization 
of features.

Functions:
    load_and_preprocess_data_for_ids(event_path, ids, max_len=None): Loads and preprocesses light curve data for specific IDs.
    load_ids_from_split_folder(split_folder): Loads training and testing IDs from the split folder.
    build_model(input_shape, num_classes): Builds and compiles a GRU model.
    main(): Main function to execute the data loading, model training, and saving.
"""

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Relative paths
events_folder = os.path.join(script_dir, '..', '..', 'Filtered_Set', 'Light_Curves')
split_data = os.path.join(script_dir, '..', '..', 'Filtered_Set', 'Model_Split')
model_save_path = os.path.join(script_dir, 'GRU_model.h5')

def load_and_preprocess_data_for_ids(event_path, ids, max_len=None):
    """
    Load and preprocess light curve data for specific IDs from a given directory.

    Parameters:
        event_path (str): The path to the directory containing the light curve data files.
        ids (list of str): List of IDs to filter the light curve data.
        max_len (int, optional): The maximum length to pad the sequences. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - X (ndarray): Array of preprocessed and padded sequences.
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
    columns_to_scale = ['mag', 'magerr', 'ra', 'dec', 'chi', 'sharp', 'filefracday', 'limitmag', 
                        'magzp', 'magzprms', 'clrcoeff', 'clrcounc', 'exptime', 'airmass']
    
    missing_cols = set(columns_to_scale) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    scaled_features = scaler.fit_transform(df[columns_to_scale])
    
    scaled_df = pd.DataFrame(scaled_features, columns=columns_to_scale)
    scaled_df['id'] = df['id'].values
    scaled_df['color'] = df['color'].values
    
    color_dummies = pd.get_dummies(scaled_df['color'], prefix='color')
    scaled_df = pd.concat([scaled_df.drop(columns=['color']), color_dummies], axis=1)
    
    grouped = scaled_df.groupby('id')
    X = []
    for id, group in grouped:
        sequence = group.drop(columns=['id']).values
        if sequence.size > 0:
            X.append(sequence)
    
    if max_len is not None:
        X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')
    else:
        X = pad_sequences(X, padding='post', dtype='float32')
    
    return X, df['id'].unique()

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

def build_model(input_shape, num_classes):
    """
    Build and compile a GRU model.

    Parameters:
        input_shape (tuple): The shape of the input data (sequence length, number of features).
        num_classes (int): The number of output classes.

    Returns:
        Sequential: The compiled GRU model.
    """
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(GRU(50, return_sequences=True))
    model.add(GRU(50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
            max_len = max(max_len, X.shape[1])
    
    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []
    
    # Second pass to process data and apply max_len
    for event_folder in event_types:
        print(f"Processing event folder: {event_folder}")
        split_folder = os.path.join(split_data, event_folder)
        train_ids, test_ids = load_ids_from_split_folder(split_folder)
        event_path = os.path.join(events_folder, event_folder)
        
        X_train, _ = load_and_preprocess_data_for_ids(event_path, train_ids, max_len)
        X_test, _ = load_and_preprocess_data_for_ids(event_path, test_ids, max_len)
        
        if X_train is not None and X_test is not None:
            y_train = np.array([event_folder] * len(X_train))
            y_test = np.array([event_folder] * len(X_test))
            
            X_train_all.append(X_train)
            y_train_all.append(y_train)
            X_test_all.append(X_test)
            y_test_all.append(y_test)
            
            # Debug: Check lengths
            print(f"Event: {event_folder} - Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    if X_train_all:
        X_train = np.concatenate(X_train_all, axis=0)
        y_train = np.concatenate(y_train_all, axis=0)
        X_test = np.concatenate(X_test_all, axis=0)
        y_test = np.concatenate(y_test_all, axis=0)
    else:
        print("No data available for training and testing.")
        return
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(event_types)
    
    # Convert y to categorical
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values
    
    # Build and train the GRU model
    model = build_model(input_shape, num_classes)
    
    # Add EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    
    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
