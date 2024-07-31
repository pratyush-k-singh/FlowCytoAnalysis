import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class LightCurveDataset(Dataset):
    """Dataset class for handling light curve data."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_and_preprocess_data_for_ids(events_folder, ids, max_len=None):
    """
    Load and preprocess light curve data for specific IDs from event type subfolders.

    Parameters:
        events_folder (str): The path to the directory containing subfolders with light curve data files.
        ids (list of str): List of IDs to filter the light curve data.
        max_len (int, optional): The maximum length to pad the sequences. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - X (Tensor): Tensor of preprocessed and padded sequences.
            - unique_ids (ndarray): Array of unique IDs.
    """
    print(f"Loading data from folder: {events_folder}")
    df_list = []
    ids_set = set(ids)

    # Iterate through each event type subfolder
    for event_type in os.listdir(events_folder):
        event_folder = os.path.join(events_folder, event_type)
        if os.path.isdir(event_folder):
            print(f"Processing event type folder: {event_folder}")
            for file_name in os.listdir(event_folder):
                if file_name.endswith('.csv'):
                    parts = file_name.split('_')
                    if len(parts) == 3 and parts[0] == 'lc':
                        color = parts[2].replace('.csv', '')
                        if color in ['r', 'g', 'i']:
                            file_id = parts[1].replace('.0', '')
                            if file_id in ids_set:
                                file_path = os.path.join(event_folder, file_name)
                                df = pd.read_csv(file_path)
                                df['id'] = file_id
                                df['color'] = color
                                df_list.append(df)

    if not df_list:
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

    # Adding color as dummy variables
    color_dummies = pd.get_dummies(scaled_df['color'], prefix='color')
    scaled_df = pd.concat([scaled_df.drop(columns=['color']), color_dummies], axis=1)

    grouped = scaled_df.groupby('id')
    X = []
    for id, group in grouped:
        sequence = group.drop(columns=['id']).values
        if sequence.size > 0:
            sequence = sequence.astype(float)  # Ensure all values are numeric
            X.append(torch.tensor(sequence, dtype=torch.float32))

    # Pad sequences
    X = pad_sequence(X, batch_first=True, padding_value=0.0)
    if max_len is not None:
        X = X[:, :max_len, :]

    return X, df['id'].unique()

def load_ids_from_split_folder(split_data):
    """
    Load training and testing IDs from each event type subfolder within the split folder.
    Also return the event types for each ID.

    Parameters:
        split_data (str): The path to the directory containing the split data files for each event.

    Returns:
        tuple: A tuple containing:
            - train_ids (list of str): List of training IDs.
            - test_ids (list of str): List of testing IDs.
            - train_event_types (list of str): List of event types for training IDs.
            - test_event_types (list of str): List of event types for testing IDs.
    """
    print(f"Loading IDs from split folder: {split_data}")
    all_train_ids = []
    all_test_ids = []
    all_train_event_types = []
    all_test_event_types = []

    # Iterate through each event type subfolder
    for event_type in os.listdir(split_data):
        event_folder = os.path.join(split_data, event_type)
        if os.path.isdir(event_folder):
            train_file = os.path.join(event_folder, 'training.csv')
            test_file = os.path.join(event_folder, 'testing.csv')

            if os.path.exists(train_file) and os.path.exists(test_file):
                train_ids = pd.read_csv(train_file)['id'].astype(str).values
                test_ids = pd.read_csv(test_file)['id'].astype(str).values
                all_train_ids.extend(train_ids)
                all_test_ids.extend(test_ids)
                all_train_event_types.extend([event_type] * len(train_ids))
                all_test_event_types.extend([event_type] * len(test_ids))
            else:
                print(f"Training or testing files not found in folder: {event_folder}")

    return all_train_ids, all_test_ids, all_train_event_types, all_test_event_types

class TotemModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(TotemModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * (input_shape[0] // 8), 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to (batch_size, channels, sequence_length)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def build_totem_model(input_shape, num_classes):
    """
    Build and compile a Totem forecasting model.

    Parameters:
        input_shape (tuple): The shape of the input data (sequence length, number of features).
        num_classes (int): The number of output classes.

    Returns:
        TotemModel: The compiled Totem forecasting model.
    """
    model = TotemModel(input_shape, num_classes)
    return model

def train_one_epoch(dataloader, model, criterion, optimizer, device):
    """
    Train the Totem model for one epoch.

    Parameters:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (Loss): Loss function to use.
        optimizer (Optimizer): Optimizer to use for training.
        device (torch.device): Device to train the model on.

    Returns:
        float: The average training loss.
    """
    model.train()
    total_loss = 0
    for data, labels in tqdm(dataloader, desc="Training"):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_one_epoch(dataloader, model, criterion, device):
    """
    Evaluate the model for one epoch.

    Parameters:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        criterion (Loss): Loss function to use.
        device (torch.device): Device to evaluate the model on.

    Returns:
        tuple: A tuple containing:
            - float: The average evaluation loss.
            - float: The evaluation accuracy.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy

def main():
    # Set file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    events_folder = os.path.join(script_dir, '..', '..', 'Filtered_Set', 'Light_Curves')
    split_data = os.path.join(script_dir, '..', '..', 'Filtered_Set', 'Model_Split')
    model_save_path = os.path.join(script_dir, 'Totem_model.pth')  # Saving as a PyTorch model instead of .h5

    # Load training and testing IDs along with their event types
    train_ids, test_ids, train_event_types, test_event_types = load_ids_from_split_folder(split_data)

    # Load and preprocess data
    max_len = 300
    X_train, train_unique_ids = load_and_preprocess_data_for_ids(events_folder, train_ids, max_len)
    X_test, test_unique_ids = load_and_preprocess_data_for_ids(events_folder, test_ids, max_len)

    # Check if data was loaded
    if X_train is None or X_test is None:
        print("No data was loaded. Please check the paths and ensure that data is available.")
        return

    # Map event types to numeric labels
    event_type_to_label = {event_type: idx for idx, event_type in enumerate(set(train_event_types + test_event_types))}
    train_labels = np.array([event_type_to_label[event_type] for event_type in train_event_types])
    test_labels = np.array([event_type_to_label[event_type] for event_type in test_event_types])

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create datasets and dataloaders
    batch_size = 32
    train_dataset = LightCurveDataset(X_train, train_labels)
    test_dataset = LightCurveDataset(X_test, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    input_shape = X_train.shape[1:]  # (sequence length, number of features)
    num_classes = len(event_type_to_label)  # Number of unique event types as the number of classes
    learning_rate = 0.001
    num_epochs = 50
    patience = 10  # Early stopping patience

    # Model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_totem_model(input_shape, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate_one_epoch(test_loader, model, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    final_val_loss, final_val_accuracy = evaluate_one_epoch(test_loader, model, criterion, device)
    print(f"Final Val Loss: {final_val_loss:.4f}, Final Val Accuracy: {final_val_accuracy:.4f}")

if __name__ == "__main__":
    main()
