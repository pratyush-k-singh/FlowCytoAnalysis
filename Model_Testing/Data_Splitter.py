import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
events_folder = os.path.join(script_dir, 'Filtered_Set', 'Event_Data')
output_folder = os.path.join(script_dir, 'Filtered_Set', 'Model_Split')

# Function to load IDs from CSV files for each event
def load_event_ids(events_folder):
    event_ids = {}
    for file_name in os.listdir(events_folder):
        if file_name.endswith('.csv') and file_name != 'duplicate_events.csv':
            file_path = os.path.join(events_folder, file_name)
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                if len(df) > 1:  # Ensure the CSV has data
                    event_name = file_name.split('_')[0]
                    event_ids[event_name] = df['_id'].tolist()
    return event_ids

# Function to create training and testing splits for each event and save them to respective folders
def split_and_save_event_ids(event_ids, output_folder, test_size=0.2, random_state=42):
    for event, ids in event_ids.items():
        print(f"Splitting IDs for event: {event}")
        train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
        
        event_folder = os.path.join(output_folder, event)
        os.makedirs(event_folder, exist_ok=True)
        
        train_file = os.path.join(event_folder, 'training.csv')
        test_file = os.path.join(event_folder, 'testing.csv')
        
        pd.DataFrame(train_ids, columns=['id']).to_csv(train_file, index=False)
        pd.DataFrame(test_ids, columns=['id']).to_csv(test_file, index=False)

# Load event IDs from CSV files
event_ids = load_event_ids(events_folder)

# Split the event IDs into training and testing sets and save to respective folders
split_and_save_event_ids(event_ids, output_folder)

print("Splitting and saving complete.")
