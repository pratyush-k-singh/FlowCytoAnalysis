# -*- coding: utf-8 -*-
"""Column_Sorter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LUdbLK91SkAMzl6gKLz9h_-KEgiwZ-iw
"""

# Install the gspread library
!pip install --upgrade gspread

# Authenticate and create the client
from google.colab import auth
auth.authenticate_user()
import gspread
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

# Open the Google Sheet using its name
sheet_name = 'Metadata'  # Replace with your actual sheet name
worksheet = gc.open(sheet_name).sheet1

# Get all the data from the sheet
data = worksheet.get_all_values()
headers = data.pop(0)  # Remove the header

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(data, columns=headers)

# Choose the column to cluster by
cluster_column = 'MaterialCode'  # Replace with your actual column name

# Sort the DataFrame based on the cluster column
clustered_df = df.sort_values(by=[cluster_column])

# Create a new sheet and write the clustered data
new_sheet_name = 'ClusteredData'  # Replace with your desired new sheet name
gc.create(new_sheet_name)
new_worksheet = gc.open(new_sheet_name).sheet1
new_worksheet.update([headers] + clustered_df.values.tolist())