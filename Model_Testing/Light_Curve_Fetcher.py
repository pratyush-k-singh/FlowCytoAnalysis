import pandas as pd
import os
import glob
import requests

def read_event_csv(file_path):
    """
    Read the event CSV file and return a DataFrame.
    """
    return pd.read_csv(file_path)

def fetch_light_curve(ra, dec, bandname, radius, output_dir, id_number):
    """
    Fetch the light curve data from IPAC using requests and return the filename.
    Radius is in arcsec.
    """
    radiusd = radius / 3600.0
    url = f'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?' \
          f'POS=CIRCLE+{ra:.5f}+{dec:.5f}+{radiusd}&BANDNAME={bandname}&' \
          f'NOBS_MIN=3&TIME=55000.0+60000.0&BAD_CATFLAGS_MASK=32768&FORMAT=CSV'
    original_filename = f'lc_{ra:.5f}_{dec:.5f}_{bandname}.csv'
    original_output_path = os.path.join(output_dir, original_filename)
    renamed_filename = f'lc_{id_number}_{bandname}.csv'
    renamed_output_path = os.path.join(output_dir, renamed_filename)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(original_output_path):
        print(f'File already exists: {original_output_path}, renaming to {renamed_output_path}.')
        os.rename(original_output_path, renamed_output_path)
        return renamed_output_path

    response = requests.get(url)
    if response.status_code == 200:
        with open(renamed_output_path, 'wb') as f:
            f.write(response.content)
        print(f'Light curve saved as {renamed_output_path}')
    else:
        print(f'Failed to fetch light curve for RA: {ra}, Dec: {dec}, Band: {bandname}')
    
    return renamed_output_path

def process_light_curves(df, ra_col, dec_col, id_col, output_dir):
    """
    Process each row of the DataFrame, generate files for 'g', 'r', 'i' bands and their folded light curves.
    """
    bandnames = ['g', 'r', 'i']  # Define bandnames to loop over
    radius = 2
    for idx, row in df.iterrows():
        for band in bandnames:
            filename = fetch_light_curve(row[ra_col], row[dec_col], band, radius, output_dir, row[id_col])

if __name__ == "__main__":
    input_dir = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Filtered_Set\Event_Data'
    output_dir = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Filtered_Set\Light_Curves'

    os.makedirs(output_dir, exist_ok=True)

    event_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    for event_file in event_files:
        event_df = read_event_csv(event_file)
        event_name = os.path.splitext(os.path.basename(event_file))[0]
        event_folder = os.path.join(output_dir, event_name)
        os.makedirs(event_folder, exist_ok=True)
        process_light_curves(event_df, 'ra', 'dec', event_df.columns[0], event_folder)
