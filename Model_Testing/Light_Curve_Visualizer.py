import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_combined_light_curves(ra, dec, bandnames, output_dir, plot_output_dir, yso_dnn=None, yso_xgb=None):
    """
    Plot combined light curves for specified bandnames in a single plot.
    """
    plt.figure(figsize=(10, 6), dpi=120)

    colors = {'g': 'C0', 'r': 'C2', 'i': 'C4'}
    
    for bandname in bandnames:
        filename = os.path.join(output_dir, f'lc_{ra:.5f}_{dec:.5f}_{bandname}.csv')
        data = pd.read_csv(filename)
        plt.errorbar(data['mjd'], data['mag'], yerr=data['magerr'], fmt='o', markersize=3, 
                     label=f'{bandname} band', color=colors[bandname], ecolor='gray', capsize=0)

    plt.gca().invert_yaxis()
    plt.xlabel('Modified Julian Date (MJD)')
    plt.ylabel('Magnitude')
    plt.title(f'RA: {ra}, DEC: {dec} (YSO_DNN: {yso_dnn}, YSO_XGB: {yso_xgb})')
    plt.legend()

    os.makedirs(plot_output_dir, exist_ok=True)
    plot_filename = os.path.join(plot_output_dir, f'plot_{ra:.5f}_{dec:.5f}.png')
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

def process_light_curve_files(event_folder, output_dir, base_plot_output_dir):
    """
    Process each CSV file in the event folder and generate a combined plot for 'g', 'r', 'i' bands.
    """
    bandnames = ['g', 'r', 'i']

    light_curve_files = glob.glob(os.path.join(event_folder, 'lc_*.csv'))
    ra_dec_set = set()
    
    for lc_file in light_curve_files:
        filename = os.path.basename(lc_file)
        parts = filename.split('_')
        ra = float(parts[1])
        dec = float(parts[2])
        ra_dec_set.add((ra, dec))
    
    event_name = os.path.basename(event_folder)
    plot_output_dir = os.path.join(base_plot_output_dir, event_name)
    os.makedirs(plot_output_dir, exist_ok=True)
    
    for ra, dec in ra_dec_set:
        yso_dnn, yso_xgb = None, None
        event_csv = os.path.join(event_folder, 'event.csv')
        if os.path.exists(event_csv):
            event_df = pd.read_csv(event_csv)
            matched_row = event_df[(event_df['ra'] == ra) & (event_df['dec'] == dec)]
            if not matched_row.empty:
                yso_dnn = matched_row.iloc[0]['yso_dnn']
                yso_xgb = matched_row.iloc[0]['yso_xgb']
        
        plot_filename = plot_combined_light_curves(ra, dec, bandnames, event_folder, plot_output_dir, yso_dnn, yso_xgb)
        print(f'Plot saved as {plot_filename}')

if __name__ == "__main__":
    input_dir = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Filtered_Set\Light_Curves'
    base_plot_output_dir = r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Filtered_Set\Visualized_Light_Curves'

    event_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for event_folder in event_folders:
        full_event_folder_path = os.path.join(input_dir, event_folder)
        process_light_curve_files(full_event_folder_path, full_event_folder_path, base_plot_output_dir)
