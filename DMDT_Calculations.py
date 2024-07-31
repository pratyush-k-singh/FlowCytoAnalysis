import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed

def create_dts(interval, max_value):
    return [i for i in range(0, max_value + interval, interval)]

dts_interval = 250
dts_max_value = 10000
dts = create_dts(dts_interval, dts_max_value)

def diffs_for(x):
    x = x[x != 0]
    if len(x) < 2:
        return np.array([])
    pairwisediff = pairwise_distances(x.reshape(-1, 1), metric='l1')
    indices = np.triu_indices(len(pairwisediff), 1)
    diffed = pairwisediff[indices]
    return diffed

def create_dms(dm):
    dm = dm[dm != 0]
    if len(dm) == 0:
        return np.array([])
    median_dm = np.median(dm)
    std_dm = np.std(dm)

    bins = np.linspace(median_dm - 5 * std_dm, median_dm + 5 * std_dm, 41)
    return bins

def dmdt(t, m, dms):
    t = t[t != 0]
    m = m[m != 0]

    if len(t) < 2 or len(m) < 2:
        return None, None, None

    dt = diffs_for(t)
    dm = diffs_for(m)

    if len(dt) == 0 or len(dm) == 0:
        return None, None, None

    H, xedges, yedges = np.histogram2d(dt, dm, bins=[dts, dms])
    dmdt = H / H.sum() if H.sum() != 0 else H

    return dmdt, xedges, yedges

def process_column(time, data, chunk_size=20):
    num_chunks = (len(data) + chunk_size - 1) // chunk_size

    mean_data = []
    mean_time = []
    for i in range(num_chunks):
        chunk = data[i * chunk_size: (i + 1) * chunk_size]
        time_chunk = time[i * chunk_size: (i + 1) * chunk_size]
        
        chunk = chunk[chunk != 0]
        time_chunk = time_chunk[time_chunk != 0]

        if len(chunk) > 0 and len(time_chunk) > 0:
            mean_data.append(chunk.mean())
            mean_time.append(time_chunk.mean())

    return np.array(mean_time), np.array(mean_data)

def plot_dmdt(mean_time, mean_data, title):
    dms = create_dms(mean_data)
    if len(dms) == 0:
        return None

    result, xedges, yedges = dmdt(mean_time, mean_data, dms)

    if result is None:
        return None

    fig, ax = plt.subplots()
    cax = ax.imshow(result, interpolation='nearest', origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto', cmap='viridis')

    fig.colorbar(cax, ax=ax)
    ax.set_xlabel('dt')
    ax.set_ylabel('dm')
    ax.set_title(title)

    return fig

def save_to_pdf(output_path, figures):
    with PdfPages(output_path) as pdf:
        for fig in figures:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)

def process_file(file_path, output_dir):
    df = pd.read_csv(file_path)
    time = df['Time']

    figures = []
    for column in df.columns:
        if column == 'Time':
            continue
        data = df[column]
        mean_time, mean_data = process_column(time, data)
        fig = plot_dmdt(mean_time, mean_data, f'DMDT for {column}')
        figures.append(fig)

    if figures:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f'{base_filename}.pdf'
        output_path = os.path.join(output_dir, output_filename)
        save_to_pdf(output_path, figures)
    else:
        print(f"No figures were generated for {file_path}.")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    Parallel(n_jobs=-1)(delayed(process_file)(os.path.join(input_dir, f), output_dir) for f in filenames)

if __name__ == "__main__":
    input_directory = r'C:\Users\praty\cytoflow\Codeset\NormalizedCytometryFiles'
    output_directory = r'C:\Users\praty\cytoflow\Codeset\OptimizedDMDT'
    process_directory(input_directory, output_directory)
