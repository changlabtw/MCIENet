import os
import h5py
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # input data
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)

    return parser.parse_args()

args = get_args()
file = args.file
output_folder = args.output_folder

os.makedirs(output_folder, exist_ok=True)

with h5py.File(file) as f:
    for phase in f.keys():
        path = os.path.join(
            output_folder, f'{phase}.csv' if phase != 'val' else 'dev.csv'
            )
        if os.path.exists(path):
            print('[Existing document]', path)
            continue

        data = pd.DataFrame(f[phase]['data'])

        data[0] = data[0].astype(str).str.upper()
        data[1] = data[1].astype(str).str.upper()
            
        data.rename(columns={0:'seq1', 1:'seq2'}, inplace=True)
        # data['sequence'] = data[0] + '[SEP]' + data[1]
        # data.drop([0, 1], axis=1, inplace=True)

        data['label'] = pd.DataFrame(f[phase]['labels'])

        data.to_csv(path, index=False)