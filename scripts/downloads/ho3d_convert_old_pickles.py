# Python 2.7 code
import os
import pickle
import numpy as np


def convert_pickle_to_npz(pickle_file, npz_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    np.savez(npz_file, **data)


def process_folder(base_folder):
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder, 'meta')
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.pkl'):
                    pickle_file = os.path.join(subfolder_path, file_name)
                    npz_file = os.path.join(subfolder_path, file_name.replace('.pkl', '.npz'))
                    convert_pickle_to_npz(pickle_file, npz_file)
                    print(f"Converted {pickle_file} to {npz_file}")


# Replace 'base_folder' with the path to your base folder
base_folder = '/mnt/personal/jelint19/data/HO3D/train'
process_folder(base_folder)
