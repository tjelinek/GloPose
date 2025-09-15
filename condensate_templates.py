from pathlib import Path


bop_base = Path('/mnt/personal/jelint19/data/bop')
dataset = 'hope'
split = 'onboarding_static'

path_to_dataset = bop_base / dataset
path_to_split = path_to_dataset / split

for sequence in path_to_dataset.iterdir():

    rgb_folder = sequence / 'rgb'
    segmentation_folder = sequence / 'mask_visib'