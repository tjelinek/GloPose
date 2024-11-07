#%%
import subprocess

import cv2
import torch
import pycolmap
import os
import h5py
import kornia as K
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import numpy as np


def load_torch_image(fname, device=torch.device('cuda')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img


device = torch.device('cuda')

#%% md
# ## Download example data
#%%
dirname = 'data/Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total'
img_fnames = [os.path.join(dirname, x) for x in os.listdir(dirname) if '.png' in x][:-11]
plt.imshow(cv2.cvtColor(cv2.imread(img_fnames[0]), cv2.COLOR_BGR2RGB))
#%% md
# ## roma Matching
#%%

import torchvision
from pathlib import Path
from romatch import roma_outdoor, roma_indoor


def get_unique_idxs(A, dim=1):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted, = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies


def match_features(img_fnames,
                   index_pairs,
                   feature_dir='.featureout_roma',
                   device=torch.device('cuda'),
                   min_matches=15, resize_to_=(640, 480)):
    roma_model = roma_outdoor(device=device)

    Path(feature_dir).mkdir(exist_ok=True)
    with h5py.File(f'{feature_dir}/matches_roma.h5', mode='w') as f_match:
        for i, pair_idx in enumerate(index_pairs):
            print(f'Processing pair {i + 1}/{len(index_pairs)}')
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            # Load img1
            timg1 = load_torch_image(fname1, device=device)
            H1, W1 = timg1.shape[2:]
            if H1 < W1:
                resize_to = resize_to_[1], resize_to_[0]
            else:
                resize_to = resize_to_
            timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
            h1, w1 = timg_resized1.shape[2:]

            # Load img2
            timg2 = load_torch_image(fname2, device=device)
            H2, W2 = timg2.shape[2:]
            if H2 < W2:
                resize_to2 = resize_to[1], resize_to[0]
            else:
                resize_to2 = resize_to_
            timg_resized2 = K.geometry.resize(timg2, resize_to2, antialias=True)
            h2, w2 = timg_resized2.shape[2:]
            with torch.inference_mode():

                img_A = torchvision.transforms.functional.to_pil_image(timg_resized1.squeeze())
                img_B = torchvision.transforms.functional.to_pil_image(timg_resized2.squeeze())

                warp, certainty = roma_model.match(img_A, img_B, device=device)
                matches, certainty = roma_model.sample(warp, certainty)

            kptsA, kptsB = roma_model.to_pixel_coordinates(matches, h1, w1, h2, w2)

            mkpts0 = kptsA.numpy(force=True)
            mkpts1 = kptsB.numpy(force=True)

            mkpts0[:, 0] *= float(W1) / float(w1)
            mkpts0[:, 1] *= float(H1) / float(h1)

            mkpts1[:, 0] *= float(W2) / float(w2)
            mkpts1[:, 1] *= float(H2) / float(h2)

            # Custom condition to filter out matches based on pixel values
            valid_matches = []
            for pt0, pt1 in zip(mkpts0, mkpts1):
                x1, y1 = int(pt0[0]), int(pt0[1])
                x2, y2 = int(pt1[0]), int(pt1[1])

                # Retrieve pixel values from original images (assuming grayscale, adjust if RGB)
                pixel_value_img1 = timg1[0, 0, y1, x1].item()  # Pixel in image1 at (x1, y1)
                pixel_value_img2 = timg2[0, 0, y2, x2].item()  # Pixel in image2 at (x2, y2)

                # Condition to check pixel values (modify the threshold as needed)
                if (pixel_value_img1 > 1e-4) and (pixel_value_img2 > 1e-4):
                    valid_matches.append((pt0, pt1))

            # Unpack valid matches into separate mkpts0 and mkpts1 arrays
            mkpts0 = np.array([match[0] for match in valid_matches])
            mkpts1 = np.array([match[1] for match in valid_matches])

            n_matches = len(mkpts1)
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
    # Let's find unique roma pixels and group them together
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)
    with h5py.File(f'{feature_dir}/matches_roma.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]

            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match
    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]), dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            mkpts = np.concatenate([unique_kpts[k1][m2[:, 0]],
                                    unique_kpts[k2][m2[:, 1]],
                                    ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1

    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
    return


#%%

#%%

#%%
# matching all to all
index_pairs = []
for i in range(len(img_fnames)):
    for j in range(i + 1, len(img_fnames)):
        if i == j - 1 or True:
            index_pairs.append((i, j))

#%%
feature_dir = '.featureout_roma'
dev = torch.device('mps')
dev = torch.device('cuda')
match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=dev)
#%% md
# 
#%%

#%% md
# ## Import into colmap and run reconstruction
#%%
from h5_to_db import add_keypoints, add_matches, COLMAPDatabase


def import_into_colmap(img_dir,
                       feature_dir='.featureout',
                       database_path='colmap.db',
                       img_ext='.jpg'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, 'simple-radial', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )

    db.commit()
    return


database_path = Path('colmap_roma.db')
database_path.unlink(missing_ok=True)
import_into_colmap(dirname, feature_dir=feature_dir, database_path=database_path)


#%%

#%%

def run_reconstruction(image_dir, output_path='colmap_rec', database_path='colmap.db'):
    pycolmap.match_exhaustive(database_path)

    Path(output_path).mkdir(exist_ok=True)
    #
    # colmap_command = [
    #     "colmap",
    #     "mapper",
    #     "--database_path", str(database_path),
    #     "--output_path", output_path,
    #     "--image_path", str(image_dir),
    #     "--Mapper.tri_ignore_two_view_tracks", str(0),
    # ]
    #
    # subprocess.run(colmap_command, check=True, capture_output=True, text=True)
    maps = pycolmap.incremental_mapping(database_path, dirname, output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    maps[0].write(output_path)


#%%
output_path = './output'
run_reconstruction(dirname, output_path, database_path)
#%%

glomap_command = [
    "glomap",
    "mapper",
    "--database_path", './colmap_roma.db',
    "--output_path", './output',
    "--image_path", f'./{dirname}',
    # "--TrackEstablishment.min_num_view_per_track", str(2),
]

subprocess.run(glomap_command, check=True, capture_output=True, text=True)

#%% md
# ## Visualize with colmap GUI (yes, it is cheating)
