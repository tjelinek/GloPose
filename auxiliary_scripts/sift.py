import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from tqdm import tqdm

from auxiliary_scripts.colmap.colmap_database import COLMAPDatabase
from auxiliary_scripts.colmap.h5_to_db import add_keypoints, add_matches

temp_dir = Path("/mnt/personal/jelint19/cache/sift_cache")
os.makedirs(temp_dir, exist_ok=True)

def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


def detect_sift(img_fnames,
                segmentations=None,
                num_feats=2048,
                device=torch.device('cpu'),
                feature_dir='.featureout', resize_to=(800, 600), progress=None):
    sift = cv2.SIFT_create(num_feats, edgeThreshold=-1000, contrastThreshold=-1000)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
            h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for i, img_path in tqdm(enumerate(img_fnames)):

            if segmentations is not None:
                seg = cv2.imread(segmentations[i], cv2.IMREAD_GRAYSCALE)
            else:
                seg = None
            img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            hw1 = torch.tensor(img1.shape[:2], device=device)
            img_fname = img_path.split('/')[-1]
            key = img_fname
            kpts1, descs1 = sift.detectAndCompute(img1, seg)
            lafs1 = laf_from_opencv_SIFT_kpts(kpts1)
            descs1 = sift_to_rootsift(torch.from_numpy(descs1)).to(device)
            desc_dim = descs1.shape[-1]
            kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
            descs1 = descs1.reshape(-1, desc_dim).detach().cpu().numpy()
            f_laf[key] = lafs1.detach().cpu().numpy()
            f_kp[key] = kpts
            f_desc[key] = descs1
            if progress is not None:
                progress(i / len(img_fnames), "SIFT Detection")
    return


def get_unique_idxs(A):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies


def match_features(img_fnames,
                   index_pairs,
                   feature_dir='.featureout',
                   device=torch.device('cpu'),
                   alg='lightglue',
                   min_matches=15,
                   verbose=False, progress=None):
    alg = alg.lower()
    assert alg in ['lightglue', 'adalam']
    if alg == 'lightglue':
        matcher = K.feature.LightGlueMatcher('sift').eval().to(device)
    elif alg == 'adalam':
        matcher = K.feature.match_adalam
    else:
        raise ValueError(f"Unknown matching algorithm {alg}")
    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
            h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        with torch.inference_mode():
            for i, pair_idx in tqdm(enumerate(index_pairs)):
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
                lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
                lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
                desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
                desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
                img1, img2 = cv2.imread(fname1), cv2.imread(fname2)
                hw1, hw2 = img1.shape[:2], img2.shape[:2]
                dists, idxs = matcher(desc1, desc2,
                                      lafs1, lafs2,  # Adalam takes into account also geometric information
                                      hw1=hw1, hw2=hw2)  # Adalam also benefits from knowing image size
                if progress is not None:
                    progress(i / len(index_pairs), "SIFT Matching")
                if len(idxs) == 0:
                    continue
                n_matches = len(idxs)
                if verbose:
                    print(f'{key1}-{key2}: {n_matches} matches')
                group = f_match.require_group(key1)
                if n_matches >= min_matches:
                    group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
    return


def get_exhaustive_image_pairs(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i + 1, len(img_fnames)):
            index_pairs.append((i, j))
    return index_pairs


def import_into_colmap(img_dir,
                       feature_dir='.featureout',
                       database_path='colmap.db',
                       img_ext='.png'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = True
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, 'simple-pinhole', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()
    return

def default_opts():
    opts = {"feature_dir": '.featureout',
            "database_path": 'colmap.db',
            "output_path": 'glomap_rec',
            "img_ext": '.png',
            "device": torch.device('cpu'),
            "resize_to": (800, 600),
            "num_feats": 8192,
            "min_matches": 15,
            "mapper": 'colmap',
            "single_camera": True}
    return opts


def default_sift_keyframe_opts():
    opts = {"feature_dir": '.featureout_kf',
            "device": torch.device('cpu'),
            "resize_to": (800, 600),
            "num_feats": 8192,
            "matcher": 'adalam',
            "min_matches": 100,
            "good_to_add_matches": 450,
            }
    return opts


def get_keyframes_and_segmentations_sift(input_images, segmentations, options=None,
                                         progress=None):
    if options is None:
        options = default_sift_keyframe_opts()

    print("Detection features")
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    current_temp_dir = temp_dir / f"temp_{current_time}"
    current_temp_dir_images = current_temp_dir / 'images'
    os.makedirs(str(current_temp_dir_images), exist_ok=True)
    keyframes_single_dir = []
    for img in input_images:
        shutil.copy(img, current_temp_dir_images / Path(img).name)
        keyframes_single_dir.append(str(current_temp_dir_images / Path(img).name))
    detect_sift(keyframes_single_dir,
                segmentations,
                options['num_feats'],
                device=options['device'],
                feature_dir=options['feature_dir'], resize_to=options['resize_to'], progress=progress)
    matcher = K.feature.match_adalam
    feature_dir = options['feature_dir']
    device = options['device']
    selected_keyframe_idxs = [0]
    matching_pairs_original_idx = []
    print("Now matching to add keyframes")
    max_matches = options['good_to_add_matches']
    min_matches = options['min_matches']
    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc:
        idx1 = selected_keyframe_idxs[-1]
        fname1 = keyframes_single_dir[idx1]
        key1 = fname1.split('/')[-1]
        lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
        desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
        img1 = cv2.imread(fname1)
        hw1 = img1.shape[:2]
        done = False
        idx2 = idx1
        we_stepped_back = False
        while not done:
            idx2 = idx2 + 1
            if progress is not None:
                progress(idx2 / len(keyframes_single_dir), "Estimating keyframes")
            is_last_frame = idx2 == len(keyframes_single_dir) - 1
            if idx2 >= len(keyframes_single_dir):
                break
            fname2 = keyframes_single_dir[idx2]
            key2 = fname2.split('/')[-1]
            lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            img2 = cv2.imread(fname2)
            hw2 = img2.shape[:2]
            with torch.inference_mode():
                dists, idxs = matcher(desc1, desc2,
                                      lafs1, lafs2,  # Adalam takes into account also geometric information
                                      hw1=hw1, hw2=hw2)
            num_matches = len(idxs)
            print(f'{key1}-{key2}: {len(idxs)} matches')
            if num_matches >= max_matches:
                if (not we_stepped_back):
                    print("Too many matches, skipping")
                    if (len(selected_keyframe_idxs) == 1) and is_last_frame:
                        # We need at least two keyframes
                        selected_keyframe_idxs.append(idx1)
                        matching_pairs_original_idx.append((idx1, idx2))
                        selected_keyframe_idxs.append(idx2)
                        break
                elif is_last_frame:
                    print("Last frame, adding")
                    selected_keyframe_idxs.append(idx1)
                    matching_pairs_original_idx.append((idx1, idx2))
                    selected_keyframe_idxs.append(idx2)
                    break
                else:
                    print(f"Step back was good, adding idx1={idx1}")
                    selected_keyframe_idxs.append(idx1)
                    we_stepped_back = False
                continue
            if (len(idxs) <= max_matches) and (len(idxs) >= min_matches):
                print("Adding keyframe")
                selected_keyframe_idxs.append(idx2)
                selected_keyframe_idxs.append(idx1)
                matching_pairs_original_idx.append((idx1, idx2))
                idx1 = idx2
                key1, lafs1, desc1, hw1 = key2, lafs2, desc2, hw2
            if len(idxs) < min_matches:  # try going back
                print("Too few matches, going back")
                idx1 = idx2 - 1
                we_stepped_back = True
                if (idx1 <= 0):
                    done = True
                elif (idx1 in selected_keyframe_idxs):
                    print(f"We cannot match {idx2}, skipping it")
                    idx2 += 1
                else:
                    fname1 = keyframes_single_dir[idx1]
                    key1 = fname1.split('/')[-1]
                    lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
                    desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
                    img1 = cv2.imread(fname1)
                    hw1 = img1.shape[:2]
    matching_pairs_new_idxs = []
    selected_keyframe_idxs = sorted(list(set(selected_keyframe_idxs)))
    print(f'{selected_keyframe_idxs=}')
    print(f'{matching_pairs_original_idx=}')

    for idx1, idx2 in matching_pairs_original_idx:
        matching_pairs_new_idxs.append((selected_keyframe_idxs.index(idx1), selected_keyframe_idxs.index(idx2)))
    keyframes = [keyframes_single_dir[i] for i in selected_keyframe_idxs]
    keysegs = [segmentations[i] for i in selected_keyframe_idxs]
    return keyframes, keysegs, matching_pairs_new_idxs
