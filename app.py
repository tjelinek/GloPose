import threading
from typing import List

import gradio as gr
from pathlib import Path
import numpy as np
import cv2
import os
from hloc.utils import viz_3d
import pycolmap
# from pose.glomap import run_glomap_from_image_list
# from sift_baseline import (estimate_camera_poses_sift,
#                            default_opts,
#                            default_sift_keyframe_opts,
#                            get_keyframes_and_segmentations_sift,
#                            get_exhaustive_image_pairs)
from datetime import datetime
import shutil
import torch

from tracker_config import TrackerConfig
from tracker6d import Tracker6D
from utils.general import load_config

temp_dir = Path("temp")
os.makedirs(temp_dir, exist_ok=True)


def get_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    #elif torch.mps.is_available():
    #    return torch.device('mps')
    else:
        return torch.device('cpu')


def visualize_reconstruction(path_to_rec):
    if not isinstance(path_to_rec, pycolmap.Reconstruction):
        rec = pycolmap.Reconstruction(f'{path_to_rec}')
    else:
        rec = path_to_rec
    fig = viz_3d.init_figure()

    fig.update_layout(
        scene=dict(
            bgcolor='white'
        ),
        paper_bgcolor='white'
    )

    viz_3d.plot_cameras(fig, rec, color='rgba(50,255, 50, 255)', name="SIFT-LG", size=3)
    viz_3d.plot_reconstruction(fig, rec, cameras=False, color='rgba(255,50,255, 255', name="Cameras", cs=5)
    fig.show()
    return fig


def create_fake_segmentations(images, out_dir=temp_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        seg = (np.ones((h, w)) * 255).astype(np.uint8)
        fname = str(Path(out_dir) / f"segmentation_{i}.png")
        cv2.imwrite(fname, seg)
        out_paths.append(fname)
    return out_paths


# Function to handle input images and generate segmentations
def process_images(input_images):
    # Convert input images to numpy arrays
    images = [cv2.imread(img_fname) for (img_fname, _) in input_images]
    segmentation_paths = create_fake_segmentations(images)
    return segmentation_paths


def get_keyframes_and_segmentations_passthrough(input_images, segmentations):
    keyframes = input_images
    keysegs = segmentations
    matching_pairs = get_exhaustive_image_pairs(keyframes)
    return [str(kf) for kf in keyframes], [str(seg) for seg in keysegs], matching_pairs


def get_keyframes_and_segmentations(input_images, segmentations, global_vars, alg='passthrough', _skip_slider=1000,
                                    _too_little_slider=0, _matchability_slider=.5, _min_certainty_slider=.95,
                                    _device_radio='cpu', _matchability_radio=None,
                                    progress=gr.Progress()):
    input_images = [img for (img, _) in input_images]
    segmentations = [seg for (seg, _) in segmentations]
    if alg == 'passthrough':
        k, s, mp = get_keyframes_and_segmentations_passthrough(input_images, segmentations)
    elif alg == 'sift':
        opts = default_sift_keyframe_opts()
        opts['device'] = _device_radio
        opts['min_matches'] = _too_little_slider
        opts['good_to_add_matches'] = _skip_slider
        k, s, mp = get_keyframes_and_segmentations_sift(input_images, segmentations, opts, progress=progress)
    elif alg == 'matchability':
        opts = default_matchability_opts()
        opts['device'] = _device_radio
        opts['matchability_score'] = _matchability_slider
        opts['matchability_algorithm'] = _matchability_radio
        opts['roma_certainty'] = _min_certainty_slider
        k, s, mp = get_keyframes_and_segmentations_roma(input_images, segmentations, opts, progress=progress,
                                                        stop_event=stop_event)
    else:
        raise ValueError(f"Unknown algorithm {alg}")
    global_vars["matching_pairs"].append(mp)
    return k, s, global_vars


def ensure_same_dir(keyframes: List[Path]):
    assert len(keyframes) > 0
    base_image_path = keyframes[0].parent
    files_same_dir = [keyframes[0]]
    for img_path in keyframes[1:]:
        new_img_path = base_image_path / img_path.name
        if img_path != new_img_path:
            shutil.copy(img_path, new_img_path)
        files_same_dir.append(new_img_path)

    return files_same_dir


def on_glotrack_click(input_images, segmentations, global_vars, mapper='colmap', matcher='RoMa',
                      num_features=1024, device_radio_='cpu', progress=gr.Progress()):
    keyframes = [Path(img) for (img, _) in input_images]
    segmentations = [Path(seg) for (seg, _) in segmentations]

    keyframes = ensure_same_dir(keyframes)
    segmentations = ensure_same_dir(segmentations)

    matching_pairs = global_vars["matching_pairs"][-1]
    opts = default_opts()
    temp_dir = Path("temp_rec")
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    temp_dir = temp_dir / f"temp_{current_time}"
    os.makedirs(temp_dir, exist_ok=True)
    opts = {**opts,
            "feature_dir": temp_dir / 'featureout',
            "database_path": temp_dir / 'colmap.db',
            "output_path": temp_dir / f'{mapper}_rec',
            "mapper": mapper,
            "num_feats": num_features,
            "sample_size": num_features,
            "min_matches": 50,
            "device": _device,
            "img_ext": '.png'}

    with torch.inference_mode():
        poses, colmap_rec = run_glomap_from_image_list(keyframes, segmentations, matching_pairs, options=opts,
                                                       progress=progress)

    fig = visualize_reconstruction(colmap_rec)
    return fig


def update_sequences(dataset):
    sequences_map = {
        "GoogleScannedObjects": [
            "INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count",
            "Twinlab_Nitric_Fuel",
            "Squirrel",
            "STACKING_BEAR",
            "Schleich_Allosaurus",
            "Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total",
            "SCHOOL_BUS",
            "Sootheze_Cold_Therapy_Elephant",
            "TOP_TEN_HI",
            "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        ],
        "SyntheticObjects": [
            "Textured_Sphere_5_y",
        ],
        "HO3D": [
            "ABF10", "BB10", "GPMF10", "GSF10", "MC1", "MDF10", "ND2", "ShSu12", "SiBF12", "SM3", "SMu41",
            "ABF11", "BB11", "GPMF11", "GSF11", "MC2", "MDF11", "SB10", "ShSu13", "SiBF13", "SM4", "SMu42",
            "ABF12", "BB12", "GPMF12", "GSF12", "MC4", "MDF12", "SB12", "ShSu14", "SiBF14", "SM5", "SS1",
            "ABF13", "BB13", "GPMF13", "GSF13", "MC5", "MDF13", "SB14", "SiBF10", "SiS1", "SMu1", "SS2",
            "ABF14", "BB14", "GPMF14", "GSF14", "MC6", "MDF14", "ShSu10", "SiBF11", "SM2", "SMu40", "SS3",
        ],
        "HANDAL": [
            "000001",
            "000002",
            "000003",
            "000004",
            "000005",
        ],
        "Custom": []
    }
    return gr.update(choices=sequences_map.get(dataset, []), interactive=True)


with gr.Blocks() as demo:
    stop_event = threading.Event()


    def stop_computation():
        stop_event.set()  # Set the stop signal
        return "Computation Stopped"


    GLOBAL_VARS = gr.State({
        "matching_pairs": [], })

    with gr.Row():
        with gr.Column():
            dataset_input = gr.Dropdown(
                label="Dataset",
                choices=["Custom", "GoogleScannedObjects", "SyntheticObjects", "HO3D", "HANDAL"],
                value="Custom",
            )

        with gr.Column():
            sequence_input = gr.Dropdown(label="Sequence")

        dataset_input.change(update_sequences, inputs=[dataset_input], outputs=[sequence_input])

    with gr.Row():
        input_gallery = gr.Gallery(label="Input Images")
        segmentations_gallery = gr.Gallery(label="White Masks (Segmentations)")
    with gr.Row():
        with gr.Column():
            keyframe_radio = gr.Radio(["sift", "passthrough", "matchability"],
                                      label='Keyframe estimation algorithm', value="sift")
        with gr.Column():
            matchability_choices = ["Match frames in (last kf, current) once lost",
                                    "Match every kf once lost", ]
            matchability_radio = gr.Radio(matchability_choices,
                                          label='Matchability Algorithm', value=matchability_choices[0])

        with gr.Column():
            skip_slider = gr.Slider(minimum=50, maximum=10000, step=256, label="Skip frame if more than X matches",
                                    value=500)
            too_little_slider = gr.Slider(minimum=0, maximum=1000, step=10, label="Go back if less than X matches",
                                          value=100)
        with gr.Column():
            matchability_slider = gr.Slider(minimum=0., maximum=1., step=0.05,
                                            label="New match if less than X reliable",
                                            value=0.5)
            reliability_slider = gr.Slider(minimum=0., maximum=1., step=0.05, label="Reliable >= X RoMa certainty",
                                           value=0.95)
        with gr.Column():
            device_radio_filter = gr.Radio(["cpu", "cuda"], label='Device', value="cuda")
        with gr.Column():
            keyframes_button = gr.Button("Estimate Keyframes")
            stop_button = gr.Button("Stop")

    with gr.Row():
        filtered_fallery = gr.Gallery(label="Keyframes")
        filtered_segmentations = gr.Gallery(label="KeySegmentations")
    with gr.Row():
        mapper_radio = gr.Radio(["colmap", "pycolmap", "glomap"], label='SfM Engine', value="pycolmap")
        matcher_radio = gr.Radio(["RoMa", "SIFT"], label='Matcher', value="RoMa")
        device_radio_matcher = gr.Radio(["cpu", "cuda"], label='Device', value="cuda")
        num_features = gr.Slider(minimum=1024, maximum=1024 * 10, step=256, label="Number of Features", value=8192)
        glotrack_button = gr.Button("GloTrack")
    with gr.Row():
        vis_plot = gr.Plot(visible=True)
    input_gallery.upload(process_images, inputs=input_gallery, outputs=segmentations_gallery)

    keyframes_button.click(get_keyframes_and_segmentations,
                           inputs=[input_gallery, segmentations_gallery, GLOBAL_VARS, keyframe_radio, skip_slider,
                                   too_little_slider, matchability_slider, reliability_slider, device_radio_filter,
                                   matchability_radio],
                           outputs=[filtered_fallery, filtered_segmentations, GLOBAL_VARS])

    glotrack_button.click(on_glotrack_click, inputs=[filtered_fallery, filtered_segmentations,
                                                     GLOBAL_VARS, mapper_radio, matcher_radio, num_features,
                                                     device_radio_matcher],
                          outputs=vis_plot)

demo.launch(share=True)
