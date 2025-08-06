import threading
from typing import List

import gradio as gr
from pathlib import Path
import numpy as np
import cv2
import os
from hloc.utils import viz_3d
import pycolmap
import shutil
import torch

from data_providers.flow_provider import UFMFlowProviderDirect
from pose.glomap import reconstruct_images_using_sfm
from run_from_webapp import prepare_config
from tracker6d import Tracker6D
from tracker_config import TrackerConfig
from utils.data_utils import get_initial_image_and_segment

temp_dir = Path("temp")
os.makedirs(temp_dir, exist_ok=True)


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


def get_keyframes_and_segmentations(input_images, segmentations, frame_filter='passthrough',
                                    skip_slider=1000, too_little_slider=0, matchability_slider=.5,
                                    min_certainty_slider=.95, device_radio='cpu', progress=gr.Progress()):
    input_images = [img for (img, _) in input_images]
    segmentations = [seg for (seg, _) in segmentations]

    config, write_folder = prepare_config(input_images)

    config.device = device_radio
    config.frame_filter = frame_filter
    config.sift_filter_min_matches = too_little_slider
    config.sift_filter_good_to_add_matches = skip_slider
    config.min_roma_certainty_threshold = min_certainty_slider
    config.flow_reliability_threshold = matchability_slider

    first_image_tensor, first_segment_tensor = get_initial_image_and_segment(input_images, segmentations)

    tracker = Tracker6D(config, write_folder, images_paths=input_images, segmentation_paths=segmentations,
                        initial_image=first_image_tensor, initial_segmentation=first_segment_tensor)
    keyframe_graph = tracker.filter_frames(progress)
    images_paths, segmentation_paths, matching_pairs = tracker.prepare_input_for_colmap(keyframe_graph)

    return images_paths, segmentation_paths, matching_pairs


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


def on_glotrack_click(input_images, segmentations, matching_pairs, matcher_radio, num_features,
                      write_folder: Path, mapper='colmap', device_radio_='cpu', progress=gr.Progress()):
    keyframes = [Path(img) for (img, _) in input_images]
    segmentations = [Path(seg) for (seg, _) in segmentations]

    keyframes = ensure_same_dir(keyframes)
    segmentations = ensure_same_dir(segmentations)

    config = TrackerConfig()
    if matcher_radio == 'UFM':
        match_provider = UFMFlowProviderDirect(device_radio_, config.ufm_config)
    elif matcher_radio == 'SIFT':
        raise NotImplementedError("SIFT is not implemented yet.")
        # match_provider = SIFTMatchingProvider(None, num_features, device_radio_)
    else:
        raise ValueError("Unknown Matching Provider")

    colmap_base_path = write_folder / f"glomap_'{write_folder.stem}"
    colmap_base_path.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        colmap_rec = reconstruct_images_using_sfm(keyframes, segmentations, matching_pairs,
                                                  config.init_with_first_two_images, mapper, match_provider,
                                                  config.roma_sample_size, colmap_base_path, device=device_radio_,
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
    _stop_event = threading.Event()


    def stop_computation():
        _stop_event.set()  # Set the stop signal
        return "Computation Stopped"


    # GLOBAL_VARS = gr.State({
    #     'pipeline_wrapper': None
    # })

    with gr.Row():
        with gr.Column():
            _dataset_input = gr.Dropdown(
                label="Dataset",
                choices=["Custom", "GoogleScannedObjects", "SyntheticObjects", "HO3D", "HANDAL"],
                value="Custom",
            )

        with gr.Column():
            _sequence_input = gr.Dropdown(label="Sequence")

        _dataset_input.change(update_sequences, inputs=[_dataset_input], outputs=[_sequence_input])

    with gr.Row():
        _input_gallery = gr.Gallery(label="Input Images")
        _segmentations_gallery = gr.Gallery(label="White Masks (Segmentations)")

    with gr.Row():
        with gr.Column():
            _frame_filter_radio = gr.Radio(["passthrough", "dense_matching"],
                                           label='Frame filter algorithm', value="dense_matching")
        with gr.Column():
            _skip_slider = gr.Slider(minimum=50, maximum=10000, step=256, label="Skip frame if more than X matches",
                                     value=500)
            _too_little_slider = gr.Slider(minimum=0, maximum=1000, step=10, label="Go back if less than X matches",
                                           value=100)
        with gr.Column():
            _matchability_slider = gr.Slider(minimum=0., maximum=1., step=0.05,
                                             label="New match if less than X reliable",
                                             value=0.5)
            _reliability_slider = gr.Slider(minimum=0., maximum=1., step=0.05, label="Reliable >= X RoMa certainty",
                                            value=0.95)
        with gr.Column():
            _device_radio_filter = gr.Radio(["cpu", "cuda"], label='Device', value="cuda")
        with gr.Column():
            _keyframes_button = gr.Button("Estimate Keyframes")
            _stop_button = gr.Button("Stop")

    with gr.Row():
        _filtered_gallery = gr.Gallery(label="Keyframes")
        _filtered_segmentations = gr.Gallery(label="KeySegmentations")

    with gr.Row():
        _mapper_radio = gr.Radio(["colmap", "pycolmap", "glomap"], label='SfM Engine', value="pycolmap")
        _matcher_radio = gr.Radio(["UFM", "SIFT"], label='Matcher', value="UFM")
        _device_radio_matcher = gr.Radio(["cpu", "cuda"], label='Device', value="cuda")
        _num_features = gr.Slider(minimum=1024, maximum=1024 * 10, step=256, label="Number of SIFT Features",
                                  value=8192)
        _glotrack_button = gr.Button("GloTrack")

    with gr.Row():
        _vis_plot = gr.Plot(visible=True)

    _input_gallery.upload(process_images, inputs=_input_gallery, outputs=_segmentations_gallery)

    _matching_pairs = []

    _keyframes_button.click(get_keyframes_and_segmentations,
                            inputs=[_input_gallery, _segmentations_gallery, _frame_filter_radio, _skip_slider,
                                    _too_little_slider, _matchability_slider, _reliability_slider,
                                    _device_radio_filter],
                            outputs=[_filtered_gallery, _filtered_segmentations, _matching_pairs])

    _glotrack_button.click(on_glotrack_click, inputs=[_filtered_gallery, _filtered_segmentations, _matching_pairs,
                                                      _mapper_radio, _matcher_radio, _num_features,
                                                      _device_radio_matcher],
                           outputs=_vis_plot)

demo.launch(share=True)
