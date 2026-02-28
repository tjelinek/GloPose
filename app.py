import hashlib
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, List

import gradio as gr
import pycolmap
import torch
from hloc.utils import viz_3d

from data_providers.flow_provider import UFMFlowProviderDirect
from data_providers.frame_provider import PrecomputedSegmentationProvider
from configs.glopose_config import GloPoseConfig
from onboarding_pipeline import OnboardingPipeline
from pose.glomap import reconstruct_images_using_sfm
from utils.data_utils import is_video_input
from utils.dataset_sequences import (
    get_handal_sequences,
    get_ho3d_sequences,
    get_navi_sequences,
    get_behave_sequences,
    get_google_scanned_objects_sequences,
    get_tum_rgbd_sequences,
    get_bop_onboarding_sequences,
)
from utils.general import load_config
from utils.image_utils import get_video_length_in_frames

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLOPOSE_DIR = Path(__file__).resolve().parent

DATASET_TO_RUNNER = {
    "HANDAL": "run_HANDAL.py",
    "HO3D": "run_HO3D.py",
    "HOPE": "run_HOPE.py",
    "NAVI": "run_NAVI.py",
    "GoogleScannedObjects": "run_GoogleScannedObjects.py",
    "SyntheticObjects": "run_SyntheticObjects.py",
    "BEHAVE": "run_BEHAVE.py",
    "TUM_RGBD": "run_TUM_RGBD.py",
}

SYNTHETIC_OBJECTS_SEQUENCES = [
    "Textured_Sphere_5_y",
]

DEFAULT_RESULTS_ROOT = Path("/mnt/personal/jelint19/results/FlowTracker/")
DEFAULT_DATA_FOLDER = Path("/mnt/data/vrg/public_datasets/")

# ---------------------------------------------------------------------------
# Global state for Custom Input tab
# ---------------------------------------------------------------------------

images_for_reconstruction_global: List[Path] = []
segmentation_for_reconstruction_global: List[Path] = []
matching_pairs_global: list = []
write_folder_global: Optional[Path] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan_config_files() -> List[str]:
    """Return config file paths relative to GloPose root, sorted."""
    configs_dir = GLOPOSE_DIR / "configs"
    if not configs_dir.exists():
        return ["configs/base_config.py"]
    config_files = []
    for p in sorted(configs_dir.rglob("*.py")):
        rel = p.relative_to(GLOPOSE_DIR)
        # Skip internal sub-configs and __pycache__
        if "__pycache__" in str(rel):
            continue
        parts = rel.parts
        if any(d in parts for d in ("components", "matching", "sequences_configs")):
            continue
        config_files.append(str(rel))
    return config_files or ["configs/base_config.py"]


def _get_sequences_for_dataset(dataset: str) -> List[str]:
    """Discover sequences for a given dataset from disk."""
    data_folder = DEFAULT_DATA_FOLDER
    try:
        if dataset == "HANDAL":
            train, test = get_handal_sequences(data_folder / "HANDAL")
            return train + test
        elif dataset == "HO3D":
            train, test = get_ho3d_sequences(data_folder / "HO3D")
            return train + test
        elif dataset == "NAVI":
            return get_navi_sequences(data_folder / "NAVI" / "navi_v1.5")
        elif dataset == "BEHAVE":
            return get_behave_sequences(data_folder / "BEHAVE" / "train")
        elif dataset == "GoogleScannedObjects":
            return get_google_scanned_objects_sequences(data_folder / "GoogleScannedObjects" / "models")
        elif dataset == "TUM_RGBD":
            return get_tum_rgbd_sequences(data_folder / "SLAM" / "tum_rgbd")
        elif dataset == "HOPE":
            dyn, up, down, both = get_bop_onboarding_sequences(data_folder / "bop", "hope")
            return dyn + up + down + both
        elif dataset == "SyntheticObjects":
            return list(SYNTHETIC_OBJECTS_SEQUENCES)
    except Exception:
        pass
    return []


def visualize_reconstruction(path_to_rec):
    if not isinstance(path_to_rec, pycolmap.Reconstruction):
        rec = pycolmap.Reconstruction(str(path_to_rec))
    else:
        rec = path_to_rec
    fig = viz_3d.init_figure()
    fig.update_layout(scene=dict(bgcolor="white"), paper_bgcolor="white")
    viz_3d.plot_cameras(fig, rec, color="rgba(50,255,50,255)", name="Cameras", size=3)
    viz_3d.plot_reconstruction(fig, rec, cameras=False, color="rgba(255,50,255,255)", name="Points", cs=5)
    return fig


def _scan_subdirs(path: Path) -> List[str]:
    """Return sorted list of subdirectory names under *path*."""
    if not path.is_dir():
        return []
    return sorted(d.name for d in path.iterdir() if d.is_dir())


def _find_glomap_subdir(seq_path: Path) -> Optional[Path]:
    """Find the first glomap_* directory inside *seq_path*."""
    if not seq_path.is_dir():
        return None
    for d in sorted(seq_path.iterdir()):
        if d.is_dir() and d.name.startswith("glomap_"):
            return d
    return None


# =========================================================================
# Tab 1: Run on Dataset
# =========================================================================

_dataset_process: Optional[subprocess.Popen] = None
_dataset_process_lock = threading.Lock()


def update_dataset_sequences(dataset):
    seqs = _get_sequences_for_dataset(dataset)
    return gr.update(choices=seqs, value=seqs[0] if seqs else None, allow_custom_value=True)


def run_dataset(dataset, sequence, config_file, experiment, device):
    """Generator that streams subprocess output line-by-line."""
    global _dataset_process

    if dataset not in DATASET_TO_RUNNER:
        yield f"Unknown dataset: {dataset}"
        return

    runner = GLOPOSE_DIR / DATASET_TO_RUNNER[dataset]
    if not runner.exists():
        yield f"Runner script not found: {runner}"
        return

    cmd = [
        sys.executable, str(runner),
        "--config", config_file,
        "--sequences", sequence,
        "--experiment", experiment,
    ]

    env = os.environ.copy()
    if device:
        env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else "0"

    yield f"$ {' '.join(cmd)}\n"

    with _dataset_process_lock:
        _dataset_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(GLOPOSE_DIR),
            env=env,
        )

    log = ""
    proc = _dataset_process
    for line in iter(proc.stdout.readline, ""):
        log += line
        yield log
    proc.wait()

    with _dataset_process_lock:
        _dataset_process = None

    status = "SUCCESS" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
    log += f"\n--- {status} ---\n"
    yield log


def stop_dataset_run():
    global _dataset_process
    with _dataset_process_lock:
        if _dataset_process is not None and _dataset_process.poll() is None:
            _dataset_process.terminate()
            try:
                _dataset_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _dataset_process.kill()
            _dataset_process = None
            return "Process terminated."
    return "No running process."


# =========================================================================
# Tab 2: Custom Input
# =========================================================================

_custom_stop_event = threading.Event()


def stop_custom_computation():
    _custom_stop_event.set()
    return "Computation stopped."


def get_keyframes_and_segmentations(
        input_images, segmentations, config_file, frame_filter, matchability_slider,
        min_certainty_slider, device_radio, progress=gr.Progress()
):
    global matching_pairs_global, images_for_reconstruction_global
    global segmentation_for_reconstruction_global, write_folder_global

    _custom_stop_event.clear()

    input_images = [Path(img) for (img, _) in input_images]
    segmentations = [Path(seg) for (seg, _) in segmentations] if segmentations else None

    # Build config inline (no CLI parse_args dependency)
    config_path = GLOPOSE_DIR / config_file
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = GloPoseConfig()

    combined_paths = "\n".join(str(p) for p in input_images)
    sequence_hash = hashlib.sha256(combined_paths.encode("utf-8")).hexdigest()[:16]

    config.run.dataset = "custom_input"
    config.run.sequence = sequence_hash
    config.run.experiment_name = "webapp"
    config.input.frame_provider = "precomputed"
    config.input.frame_provider_config.erode_segmentation = True
    config.run.device = device_radio
    config.onboarding.frame_filter = frame_filter
    config.onboarding.min_certainty_threshold = min_certainty_slider
    config.onboarding.flow_reliability_threshold = matchability_slider

    write_folder = config.paths.results_folder / "webapp" / "custom_input" / sequence_hash
    write_folder_global = write_folder

    if segmentations is not None:
        first_segment_tensor = PrecomputedSegmentationProvider.get_initial_segmentation(
            input_images, segmentations, device=device_radio
        )
    else:
        first_segment_tensor = None
        config.input.segmentation_provider = "whites"

    if is_video_input(input_images):
        input_images = input_images[0]
    if segmentations is not None and is_video_input(segmentations):
        segmentations = segmentations[0]

    config.input.input_frames = len(input_images) if isinstance(input_images, list) else get_video_length_in_frames(
        input_images)
    config.input.skip_indices = math.ceil(config.input.input_frames / 200)

    tracker = OnboardingPipeline(
        config, write_folder, input_images=input_images,
        input_segmentations=segmentations,
        initial_segmentation=first_segment_tensor, progress=progress,
    )
    keyframe_graph = tracker.filter_frames(progress, _custom_stop_event)
    images_paths, segmentation_paths, matching_pairs = tracker.prepare_input_for_colmap(keyframe_graph)

    matching_pairs_global = matching_pairs
    images_for_reconstruction_global = images_paths
    segmentation_for_reconstruction_global = segmentation_paths

    temp_dir = Path("/tmp")
    os.makedirs(temp_dir, exist_ok=True)

    temp_images = []
    temp_segmentations = []
    for i, img_path in enumerate(images_paths):
        temp_img_path = temp_dir / f"keyframe_{i}_{Path(img_path).name}"
        shutil.copy(img_path, temp_img_path)
        temp_images.append(str(temp_img_path))
    for i, seg_path in enumerate(segmentation_paths):
        temp_seg_path = temp_dir / f"keyseg_{i}_{Path(seg_path).name}"
        shutil.copy(seg_path, temp_seg_path)
        temp_segmentations.append(str(temp_seg_path))

    return temp_images, temp_segmentations


def on_reconstruct_click(
        mapper, matcher_radio, num_features, device_radio, progress=gr.Progress()
):
    global matching_pairs_global, images_for_reconstruction_global
    global segmentation_for_reconstruction_global, write_folder_global

    config = GloPoseConfig()
    if matcher_radio == "UFM":
        match_provider = UFMFlowProviderDirect(device_radio, config.onboarding.ufm)
    elif matcher_radio == "SIFT":
        raise NotImplementedError("SIFT is not implemented yet.")
    else:
        raise ValueError(f"Unknown matching provider {matcher_radio}")

    colmap_base_path = write_folder_global / f"glomap_{write_folder_global.stem}"
    colmap_base_path.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        colmap_rec = reconstruct_images_using_sfm(
            images_for_reconstruction_global,
            segmentation_for_reconstruction_global,
            matching_pairs_global,
            config.onboarding.init_with_first_two_images,
            mapper,
            match_provider,
            config.onboarding.sample_size,
            colmap_base_path,
            config.onboarding.add_track_merging_matches,
            device=device_radio,
            progress=progress,
        )

    return visualize_reconstruction(colmap_rec)


# =========================================================================
# Tab 3: Browse Results
# =========================================================================


def scan_experiments(results_root):
    root = Path(results_root)
    return gr.update(choices=_scan_subdirs(root), value=None)


def scan_datasets(results_root, experiment):
    if not experiment:
        return gr.update(choices=[], value=None)
    root = Path(results_root) / experiment
    return gr.update(choices=_scan_subdirs(root), value=None)


def scan_sequences(results_root, experiment, dataset):
    if not experiment or not dataset:
        return gr.update(choices=[], value=None)
    root = Path(results_root) / experiment / dataset
    return gr.update(choices=_scan_subdirs(root), value=None)


def load_results(results_root, experiment, dataset, sequence):
    if not all([results_root, experiment, dataset, sequence]):
        return [], [], None

    seq_path = Path(results_root) / experiment / dataset / sequence
    glomap_dir = _find_glomap_subdir(seq_path)

    images = []
    segmentations = []
    fig = None

    if glomap_dir is None:
        return images, segmentations, fig

    # Load keyframe images
    images_dir = glomap_dir / "images"
    if images_dir.is_dir():
        images = sorted(str(p) for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))

    # Load segmentations
    seg_dir = glomap_dir / "segmentations"
    if seg_dir.is_dir():
        segmentations = sorted(str(p) for p in seg_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

    # Load 3D reconstruction
    rec_dir = glomap_dir / "output" / "0"
    if rec_dir.is_dir() and (rec_dir / "cameras.bin").exists():
        try:
            fig = visualize_reconstruction(rec_dir)
        except Exception as e:
            print(f"Failed to load reconstruction: {e}")

    return images, segmentations, fig


# =========================================================================
# Build Gradio UI
# =========================================================================

config_choices = _scan_config_files()

with gr.Blocks(title="GloPose - 6DoF Pose Tracking") as demo:
    gr.Markdown("# GloPose - 6DoF Object Pose Tracking & Reconstruction")

    # ---- Tab 1: Run on Dataset ----
    with gr.Tab("Run on Dataset"):
        with gr.Row():
            ds_dataset = gr.Dropdown(
                label="Dataset",
                choices=list(DATASET_TO_RUNNER.keys()),
                value="HANDAL",
            )
            ds_sequence = gr.Dropdown(
                label="Sequence",
                choices=[],
                allow_custom_value=True,
            )

        with gr.Row():
            ds_config = gr.Dropdown(
                label="Config",
                choices=config_choices,
                value="configs/base_config.py",
                allow_custom_value=True,
            )
            ds_experiment = gr.Textbox(label="Experiment Name", value="default")
            ds_device = gr.Radio(["cpu", "cuda"], label="Device", value="cuda")

        with gr.Row():
            ds_run_btn = gr.Button("Run", variant="primary")
            ds_stop_btn = gr.Button("Stop", variant="stop")

        ds_log = gr.Textbox(label="Log Output", lines=20, max_lines=40, interactive=False)
        ds_stop_status = gr.Textbox(label="Status", visible=False)

        # Wire events
        ds_dataset.change(update_dataset_sequences, inputs=[ds_dataset], outputs=[ds_sequence])
        ds_run_btn.click(
            run_dataset,
            inputs=[ds_dataset, ds_sequence, ds_config, ds_experiment, ds_device],
            outputs=[ds_log],
        )
        ds_stop_btn.click(stop_dataset_run, outputs=[ds_stop_status])

    # ---- Tab 2: Custom Input ----
    with gr.Tab("Custom Input"):
        with gr.Row():
            ci_images = gr.Gallery(label="Input Images / Video Frames")
            ci_segmentations = gr.Gallery(label="Masks (optional)")

        with gr.Row():
            ci_config = gr.Dropdown(
                label="Config",
                choices=config_choices,
                value="configs/base_config.py",
                allow_custom_value=True,
            )
            ci_filter = gr.Radio(
                ["passthrough", "dense_matching"],
                label="Frame filter algorithm",
                value="dense_matching",
            )
            ci_matchability = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05,
                label="Matchability threshold",
                value=0.5,
            )
            ci_certainty = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05,
                label="Reliable match certainty",
                value=0.95,
            )
            ci_device_filter = gr.Radio(["cpu", "cuda"], label="Device", value="cuda")

        with gr.Row():
            ci_keyframes_btn = gr.Button("Estimate Keyframes", variant="primary")
            ci_stop_btn = gr.Button("Stop", variant="stop")

        with gr.Row():
            ci_filtered = gr.Gallery(label="Keyframes")
            ci_filtered_seg = gr.Gallery(label="Key Segmentations")

        with gr.Row():
            ci_mapper = gr.Radio(["colmap", "pycolmap", "glomap"], label="SfM Engine", value="pycolmap")
            ci_matcher = gr.Radio(["UFM", "SIFT"], label="Matcher", value="UFM")
            ci_device_matcher = gr.Radio(["cpu", "cuda"], label="Device", value="cuda")
            ci_num_features = gr.Slider(
                minimum=1024, maximum=10240, step=256,
                label="SIFT Features", value=8192,
            )
            ci_recon_btn = gr.Button("Run Reconstruction", variant="primary")

        ci_plot = gr.Plot(label="3D Reconstruction")

        # Wire events
        ci_keyframes_btn.click(
            get_keyframes_and_segmentations,
            inputs=[ci_images, ci_segmentations, ci_config, ci_filter,
                    ci_matchability, ci_certainty, ci_device_filter],
            outputs=[ci_filtered, ci_filtered_seg],
        )
        ci_stop_btn.click(stop_custom_computation)
        ci_recon_btn.click(
            on_reconstruct_click,
            inputs=[ci_mapper, ci_matcher, ci_num_features, ci_device_matcher],
            outputs=[ci_plot],
        )

    # ---- Tab 3: Browse Results ----
    with gr.Tab("Browse Results"):
        br_root = gr.Textbox(
            label="Results Root",
            value=str(DEFAULT_RESULTS_ROOT),
        )

        with gr.Row():
            br_experiment = gr.Dropdown(label="Experiment", choices=[], allow_custom_value=True)
            br_dataset = gr.Dropdown(label="Dataset", choices=[], allow_custom_value=True)
            br_sequence = gr.Dropdown(label="Sequence", choices=[], allow_custom_value=True)

        br_refresh_btn = gr.Button("Scan Results Directory")
        br_load_btn = gr.Button("Load", variant="primary")

        with gr.Row():
            br_images = gr.Gallery(label="Keyframe Images")
            br_segmentations = gr.Gallery(label="Segmentations")

        br_plot = gr.Plot(label="3D Reconstruction")

        # Wire events
        br_refresh_btn.click(scan_experiments, inputs=[br_root], outputs=[br_experiment])
        br_experiment.change(scan_datasets, inputs=[br_root, br_experiment], outputs=[br_dataset])
        br_dataset.change(scan_sequences, inputs=[br_root, br_experiment, br_dataset], outputs=[br_sequence])
        br_load_btn.click(
            load_results,
            inputs=[br_root, br_experiment, br_dataset, br_sequence],
            outputs=[br_images, br_segmentations, br_plot],
        )

    # Populate dataset sequences on load
    demo.load(update_dataset_sequences, inputs=[ds_dataset], outputs=[ds_sequence])

demo.launch(share=True)
