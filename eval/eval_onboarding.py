from pathlib import Path
from typing import Dict

from kornia.geometry import Se3

from data_structures.view_graph import ViewGraph
from eval.eval_reconstruction import (
    evaluate_reconstruction,
    update_sequence_reconstructions_stats,
    update_dataset_reconstruction_statistics,
)
from configs.glopose_config import RunConfig, PathsConfig
from configs.components.bop_config import BaseBOPConfig


def evaluate_onboarding(
    view_graph: ViewGraph,
    gt_Se3_world2cam: Dict[int, Se3] | None,
    run: RunConfig,
    bop: BaseBOPConfig,
    write_folder: Path,
) -> None:
    """Evaluate an onboarding result and write CSV statistics.

    Reads metadata (timing, success flags, image_name_to_frame_id) from the
    ViewGraph. Writes per-keyframe, per-sequence, and per-dataset CSVs to the
    experiment results folder (two levels above write_folder).

    Args:
        view_graph: The ViewGraph returned by run_pipeline(), carrying metadata.
        gt_Se3_world2cam: Ground-truth world-to-camera poses keyed by frame index,
            or None if GT is unavailable.
        run: The RunConfig used for this run.
        bop: The BaseBOPConfig used for this run.
        write_folder: The per-sequence output folder (e.g. results/exp/dataset/seq/).
    """
    keyframe_nodes = sorted(view_graph.view_graph.nodes)
    num_keyframes = len(keyframe_nodes)

    # Determine whether we know GT poses for all keyframes
    if gt_Se3_world2cam is not None and len(gt_Se3_world2cam) > 0:
        known_gt_poses = all(idx in gt_Se3_world2cam for idx in keyframe_nodes)
    else:
        known_gt_poses = False

    if not known_gt_poses:
        return

    # Build dataset/sequence names for CSV columns
    dataset_name_for_eval = run.dataset
    if bop.onboarding_type is not None:
        dataset_name_for_eval = f'{dataset_name_for_eval}_{bop.onboarding_type}_onboarding'

    sequence_name = run.sequence
    if run.special_hash is not None and len(run.special_hash) > 0:
        sequence_name = f'{sequence_name}_{run.special_hash}'

    # CSV paths live two levels above write_folder (experiment root / dataset level)
    rec_csv_detailed_stats = write_folder.parent.parent / 'reconstruction_keyframe_stats.csv'
    rec_csv_per_sequence_stats = write_folder.parent.parent / 'reconstruction_sequence_stats.csv'

    # Load COLMAP reconstruction from disk if it succeeded
    reconstruction = None
    if view_graph.reconstruction_success:
        import pycolmap
        rec_path = view_graph.colmap_reconstruction_path
        if rec_path is not None and rec_path.exists():
            reconstruction = pycolmap.Reconstruction(str(rec_path))

    # Per-keyframe evaluation (rotation/translation errors)
    if reconstruction is not None and known_gt_poses:
        evaluate_reconstruction(
            reconstruction, gt_Se3_world2cam, view_graph.image_name_to_frame_id,
            rec_csv_detailed_stats, dataset_name_for_eval, sequence_name,
        )

    # Per-sequence summary statistics
    reconstruction_success = view_graph.reconstruction_success
    alignment_success = view_graph.alignment_success

    update_sequence_reconstructions_stats(
        rec_csv_detailed_stats, rec_csv_per_sequence_stats, num_keyframes,
        view_graph.num_input_frames, reconstruction, dataset_name_for_eval,
        sequence_name, reconstruction_success, alignment_success,
        view_graph.frame_filtering_time, view_graph.reconstruction_time,
    )

    # Per-dataset aggregate statistics
    update_dataset_reconstruction_statistics(rec_csv_per_sequence_stats, dataset_name_for_eval)


def resolve_gt_model_path(run: RunConfig, paths: PathsConfig) -> Path | None:
    """Resolve the path to the GT 3D model for the current dataset/object.

    Returns None if no GT model is available for the dataset.
    """
    dataset = run.dataset
    object_id = run.object_id

    # BOP datasets (handal, hope, tless, lmo, icbin, etc.)
    bop_datasets = {'handal', 'handal_native', 'hope', 'tless', 'lmo', 'icbin', 'itodd', 'tudl', 'ycbv', 'hb'}
    dataset_lower = dataset.lower()
    for bop_name in bop_datasets:
        if bop_name in dataset_lower:
            try:
                obj_int = int(object_id)
            except (ValueError, TypeError):
                return None
            bop_dataset_name = bop_name
            if bop_dataset_name == 'handal_native':
                bop_dataset_name = 'handal'
            model_path = paths.bop_data_folder / bop_dataset_name / 'models' / f'obj_{obj_int:06d}.ply'
            return model_path if model_path.exists() else None

    # NAVI
    if 'navi' in dataset_lower:
        if object_id is not None:
            obj_name = str(object_id)
            model_path = paths.navi_data_folder / obj_name / '3d_scan' / f'{obj_name}.obj'
            return model_path if model_path.exists() else None

    # HO3D
    if 'ho3d' in dataset_lower:
        if object_id is not None:
            model_path = paths.ho3d_data_folder / 'models' / str(object_id) / 'textured_simple.obj'
            return model_path if model_path.exists() else None

    return None
