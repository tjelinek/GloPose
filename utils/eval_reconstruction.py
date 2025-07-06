from pathlib import Path
from typing import Dict

import pandas as pd
from kornia.geometry import Se3

from pose.glomap import get_image_Se3_world2cam


def evaluate_reconstruction(
        reconstruction,
        ground_truth_poses: Dict[int, Se3],
        image_name_to_frame_id: Dict[str, int],
        csv_output_path: Path,
        dataset: str,
        sequence: str
):

    stats = []

    for image in reconstruction.images.values():
        image_frame_id = image_name_to_frame_id[image.name]

        Se3_obj2cam_pred = get_image_Se3_world2cam(image, 'cpu')

        # Get ground truth pose for this frame
        Se3_world2cam_gt = ground_truth_poses.get(image_frame_id)

        # Ground-truth rotation and translation
        gt_rotation = Se3_world2cam_gt.rotation.matrix().tolist() if Se3_world2cam_gt is not None else None
        gt_translation = Se3_world2cam_gt.translation.tolist() if Se3_world2cam_gt is not None else None

        pred_rotation = Se3_obj2cam_pred.rotation.matrix().tolist()
        pred_translation = Se3_obj2cam_pred.translation.tolist()

        # Add stats for the current image frame
        stats.append({
            'dataset': dataset,
            'sequence': sequence,
            'image_frame_id': image_frame_id,
            'gt_rotation': gt_rotation,
            'gt_translation': gt_translation,
            'pred_rotation': pred_rotation,
            'pred_translation': pred_translation
        })

    # Convert stats to a Pandas DataFrame
    stats_df = pd.DataFrame(stats)

    # If the CSV file exists, append; otherwise, create a new one
    if csv_output_path.exists():
        existing_df = pd.read_csv(csv_output_path)
        filtered_df = existing_df[~((existing_df['dataset'] == dataset) &
                                    (existing_df['sequence'] == sequence))]
        updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
        updated_df.to_csv(csv_output_path, index=False)
    else:
        stats_df.to_csv(csv_output_path, index=False)
