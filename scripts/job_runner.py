import os
import subprocess
from enum import Enum
from pathlib import Path


class Datasets(Enum):
    SyntheticObjects = "SyntheticObjects"
    GoogleScannedObjects = "GoogleScannedObjects"
    # photo360 = "360photo"


runners = {
    Datasets.SyntheticObjects: "run_SyntheticObjects.py",
    Datasets.GoogleScannedObjects: "run_GoogleScannedObjects.py",
    # Datasets.photo360: "run_360.py"
}


def run_batch(configuration_name: str, sequences, dataset: Datasets, output_folder: Path) -> None:
    configuration_path = Path('configs') / (configuration_name + '.py')

    args = []
    args.append("--config")
    args.append(configuration_path)
    args.append("--dataset-runner")
    args.append(runners[dataset])
    args.append("--experiment")
    args.append(str(Path(configuration_name).stem))
    args.append("--output-folder")
    args.append(output_folder)
    args.append("--sequences")
    args.extend(sequences)

    # Echo the arguments
    print("Running sbatch job.batch with arguments:", args)
    print('----------------------------------------')
    subprocess.run(["sbatch", "scripts/job.batch"] + args)


def create_unused_folder(output_folder: Path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    i = 1
    while os.path.exists(f"{output_folder}_{i}"):
        i += 1
    final_output_folder = f"{output_folder}_{i}"
    os.makedirs(final_output_folder, exist_ok=True)

    return final_output_folder


def main():
    configurations = [
        # "config_deep_no_flow_all_frames_keyframes",
        # "config_deep_no_flow_gt_mesh",
        # "config_deep_no_flow",
        # "config_deep_only_flow_gt_mesh",
        # "config_deep_only_flow_gt_one_kf",
        # "config_deep_only_flow_gt_two_kf",
        # "config_deep_only_flow",
        # "config_deep_with_flow_all_frames_keyframes",
        # "config_deep_with_flow_gt_no_rgb",
        # "config_deep_with_flow_gt",
        # "config_deep_with_flow_gt_only_frontview",
        # "config_deep_with_flow_gt_with_backview",
        # "config_deep_with_flow_no_rgb",
        # "config_deep_with_flow",
        'only_ok_flows_thresholds/thresh_001',
        'only_ok_flows_thresholds/thresh_01',
        'only_ok_flows_thresholds/thresh_1',
        'only_ok_flows_thresholds/thresh_05',
        'only_ok_flows_thresholds/thresh_3',
        # 'essential_matrix/config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point',
        # 'essential_matrix/config_deep_with_flow_gt_esmatrix_frontview_backview_synthetic_8point',
        # 'essential_matrix/config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_magsac',
        # 'essential_matrix/config_deep_with_flow_gt_esmatrix_frontview_backview_synthetic_magsac',
        # 'essential_matrix/config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_ransac',
        # 'essential_matrix/config_deep_with_flow_gt_esmatrix_frontview_backview_synthetic_ransac',
        'essential_matrix/config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_pygrancsac',
        'essential_matrix/config_deep_with_flow_gt_esmatrix_frontview_backview_synthetic_pygransac',
    ]

    sequences = {
        Datasets.SyntheticObjects: [
            'Textured_Sphere_5_y',
            'Textured_Sphere_5_z',
            'Textured_Sphere_5_x',
            # 'Textured_Sphere_10_y',
            'Translating_Textured_Sphere',
            # 'Textured_Sphere_5_xy',
            'Rotating_Translating_Textured_Sphere_5_y',
            # 'Rotating_Translating_Textured_Sphere_5_xy',
            # 'Rotating_Contra_Translating_Textured_Sphere_5_y',
            # 'Rotating_Contra_Translating_Textured_Sphere_5_xy',
            # '8_Colored_Sphere_5_x',
            # '6_Colored_Cube_5_z'
        ],
        Datasets.GoogleScannedObjects: [
            'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count',
            # 'Twinlab_Nitric_Fuel',
            'Squirrel',
            # 'STACKING_BEAR',
            'Schleich_Allosaurus',
            #     'Threshold_Ramekin_White_Porcelain',
            #     'Tag_Dishtowel_Green'
        ],
        # Datasets.photo360: [
        #     "09"
        # ]
    }

    # Set batch length
    batch_length = 1

    for configuration in configurations:

        output_folder_root = Path("/mnt/personal/jelint19/results/FlowTracker/")
        output_folder = output_folder_root / configuration
        output_folder = create_unused_folder(output_folder)

        for dataset in sequences:
            seq_index = 0
            while seq_index < len(sequences[dataset]):
                batch_seqs = sequences[dataset][seq_index:seq_index + batch_length]
                run_batch(configuration, batch_seqs, dataset, output_folder)
                seq_index += batch_length


if __name__ == "__main__":
    main()
