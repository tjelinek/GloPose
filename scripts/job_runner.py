import argparse
import os
import random
import subprocess
from enum import Enum
from pathlib import Path

from configs.glopose_config import GloPoseConfig
from utils.dataset_sequences import (
    get_handal_sequences, get_navi_sequences, get_ho3d_sequences,
    get_tum_rgbd_sequences, get_behave_sequences, get_google_scanned_objects_sequences,
    get_bop_val_sequences, get_bop_onboarding_sequences, get_bop_classic_sequences,
    get_hot3d_onboarding_sequences
)


class Datasets(Enum):
    SyntheticObjects = "SyntheticObjects"
    GoogleScannedObjects = "GoogleScannedObjects"
    HO3D_eval = "HO3D_eval"
    HO3D_train = "HO3D_train"
    NAVI = "NAVI"
    HANDAL = "HANDAL"
    BOP_HANDAL = "BOP_HANDAL"
    BOP_HANDAL_ONBOARDING_STATIC = "BOP_HANDAL_ONBOARDING_STATIC"
    BOP_HANDAL_ONBOARDING_BOTH = "BOP_HANDAL_ONBOARDING_BOTH"
    BOP_HANDAL_ONBOARDING_DYNAMIC = "BOP_HANDAL_ONBOARDING_DYNAMIC"
    HOPE_ONBOARDING_STATIC = "HOPE_ONBOARDING_STATIC"
    HOPE_ONBOARDING_BOTH = "HOPE_ONBOARDING_BOTH"
    HOPE_ONBOARDING_DYNAMIC = "HOPE_ONBOARDING_DYNAMIC"
    BOP_CLASSIC_ONBOARDING_SEQUENCES = "BOP_CLASSIC_ONBOARDING_SEQUENCES"
    HOT3D_ARIA_ONBOARDING_STATIC = "HOT3D_ARIA_ONBOARDING_STATIC"
    HOT3D_ARIA_ONBOARDING_DYNAMIC = "HOT3D_ARIA_ONBOARDING_DYNAMIC"
    HOT3D_QUEST3_ONBOARDING_STATIC = "HOT3D_QUEST3_ONBOARDING_STATIC"
    HOT3D_QUEST3_ONBOARDING_DYNAMIC = "HOT3D_QUEST3_ONBOARDING_DYNAMIC"
    BEHAVE = "BEHAVE"
    TUM_RGBD = "TUM_RGBD"


runners = {
    Datasets.SyntheticObjects: "run_SyntheticObjects.py",
    Datasets.GoogleScannedObjects: "run_GoogleScannedObjects.py",
    Datasets.HO3D_eval: "run_HO3D.py",
    Datasets.HO3D_train: "run_HO3D.py",
    Datasets.NAVI: "run_NAVI.py",
    Datasets.HANDAL: "run_HANDAL.py",
    Datasets.BOP_HANDAL: "run_bop_HANDAL.py",
    Datasets.BOP_HANDAL_ONBOARDING_STATIC: "run_bop_HANDAL_onboarding.py",
    Datasets.BOP_HANDAL_ONBOARDING_BOTH: "run_bop_HANDAL_onboarding.py",
    Datasets.BOP_HANDAL_ONBOARDING_DYNAMIC: "run_bop_HANDAL_onboarding.py",
    Datasets.HOPE_ONBOARDING_STATIC: "run_HOPE.py",
    Datasets.HOPE_ONBOARDING_BOTH: "run_HOPE.py",
    Datasets.HOPE_ONBOARDING_DYNAMIC: "run_HOPE.py",
    Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES: "run_BOP_classic_onboarding.py",
    Datasets.HOT3D_ARIA_ONBOARDING_STATIC: "run_HOT3D.py",
    Datasets.HOT3D_ARIA_ONBOARDING_DYNAMIC: "run_HOT3D.py",
    Datasets.HOT3D_QUEST3_ONBOARDING_STATIC: "run_HOT3D.py",
    Datasets.HOT3D_QUEST3_ONBOARDING_DYNAMIC: "run_HOT3D.py",
    Datasets.BEHAVE: "run_BEHAVE.py",
    Datasets.TUM_RGBD: "run_TUM_RGBD.py",
}

# Extra CLI args passed to the runner script (e.g., --device quest3 for HOT3D Quest3)
runner_extra_args = {
    Datasets.HOT3D_QUEST3_ONBOARDING_STATIC: "--device quest3",
    Datasets.HOT3D_QUEST3_ONBOARDING_DYNAMIC: "--device quest3",
}


def run_batch(configuration_name: str, sequences, dataset: Datasets, output_folder: Path) -> None:
    configuration_path = Path('configs')/(configuration_name + '.py')

    args = []
    args.append("--config")
    args.append(str(configuration_path))
    args.append("--dataset-runner")
    args.append(runners[dataset])
    args.append("--experiment")
    args.append(str(Path(configuration_name).stem))
    args.append("--output-folder")
    args.append(str(output_folder))
    extra = runner_extra_args.get(dataset)
    if extra:
        args.extend(["--runner-extra-args", extra])
    args.append("--sequences")
    args.extend(sequences)

    # Echo the arguments
    print('----------------------------------------')
    print("Running sbatch job.batch with arguments:", args)
    if os.path.basename(os.getcwd()) == "scripts":
        subprocess.run(["sbatch", "job.batch"] + args)
    else:
        print(' '.join(["sbatch", "job.batch"] + args))
        subprocess.run(["sbatch", "scripts/job.batch"] + args)
    print('----------------------------------------')


def run_job_array(configuration_name: str, all_sequences: list, dataset: Datasets, output_folder: Path,
                   chunk_size: int = 2) -> None:
    """Submit all sequences for a config as a single SLURM job array.

    Each array task processes `chunk_size` sequences sequentially, reducing the total number of SLURM jobs.
    """
    if not all_sequences:
        return

    configuration_path = Path('configs') / (configuration_name + '.py')
    experiment = str(Path(configuration_name).stem)

    # Write sequence list to a file in the output folder
    os.makedirs(output_folder, exist_ok=True)
    seq_list_path = output_folder / f'{experiment}_{dataset.value}_sequences.txt'
    with open(seq_list_path, 'w') as f:
        for seq in all_sequences:
            f.write(seq + '\n')

    n = len(all_sequences)
    n_tasks = (n + chunk_size - 1) // chunk_size  # ceiling division
    array_spec = f'0-{n_tasks - 1}'

    batch_script = 'job_array.batch' if os.path.basename(os.getcwd()) == 'scripts' else 'scripts/job_array.batch'

    cmd = [
        'sbatch', f'--array={array_spec}',
        batch_script,
        '--config', str(configuration_path),
        '--dataset-runner', runners[dataset],
        '--experiment', experiment,
        '--output-folder', str(output_folder),
        '--sequence-list', str(seq_list_path),
        '--chunk-size', str(chunk_size),
    ]
    extra = runner_extra_args.get(dataset)
    if extra:
        cmd.extend(['--runner-extra-args', extra])

    print('----------------------------------------')
    print(f'Submitting job array ({n_tasks} tasks, {chunk_size} seq/task, {n} total) '
          f'for {configuration_name} on {dataset.value}')
    print(' '.join(cmd))
    subprocess.run(cmd)
    print('----------------------------------------')


def create_unused_folder(output_folder: Path):
    # i = 1
    # while os.path.exists(f"{output_folder}_{i}"):
    #     i += 1
    # final_output_folder = f"{output_folder}_{i}"
    # os.makedirs(final_output_folder, exist_ok=True)
    #
    # return final_output_folder

    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def get_configurations():
    return [
        # --- Baseline ---
        # 'onboarding/ufm_c0975r05',

        # --- P3.1: Frame selection ablation ---
        # 'onboarding/passthroughs/every_frame',
        # 'onboarding/passthroughs/every_2nd_frame',
        'onboarding/passthroughs/every_4th_frame',
        'onboarding/passthroughs/every_8th_frame',
        'onboarding/passthroughs/every_16th_frame',
        'onboarding/passthroughs/every_32th_frame',
        'onboarding/passthroughs/every_64th_frame',
        # 
        # # --- P3.2: ViewGraph density ablation ---
        # 'onboarding/ufm_c0975r05_dense',            # all-to-all edges
        # 
        # # --- P3.3: Matching method ablation ---
        # 'onboarding/roma_c0975r05',                  # RoMa matcher
        # 'onboarding/sift_matching',                   # SIFT + LightGlue
        # 
        # # --- P3.4: External reconstruction methods ---
        # # VGGT — adaptive keyframes
        # 'reconstruction/vggt',                        # white bg (convention)
        # 'reconstruction/vggt_black_bg',
        # 'reconstruction/vggt_original_bg',
        # # VGGT — every 8th frame
        # 'reconstruction/vggt_every_8th',              # white bg (convention)
        # 'reconstruction/vggt_every_8th_original_bg',
        # # Mast3r — adaptive keyframes
        # 'reconstruction/mast3r',                      # black bg (convention)
        # 'reconstruction/mast3r_white_bg',
        # 'reconstruction/mast3r_original_bg',
        # # Mast3r — every 8th frame
        # 'reconstruction/mast3r_every_8th',            # black bg (convention)
        # 'reconstruction/mast3r_every_8th_original_bg',
        # 
        # # --- P3.5: Background removal ablation ---
        # 'onboarding/ufm_c0975r05_bbg',               # black background
        # 
        # # --- P3.6: Track merging ablation ---
        # 'onboarding/ufm_c0975r05_no_track_merging',
        # 
        # # --- Additional ablations ---
        # # RANSAC-based frame adding
        # 'onboarding/ufm_ransac_pycolmap',
        # 'onboarding/ufm_ransac_magsac',
        # # Bundle adjustment after segmentation filtering
        # 'reconstruction/colmap_seg_filter',
        # 'reconstruction/colmap_seg_filter_black_bg',
        # # Otsu vs fixed certainty threshold
        # 'onboarding/ufm_c0975r05_fixed_threshold',
        # # Matchability-based reliability
        # 'onboarding/ufm_c0975r05_matchability',
    ]


def get_sequences():
    cfg = GloPoseConfig()
    bop_path = cfg.paths.bop_data_folder

    handal_train, handal_test = get_handal_sequences(cfg.paths.handal_data_folder / 'HANDAL')
    ho3d_train, ho3d_eval = get_ho3d_sequences(cfg.paths.ho3d_data_folder)
    handal_dynamic, handal_up, handal_down, handal_both = get_bop_onboarding_sequences(bop_path, 'handal')
    hope_dynamic, hope_up, hope_down, hope_both = get_bop_onboarding_sequences(bop_path, 'hope')
    hot3d_aria_dynamic, hot3d_aria_static = get_hot3d_onboarding_sequences(bop_path, device='aria')
    hot3d_quest3_dynamic, hot3d_quest3_static = get_hot3d_onboarding_sequences(bop_path, device='quest3')

    return {
        # --- BOP onboarding: static (up+down separate), both (up+down combined), dynamic ---
        Datasets.BOP_HANDAL_ONBOARDING_STATIC: handal_up + handal_down,
        Datasets.BOP_HANDAL_ONBOARDING_BOTH: handal_both,
        Datasets.BOP_HANDAL_ONBOARDING_DYNAMIC: handal_dynamic,
        Datasets.HOPE_ONBOARDING_STATIC: hope_up + hope_down,
        Datasets.HOPE_ONBOARDING_BOTH: hope_both,
        Datasets.HOPE_ONBOARDING_DYNAMIC: hope_dynamic,
        Datasets.HOT3D_ARIA_ONBOARDING_STATIC: hot3d_aria_static,
        # Datasets.HOT3D_ARIA_ONBOARDING_DYNAMIC: hot3d_aria_dynamic,  # TODO: needs depth-based alignment support
        Datasets.HOT3D_QUEST3_ONBOARDING_STATIC: hot3d_quest3_static,
        # Datasets.HOT3D_QUEST3_ONBOARDING_DYNAMIC: hot3d_quest3_dynamic,  # TODO: needs depth-based alignment support
        # --- BOP classic ---
        Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES: (
            get_bop_classic_sequences(bop_path, 'tless', 'train_primesense') +
            get_bop_classic_sequences(bop_path, 'lmo', 'train') +
            get_bop_classic_sequences(bop_path, 'icbin', 'train')
        ),
        # --- Other datasets ---
        Datasets.HO3D_train: ho3d_train,
        Datasets.NAVI: get_navi_sequences(cfg.paths.navi_data_folder),
        # --- Disabled ---
        # Datasets.SyntheticObjects: [],
        # Datasets.GoogleScannedObjects: get_google_scanned_objects_sequences(
        #     cfg.paths.google_scanned_objects_data_folder / 'models'),
        # Datasets.HO3D_eval: ho3d_eval,
        # Datasets.HANDAL: handal_train + handal_test,
        # Datasets.BOP_HANDAL: get_bop_val_sequences(bop_path / 'handal' / 'val'),
        # Datasets.BEHAVE: get_behave_sequences(cfg.paths.behave_data_folder / 'train'),
        # Datasets.TUM_RGBD: get_tum_rgbd_sequences(cfg.paths.tum_rgbd_data_folder),
    }


def get_results_root():
    return Path("/mnt/personal/jelint19/results/FlowTracker/")


def subsample_sequences(sequences: dict, max_per_dataset: int, seed: int = 42) -> dict:
    """Deterministically subsample up to max_per_dataset sequences from each dataset."""
    subsampled = {}
    for dataset, seqs in sequences.items():
        if len(seqs) <= max_per_dataset:
            subsampled[dataset] = seqs
        else:
            rng = random.Random(seed)
            subsampled[dataset] = sorted(rng.sample(seqs, max_per_dataset))
    return subsampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run on 20 random (but deterministic) sequences per dataset')
    args = parser.parse_args()

    configurations = get_configurations()
    sequences = get_sequences()
    output_folder_root = get_results_root()

    if args.quick:
        sequences = subsample_sequences(sequences, max_per_dataset=20)
        total = sum(len(s) for s in sequences.values())
        print(f"[--quick] Subsampled to {total} sequences total (max 20 per dataset)")

    for configuration in configurations:
        output_folder = create_unused_folder(output_folder_root / configuration)

        for dataset in sequences:
            run_job_array(configuration, sequences[dataset], dataset, output_folder)


if __name__ == "__main__":
    main()
