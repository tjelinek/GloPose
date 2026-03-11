import os
import subprocess
from enum import Enum
from pathlib import Path

from configs.glopose_config import GloPoseConfig
from utils.dataset_sequences import (
    get_handal_sequences, get_navi_sequences, get_ho3d_sequences,
    get_tum_rgbd_sequences, get_behave_sequences, get_google_scanned_objects_sequences,
    get_bop_val_sequences, get_bop_onboarding_sequences, get_bop_classic_sequences
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
    BOP_HANDAL_ONBOARDING_DYNAMIC = "BOP_HANDAL_ONBOARDING_DYNAMIC"
    HOPE_ONBOARDING_STATIC = "HOPE_ONBOARDING_STATIC"
    HOPE_ONBOARDING_DYNAMIC = "HOPE_ONBOARDING_DYNAMIC"
    BOP_CLASSIC_ONBOARDING_SEQUENCES = "BOP_CLASSIC_ONBOARDING_SEQUENCES"
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
    Datasets.BOP_HANDAL_ONBOARDING_DYNAMIC: "run_bop_HANDAL_onboarding.py",
    Datasets.HOPE_ONBOARDING_STATIC: "run_HOPE.py",
    Datasets.HOPE_ONBOARDING_DYNAMIC: "run_HOPE.py",
    Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES: "run_BOP_classic_onboarding.py",
    Datasets.BEHAVE: "run_BEHAVE.py",
    Datasets.TUM_RGBD: "run_TUM_RGBD.py",
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


def run_job_array(configuration_name: str, all_sequences: list, dataset: Datasets, output_folder: Path) -> None:
    """Submit all sequences for a config as a single SLURM job array."""
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
    array_spec = f'0-{n - 1}'

    batch_script = 'job_array.batch' if os.path.basename(os.getcwd()) == 'scripts' else 'scripts/job_array.batch'

    cmd = [
        'sbatch', f'--array={array_spec}',
        batch_script,
        '--config', str(configuration_path),
        '--dataset-runner', runners[dataset],
        '--experiment', experiment,
        '--output-folder', str(output_folder),
        '--sequence-list', str(seq_list_path),
    ]

    print('----------------------------------------')
    print(f'Submitting job array ({n} tasks) for {configuration_name} on {dataset.value}')
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


def main():
    configurations = [
        # 'base_config',
        # 'passthrough',
        # 'onboarding/passthroughs/every_32th_frame',
        # 'onboarding/passthroughs/every_16th_frame',
        # 'onboarding/passthroughs/every_8th_frame',
        # 'onboarding/passthroughs/every_4th_frame',
        # 'onboarding/passthroughs/every_2nd_frame',
        # 'onboarding/ufm_c075r05',
        # 'onboarding/ufm_c075r075',
        # 'onboarding/ufm_c075r09',
        # 'onboarding/ufm_c075r095',
        # 'onboarding/ufm_c09r075',
        # 'onboarding/ufm_c09r09',
        # 'onboarding/ufm_c09r095',
        # 'onboarding/ufm_c095r05_bbg',
        # 'onboarding/ufm_c095r05_bbg_dense',
        # 'onboarding/ufm_c095r05_viewgraph_from_matching',
        # 'onboarding/ufm_c095r075',
        # 'onboarding/ufm_c095r075_bbg',
        # 'onboarding/ufm_c0975r075_bbg_dense',
        # 'onboarding/ufm_c095r075_viewgraph_from_matching',
        # 'onboarding/ufm_c095r09',
        # 'onboarding/ufm_c095r095',
        'onboarding/ufm_c0975r05',
        'onboarding/ufm_c0975r05_fixed_threshold',
        'onboarding/ufm_c0975r05_bbg',
        'onboarding/ufm_c0975r05_bbg_dense',
        'onboarding/ufm_c0975r05_dense',
        # 'onboarding/ufm_c0975r075',
        # 'onboarding/ufm_c0975r075_viewgraph_from_matching',
        # 'onboarding/ufm_c095r075_new',
        # 'onboarding/ufm_c0975r09',
        # 'onboarding/ufm_c0975r095',
        # --- P3.4: External reconstruction methods ---
        # VGGT — adaptive keyframes
        # 'reconstruction/vggt',                    # white bg (convention)
        # 'reconstruction/vggt_black_bg',
        # 'reconstruction/vggt_original_bg',
        # VGGT — every 8th frame
        # 'reconstruction/vggt_every_8th',          # white bg (convention)
        # 'reconstruction/vggt_every_8th_original_bg',
        # Mast3r — adaptive keyframes
        # 'reconstruction/mast3r',                  # black bg (convention)
        # 'reconstruction/mast3r_white_bg',
        # 'reconstruction/mast3r_original_bg',
        # Mast3r — every 8th frame
        # 'reconstruction/mast3r_every_8th',        # black bg (convention)
        # 'reconstruction/mast3r_every_8th_original_bg',
    ]

    cfg = GloPoseConfig()
    bop_path = cfg.paths.bop_data_folder

    handal_train, handal_test = get_handal_sequences(cfg.paths.handal_data_folder / 'HANDAL')
    ho3d_train, ho3d_eval = get_ho3d_sequences(cfg.paths.ho3d_data_folder)
    handal_dynamic, handal_up, handal_down, handal_both = get_bop_onboarding_sequences(bop_path, 'handal')
    hope_dynamic, hope_up, hope_down, hope_both = get_bop_onboarding_sequences(bop_path, 'hope')

    sequences = {
        Datasets.SyntheticObjects: [],
        Datasets.GoogleScannedObjects: get_google_scanned_objects_sequences(
            cfg.paths.google_scanned_objects_data_folder / 'models'),
        Datasets.HO3D_eval: ho3d_eval,
        Datasets.HO3D_train: ho3d_train,
        Datasets.NAVI: get_navi_sequences(cfg.paths.navi_data_folder),
        Datasets.HANDAL: handal_train + handal_test,
        Datasets.BOP_HANDAL: get_bop_val_sequences(bop_path / 'handal' / 'val'),
        Datasets.BOP_HANDAL_ONBOARDING_STATIC: handal_both,
        Datasets.BOP_HANDAL_ONBOARDING_DYNAMIC: handal_dynamic,
        Datasets.HOPE_ONBOARDING_STATIC: hope_both,
        Datasets.HOPE_ONBOARDING_DYNAMIC: hope_dynamic,
        Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES: (
            get_bop_classic_sequences(bop_path, 'tless', 'train_primesense') +
            get_bop_classic_sequences(bop_path, 'lmo', 'train') +
            get_bop_classic_sequences(bop_path, 'icbin', 'train')
        ),
        Datasets.BEHAVE: get_behave_sequences(cfg.paths.behave_data_folder / 'train'),
        Datasets.TUM_RGBD: get_tum_rgbd_sequences(cfg.paths.tum_rgbd_data_folder),
    }

    output_folder_root = Path("/mnt/personal/jelint19/results/FlowTracker/")

    for configuration in configurations:
        output_folder = create_unused_folder(output_folder_root / configuration)

        for dataset in sequences:
            run_job_array(configuration, sequences[dataset], dataset, output_folder)


if __name__ == "__main__":
    main()
