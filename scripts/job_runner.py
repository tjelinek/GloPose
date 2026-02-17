import os
import subprocess
from enum import Enum
from pathlib import Path

from tracker_config import TrackerConfig
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
        # 'sam2_eval',
        # 'passthroughs/every_32th_frame',
        # 'passthroughs/every_16th_frame',
        # 'passthroughs/every_8th_frame',
        # 'passthroughs/every_4th_frame',
        # 'passthroughs/every_2nd_frame',
        # 'matchability_thresholds/ufm_c005r075',
        # 'matchability_thresholds/ufm_c005r09',
        # 'matchability_thresholds/ufm_c005r095',
        # 'matchability_thresholds/ufm_c005r099',
        # 'matchability_thresholds/ufm_c01r075',
        # 'matchability_thresholds/ufm_c01r09',
        # 'matchability_thresholds/ufm_c01r095',
        # 'matchability_thresholds/ufm_c01r099',
        # 'matchability_thresholds/ufm_c025r075',
        # 'matchability_thresholds/ufm_c025r09',
        # 'matchability_thresholds/ufm_c025r095',
        # 'matchability_thresholds/ufm_c025r099',
        # 'matchability_thresholds/ufm_c05r05',
        # 'matchability_thresholds/ufm_c05r075',
        # 'matchability_thresholds/ufm_c05r09',
        # 'matchability_thresholds/ufm_c05r095',
        # 'matchability_thresholds/ufm_c05r099',
        # 'matchability_thresholds/ufm_c075r025',
        # 'matchability_thresholds/ufm_c075r05',
        # 'matchability_thresholds/ufm_c075r075',
        # 'matchability_thresholds/ufm_c075r09',
        # 'matchability_thresholds/ufm_c075r095',
        # 'matchability_thresholds/ufm_c075r099',
        # 'matchability_thresholds/ufm_c09r01',
        # 'matchability_thresholds/ufm_c09r025',
        # 'matchability_thresholds/ufm_c09r05',
        # 'matchability_thresholds/ufm_c09r075',
        # 'matchability_thresholds/ufm_c09r09',
        # 'matchability_thresholds/ufm_c09r095',
        # 'matchability_thresholds/ufm_c09r099',
        # 'matchability_thresholds/ufm_c095r01',
        # 'matchability_thresholds/ufm_c095r025',
        # 'matchability_thresholds/ufm_c095r05_bbg',
        # 'matchability_thresholds/ufm_c095r05_bbg_dense',
        # 'matchability_thresholds/ufm_c095r05_viewgraph_from_matching',
        # 'matchability_thresholds/ufm_c095r075',
        # 'matchability_thresholds/ufm_c095r075_bbg',
        # 'matchability_thresholds/ufm_c0975r075_bbg_dense',
        # 'matchability_thresholds/ufm_c095r075_viewgraph_from_matching',
        # 'matchability_thresholds/ufm_c095r09',
        # 'matchability_thresholds/ufm_c095r095',
        # 'matchability_thresholds/ufm_c095r099',
        # 'matchability_thresholds/ufm_c0975r01',
        # 'matchability_thresholds/ufm_c0975r025',
        'matchability_thresholds/ufm_c0975r05',
        'matchability_thresholds/ufm_c0975r05_bbg',
        'matchability_thresholds/ufm_c0975r05_bbg_dense',
        'matchability_thresholds/ufm_c0975r05_dense',
        # 'matchability_thresholds/ufm_c0975r075',
        # 'matchability_thresholds/ufm_c0975r075_viewgraph_from_matching',
        # 'matchability_thresholds/ufm_c095r075_new',
        # 'matchability_thresholds/ufm_c0975r09',
        # 'matchability_thresholds/ufm_c0975r095',
        # 'matchability_thresholds/ufm_c0975r099',
        # 'matchability_thresholds/ufm_c099r01',
        # 'matchability_thresholds/ufm_c099r025',
        # 'matchability_thresholds/ufm_c099r05',
        # 'matchability_thresholds/ufm_c099r075',
        # 'matchability_thresholds/ufm_c099r09',
        # 'matchability_thresholds/ufm_c099r095',
        # 'matchability_thresholds/ufm_c099r099',
    ]

    data_folder = TrackerConfig().default_data_folder
    bop_path = data_folder / 'bop'

    handal_train, handal_test = get_handal_sequences(data_folder / 'HANDAL')
    ho3d_train, ho3d_eval = get_ho3d_sequences(data_folder / 'HO3D')
    handal_dynamic, handal_up, handal_down, handal_both = get_bop_onboarding_sequences(bop_path, 'handal')
    hope_dynamic, hope_up, hope_down, hope_both = get_bop_onboarding_sequences(bop_path, 'hope')

    sequences = {
        Datasets.SyntheticObjects: [],
        Datasets.GoogleScannedObjects: get_google_scanned_objects_sequences(
            data_folder / 'GoogleScannedObjects' / 'models'),
        Datasets.HO3D_eval: ho3d_eval,
        Datasets.HO3D_train: ho3d_train,
        Datasets.NAVI: get_navi_sequences(data_folder / 'NAVI' / 'navi_v1.5'),
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
        Datasets.BEHAVE: get_behave_sequences(data_folder / 'BEHAVE' / 'train'),
        Datasets.TUM_RGBD: get_tum_rgbd_sequences(data_folder / 'SLAM' / 'tum_rgbd'),
    }

    # Set batch length
    batch_length = 1

    for configuration in configurations:

        output_folder_root = Path("/mnt/personal/jelint19/results/FlowTracker/")
        output_folder = output_folder_root/configuration
        output_folder = create_unused_folder(output_folder)

        for dataset in sequences:
            seq_index = 0
            while seq_index < len(sequences[dataset]):
                batch_seqs = sequences[dataset][seq_index:seq_index + batch_length]
                run_batch(configuration, batch_seqs, dataset, output_folder)
                seq_index += batch_length


if __name__ == "__main__":
    main()
