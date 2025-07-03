import os
import subprocess
from enum import Enum
from pathlib import Path


class Datasets(Enum):
    SyntheticObjects = "SyntheticObjects"
    GoogleScannedObjects = "GoogleScannedObjects"
    HO3D = "HO3D"
    HANDAL = "HANDAL"
    HANDAL_ONBOARDING = "HANDAL_ONBOARDING"
    BEHAVE = "BEHAVE"
    TUM_RGBD = "TUM_RGBD"


runners = {
    Datasets.SyntheticObjects: "run_SyntheticObjects.py",
    Datasets.GoogleScannedObjects: "run_GoogleScannedObjects.py",
    Datasets.HO3D: "run_HO3D.py",
    Datasets.HANDAL: "run_HANDAL.py",
    Datasets.HANDAL_ONBOARDING: "run_HANDAL_onboarding.py",
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
    # if not os.path.exists(output_folder) or True:
    #     os.makedirs(output_folder, exist_ok=True)
    #     return output_folder
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
        'base_config',
        # 'passthrough',
        # 'sam2_eval'
        'matchability_thresholds/ufm_c005r01',
        'matchability_thresholds/ufm_c005r025',
        'matchability_thresholds/ufm_c005r05',
        'matchability_thresholds/ufm_c005r075',
        'matchability_thresholds/ufm_c005r09',
        'matchability_thresholds/ufm_c01r01',
        'matchability_thresholds/ufm_c01r025',
        'matchability_thresholds/ufm_c01r05',
        'matchability_thresholds/ufm_c01r075',
        'matchability_thresholds/ufm_c01r09',
        'matchability_thresholds/ufm_c025r01',
        'matchability_thresholds/ufm_c025r025',
        'matchability_thresholds/ufm_c025r05',
        'matchability_thresholds/ufm_c025r075',
        'matchability_thresholds/ufm_c025r09',
        'matchability_thresholds/ufm_c05r01',
        'matchability_thresholds/ufm_c05r025',
        'matchability_thresholds/ufm_c05r05',
        'matchability_thresholds/ufm_c05r075',
        'matchability_thresholds/ufm_c05r09',
        'matchability_thresholds/ufm_c075r01',
        'matchability_thresholds/ufm_c075r025',
        'matchability_thresholds/ufm_c075r05',
        'matchability_thresholds/ufm_c075r075',
        'matchability_thresholds/ufm_c075r09',
        'matchability_thresholds/ufm_c09r01',
        'matchability_thresholds/ufm_c09r025',
        'matchability_thresholds/ufm_c09r05',
        'matchability_thresholds/ufm_c09r075',
        'matchability_thresholds/ufm_c09r09',
        'matchability_thresholds/ufm_c095r01',
        'matchability_thresholds/ufm_c095r025',
        'matchability_thresholds/ufm_c095r05',
        'matchability_thresholds/ufm_c095r075',
        'matchability_thresholds/ufm_c095r09',
    ]

    sequences = {
        Datasets.SyntheticObjects: [
            'Textured_Sphere_5_y',
            # 'Textured_Sphere_5_x',
            # 'Textured_Sphere_5_z',
            # 'Textured_Cube_5_y',
            # 'Textured_Sphere_5_z',
            # 'Textured_Sphere_5_x',
            # 'Textured_Sphere_10_y',
            # 'Translating_Textured_Sphere',
            # 'Textured_Sphere_5_xy',
            # 'Rotating_Translating_Textured_Sphere_5_y',
            # 'Rotating_Translating_Textured_Sphere_5_xy',
            # 'Rotating_Contra_Translating_Textured_Sphere_5_y',
            # 'Rotating_Contra_Translating_Textured_Sphere_5_xy',
            # '8_Colored_Sphere_5_x',
            # '6_Colored_Cube_5_z'
        ],
        Datasets.GoogleScannedObjects: [
            # 'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count',
            # 'Twinlab_Nitric_Fuel',
            'Squirrel',
            # 'STACKING_BEAR',
            # 'Schleich_Allosaurus',
            'Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total',
            # 'SCHOOL_BUS',
            # 'Sootheze_Cold_Therapy_Elephant',
            # 'TOP_TEN_HI',
            'Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure',
        ],
        Datasets.HO3D: [
            'ABF10', 'BB10', 'GPMF10', 'GSF10', 'MC1', 'MDF10', #'ND2', 'ShSu12', 'SiBF12', 'SM3', 'SMu41',
            # 'ABF11', 'BB11', 'GPMF11', 'GSF11', 'MC2', 'MDF11', 'SB10', 'ShSu13', 'SiBF13', 'SM4', 'SMu42',
            # 'ABF12', 'BB12', 'GPMF12', 'GSF12', 'MC4', 'MDF12', 'SB12', 'ShSu14', 'SiBF14', 'SM5', 'SS1',
            # 'ABF13', 'BB13', 'GPMF13', 'GSF13', 'MC5', 'MDF13', 'SB14', 'SiBF10', 'SiS1', 'SMu1', 'SS2',
            # 'ABF14', #'BB14', 'GPMF14', #'GSF14', 'MC6', 'MDF14', 'ShSu10', 'SiBF11', 'SM2', 'SMu40', 'SS3',
        ],
        Datasets.HANDAL: [
            '000001_000000', '000001_000001', '000001_000002', '000001_000003', '000001_000004',
            '000002_000000', '000002_000001', '000002_000002', '000002_000003', '000002_000004', '000002_000005',
            '000003_000000', '000003_000001', '000003_000002', '000003_000003', '000003_000004',
            '000004_000000', '000004_000001', '000004_000002', '000004_000003', '000004_000004', '000004_000005',
            '000005_000000', '000005_000001', '000005_000002', '000005_000003', '000005_000004',
            '000006_000000', '000006_000001', '000006_000002', '000006_000003', '000006_000004', '000006_000005',
            '000007_000000', '000007_000001', '000007_000002', '000007_000003', '000007_000004', '000007_000005',
            '000008_000000', '000008_000001', '000008_000002', '000008_000003', '000008_000004',
            '000009_000000', '000009_000001', '000009_000002', '000009_000003', '000009_000004',
            '000010_000000', '000010_000001', '000010_000002', '000010_000003', '000010_000004', '000010_000005',
        ],
        Datasets.HANDAL_ONBOARDING: [
            'obj_000001',
            'obj_000002',
            'obj_000003',
            'obj_000004',
            'obj_000005',
            'obj_000006',
            'obj_000007',
            'obj_000008',
            'obj_000009',
            'obj_000010',
            'obj_000011',
            'obj_000012',
            'obj_000013',
            'obj_000014',
            'obj_000015',
            'obj_000016',
            'obj_000017',
            'obj_000018',
            'obj_000019',
            'obj_000020',
            'obj_000021',
            'obj_000022',
            'obj_000023',
            'obj_000024',
            'obj_000025',
            'obj_000026',
            'obj_000027',
            'obj_000028',
            'obj_000029',
            'obj_000030',
            'obj_000031',
            'obj_000032',
            'obj_000033',
            'obj_000034',
            'obj_000035',
            'obj_000036',
            'obj_000037',
            'obj_000038',
            'obj_000039',
            'obj_000040',
        ],
        Datasets.BEHAVE: [
            # '225z4rz6dtrsezi34lsrcnukni',
            # '227ybq4jddcxeobo7njvjnkmgy',
            # '24bw7vtbjt3ony3cgvye2oyjgu',
            # '24n2fzuerdocahja7fxod3jzfe',
            # '25zqalav3mxmbuvwzrgdxvp6ne',
            # '26623u6vetquo3323cyano7xpu',
            # '27pfmpfuewryv7u2vqe56sbsua',
            # '2ayiktcgtfbj45woxvfv74plui',
            # '2b2o7cfrp6j5luxwixtq2syeoy',
            # '2csdgc36d5txks6kpssnrojmby',
        ],
        Datasets.TUM_RGBD: [
            # 'rgbd_dataset_freiburg1_360',
            # 'rgbd_dataset_freiburg1_desk',
            # 'rgbd_dataset_freiburg1_desk2',
            # 'rgbd_dataset_freiburg1_floor',
            # 'rgbd_dataset_freiburg1_plant',
            # 'rgbd_dataset_freiburg1_room',
            # 'rgbd_dataset_freiburg1_rpy',
            # 'rgbd_dataset_freiburg1_teddy',
            # 'rgbd_dataset_freiburg1_xyz',
            # 'rgbd_dataset_freiburg2_360_hemisphere',
            # 'rgbd_dataset_freiburg2_360_kidnap',
            # 'rgbd_dataset_freiburg2_coke',
            # 'rgbd_dataset_freiburg2_desk',
            # 'rgbd_dataset_freiburg2_dishes',
            # 'rgbd_dataset_freiburg2_flowerbouquet',
            # 'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
            # 'rgbd_dataset_freiburg2_large_no_loop',
            # 'rgbd_dataset_freiburg2_large_with_loop',
            # 'rgbd_dataset_freiburg2_metallic_sphere',
            # 'rgbd_dataset_freiburg2_metallic_sphere2',
            # 'rgbd_dataset_freiburg2_pioneer_360',
            # 'rgbd_dataset_freiburg2_pioneer_slam',
            # 'rgbd_dataset_freiburg2_pioneer_slam2',
            # 'rgbd_dataset_freiburg2_pioneer_slam3',
            # 'rgbd_dataset_freiburg2_rpy',
            # 'rgbd_dataset_freiburg2_xyz',
            # 'rgbd_dataset_freiburg3_cabinet',
            # 'rgbd_dataset_freiburg3_large_cabinet',
            # 'rgbd_dataset_freiburg3_long_office_household',
            # 'rgbd_dataset_freiburg3_sitting_halfsphere',
            # 'rgbd_dataset_freiburg3_sitting_rpy',
            # 'rgbd_dataset_freiburg3_sitting_static',
            # 'rgbd_dataset_freiburg3_sitting_xyz',
            # 'rgbd_dataset_freiburg3_teddy',
            # 'rgbd_dataset_freiburg3_walking_halfsphere',
            # 'rgbd_dataset_freiburg3_walking_rpy',
            # 'rgbd_dataset_freiburg3_walking_static',
            # 'rgbd_dataset_freiburg3_walking_xyz',
        ],
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
