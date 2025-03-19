import os
import subprocess
from enum import Enum
from pathlib import Path


class Datasets(Enum):
    SyntheticObjects = "SyntheticObjects"
    GoogleScannedObjects = "GoogleScannedObjects"
    HO3D = "HO3D"
    HANDAL = "HANDAL"
    HANDAL_ONBOARDING = "HANDAL"
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
    if os.path.basename(os.getcwd()) == "scripts":
        subprocess.run(["sbatch", "job.batch"] + args)
    else:
        subprocess.run(["sbatch", "scripts/job.batch"] + args)


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
        'base_config'
        # 'glotracker/roma_thresholds/glotracker_roma',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_75_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_75_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_75_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_75_m2500',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_95_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_95_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_95_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_95_m2500',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_50_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_50_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_50_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c95_fg_50_m2500',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_75_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_75_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_75_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_75_m2500',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_95_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_95_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_95_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_95_m2500',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_50_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_50_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_50_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c90_fg_50_m2500',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_75_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_75_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_75_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_75_m2500',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_95_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_95_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_95_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_95_m2500',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_50_m200',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_50_m500',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_50_m1000',
        # 'glotracker/roma_thresholds/glotracker_roma_c50_fg_50_m2500',
        # 'glotracker/sift_thresholds/glotracker_sift_min50_good200',
        # 'glotracker/sift_thresholds/glotracker_sift_min50_good400',
        # 'glotracker/sift_thresholds/glotracker_sift_min50_good800',
        # 'glotracker/sift_thresholds/glotracker_sift_min100_good200',
        # 'glotracker/sift_thresholds/glotracker_sift_min100_good400',
        # 'glotracker/sift_thresholds/glotracker_sift_min100_good800',
        # 'glotracker/sift_thresholds/glotracker_sift_min200_good400',
        # 'glotracker/sift_thresholds/glotracker_sift_min200_good800',
        # 'glotracker/sift_thresholds/glotracker_sift_min400_good800',

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
            '000001',
            '000002',
            '000003',
            '000004',
            '000005',
        ],
        Datasets.HANDAL_ONBOARDING: [
            'obj_000001',
            'obj_000010',
            'obj_000020',
            'obj_000030',
            'obj_000040',
        ],
        Datasets.BEHAVE: [
            '225z4rz6dtrsezi34lsrcnukni',
            '227ybq4jddcxeobo7njvjnkmgy',
            '24bw7vtbjt3ony3cgvye2oyjgu',
            '24n2fzuerdocahja7fxod3jzfe',
            '25zqalav3mxmbuvwzrgdxvp6ne',
            # '26623u6vetquo3323cyano7xpu',
            # '27pfmpfuewryv7u2vqe56sbsua',
            # '2ayiktcgtfbj45woxvfv74plui',
            # '2b2o7cfrp6j5luxwixtq2syeoy',
            # '2csdgc36d5txks6kpssnrojmby',
        ],
        Datasets.TUM_RGBD: [
            'rgbd_dataset_freiburg1_360',
            # 'rgbd_dataset_freiburg1_desk',
            # 'rgbd_dataset_freiburg1_desk2',
            # 'rgbd_dataset_freiburg1_floor',
            # 'rgbd_dataset_freiburg1_plant',
            # 'rgbd_dataset_freiburg1_room',
            # 'rgbd_dataset_freiburg1_rpy',
            'rgbd_dataset_freiburg1_teddy',
            # 'rgbd_dataset_freiburg1_xyz',
            # 'rgbd_dataset_freiburg2_360_hemisphere',
            # 'rgbd_dataset_freiburg2_360_kidnap',
            # 'rgbd_dataset_freiburg2_coke',
            'rgbd_dataset_freiburg2_desk',
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
