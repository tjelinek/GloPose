from pathlib import Path


def save_masks_for_dataset(test_folder_path: Path):

    pass


if __name__ == '__main__':

    base_path = Path('/mnt/personal/jelint19/data/')
    bop_path = base_path / 'bop'

    default_detections_folder_classic = bop_path / 'default_detections' / 'classic_bop23_model_based_unseen'
    default_detections_bop24 = bop_path / 'default_detections' / 'h3_bop24_model_based_unseen'

    folder_paths = [
        bop_path / 'lmo' / 'test',
        bop_path / 'tless' / 'test_primesense',
        bop_path / 'tudl' / 'test',
        bop_path / 'icbin' / 'test',
        bop_path / 'itodd' / 'test',
        bop_path / 'hb' / 'test_kinect',
        bop_path / 'hb' / 'test_primesense',
        bop_path / 'ycbv' / 'test',
        bop_path / 'handal' / 'test',
    ]

    for folder_path in folder_paths:
        save_masks_for_dataset(folder_path)