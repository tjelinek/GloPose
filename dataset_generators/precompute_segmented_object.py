import shutil
from pathlib import Path

from torchvision.utils import save_image
from tqdm import tqdm

from data_providers.frame_provider import FrameProviderAll
from tracker_config import TrackerConfig
from utils.bop_challenge import load_gt_images, load_gt_segmentations
from utils.data_utils import get_initial_image_and_segment
from utils.image_utils import overlay_mask_torch


def get_segmentation_provider(segmentation_type, initial_segmentation, initial_image, images_paths, segmentation_paths,
                              cache_folder_SAM2) -> FrameProviderAll:
    config = TrackerConfig()

    config.segmentation_provider = segmentation_type
    config.frame_provider = 'precomputed'
    config.write_folder = Path('/mnt/personal/jelint19/tmp/sam2_segs')

    tracker = FrameProviderAll(config,
                               initial_segmentation=initial_segmentation,
                               initial_image=initial_image, images_paths=images_paths,
                               segmentation_paths=segmentation_paths,
                               sam2_cache_folder=cache_folder_SAM2)

    return tracker


def process_bop_sequences(dataset, splits):
    bop_folder = Path('/mnt/personal/jelint19/data/bop').expanduser()
    dataset_path = bop_folder / dataset

    for split in tqdm(splits, desc='Split'):
        split_path = dataset_path / split

        for sequence in tqdm(split_path.iterdir(), desc='Sequence'):

            if not sequence.is_dir():
                continue

            images = load_gt_images(sequence / 'rgb')
            segs = load_gt_segmentations(sequence / 'mask_visib')

            first_image, first_segmentation = get_initial_image_and_segment(
                images,
                segs,
                segmentation_channel=0
            )

            process_sequence(first_image, first_segmentation, images, segs, sequence)


def process_ho3d_sequences(splits):
    ho3d_folder = Path('/mnt/personal/jelint19/data/HO3D').expanduser()

    for split in tqdm(splits, desc='Split'):
        split_path = ho3d_folder / split

        for sequence in tqdm(split_path.iterdir(), desc='Sequence'):

            if not sequence.is_dir():
                continue

            image_folder = sequence / 'rgb'
            segmentation_folder = sequence / 'seg'

            images = [file for file in sorted(image_folder.iterdir()) if file.is_file()]
            segs = [file for file in sorted(segmentation_folder.iterdir()) if file.is_file()]

            first_image, first_segmentation = get_initial_image_and_segment(
                images,
                segs,
                segmentation_channel=1
            )

            process_sequence(first_image, first_segmentation, images, segs, sequence)


def process_sequence(first_image, first_segmentation, images, segs, sequence):
    SAM2_cache_folder = Path('/mnt/personal/jelint19/cache/generating_segmentations_cache')
    tracker_provider_precomputed = get_segmentation_provider('precomputed', first_segmentation,
                                                             first_image, images, segs, SAM2_cache_folder)
    tracker_provider_sam2 = get_segmentation_provider('SAM2', first_segmentation,
                                                      first_image, images, segs, SAM2_cache_folder)
    sequence_length = len(images)

    gt_seg_path = sequence / 'rgb_segmented_gt'
    sam2_seg_path = sequence / 'rgb_segmented_sam2'

    gt_seg_path.mkdir(parents=True, exist_ok=True)
    sam2_seg_path.mkdir(parents=True, exist_ok=True)

    for frame_idx in tqdm(range(sequence_length), desc='Frame'):
        image_name = tracker_provider_precomputed.frame_provider.get_n_th_image_name(frame_idx).stem
        image = tracker_provider_precomputed.frame_provider.next_image_255(frame_idx)
        seg_gt = tracker_provider_precomputed.segmentation_provider.next_segmentation(frame_idx)
        seg_sam2 = tracker_provider_sam2.segmentation_provider.next_segmentation(frame_idx, image)

        image_seg_gt = overlay_mask_torch(image, 1 - seg_gt, 1.0, (0, 0, 0))
        image_seg_sam2 = overlay_mask_torch(image, 1 - seg_sam2, 1.0, (0, 0, 0))

        image_seg_gt_path = gt_seg_path / f'{image_name}.jpg'
        image_seg_sa2_path = sam2_seg_path / f'{image_name}.jpg'

        image_seg_gt_norm = image_seg_gt.squeeze().float() / 255.0
        image_seg_sam2_norm = image_seg_gt.squeeze().float() / 255.0

        save_image(image_seg_gt_norm, str(image_seg_gt_path))
        save_image(image_seg_sam2_norm, str(image_seg_sa2_path))
    shutil.rmtree(SAM2_cache_folder)


if __name__ == '__main__':

    print('Processing HO3D')
    process_ho3d_sequences(['train', 'evaluation'])

    print('Processing BOP HANDAL')
    process_bop_sequences('handal', ['train', 'test', 'onboarding_static'])

    print('Processing BOP HOPEv2')
    process_bop_sequences('hope', ['train', 'test', 'onboarding_static', 'onboarding_dynamic'])
