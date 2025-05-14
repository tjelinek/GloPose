from typing import List


from data_providers.frame_provider import PrecomputedFrameProvider
from data_providers.metric3d import *
from tracker_config import TrackerConfig


@torch.no_grad()
def compute_missing_depths(base_bop_folder: Path, relevant_datasets: List[str]):
    # metric3d = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    metric3d = metric3d_vit_large(pretrain=True).cuda().eval()
    config = TrackerConfig()

    for dataset in base_bop_folder.iterdir():

        if dataset.name not in relevant_datasets:
            continue

        for split in dataset.iterdir():

            if not split.is_dir():
                continue

            all_items = [x for x in split.iterdir() if x.is_dir()]

            if len(all_items) == 0 or (all_items[0] / 'depth').exists():
                continue

            for scene in all_items:

                rgb_folder = scene / 'rgb'

                if not scene.is_dir() or not rgb_folder.exists():
                    continue

                new_depth_folder = scene / 'depth_depthanythingv2'

                all_images = sorted(rgb_folder.iterdir())
                frame_provider = PrecomputedFrameProvider(config, all_images)

                for i in range(frame_provider.sequence_length):
                    image = frame_provider.next_image(i)

                    image_path = frame_provider.images_paths[i]

                    depth = prepare_image_for_metric3d(image_path, metric3d)
                    breakpoint()
                    pred_depth, confidence, output_dict = metric3d.inference({'input': image})



if __name__ == '__main__':

    _relevant_datasets = ['handal', 'hope']
    _base_bop_folder = Path('/mnt/personal/jelint19/data/bop')
    compute_missing_depths(_base_bop_folder, _relevant_datasets)
