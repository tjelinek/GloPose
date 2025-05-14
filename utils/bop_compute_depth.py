from typing import List

from torchvision.utils import save_image

from data_providers.frame_provider import PrecomputedFrameProvider
from data_providers.metric3d import *
from tracker_config import TrackerConfig
from utils.bop_challenge import get_pinhole_params


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

                json_file_path = scene / 'scene_camera.json'
                pinhole_params = get_pinhole_params(json_file_path)

                first_image_pinhole_params = pinhole_params[0]
                last_pinhole_params = first_image_pinhole_params

                new_depth_folder = scene / 'depth_metric3d'

                all_images = sorted(rgb_folder.iterdir())
                frame_provider = PrecomputedFrameProvider(config, all_images)

                for i in range(frame_provider.sequence_length):

                    pinhole_params_i = pinhole_params.get(i)
                    if pinhole_params_i is None:
                        pinhole_params_i = last_pinhole_params
                    else:
                        last_pinhole_params = pinhole_params_i
                    cam_K = pinhole_params_i.intrinsics.squeeze()

                    # image = frame_provider.next_image(i)

                    image_path = frame_provider.images_paths[i]
                    image_name = frame_provider.get_n_th_image_name(i).stem

                    depth = infer_depth_using_metric3d(image_path, metric3d, cam_K)

                    depth_image_path = new_depth_folder / (image_name + '.png')
                    save_image(depth, str(depth_image_path))


if __name__ == '__main__':

    _relevant_datasets = ['handal', 'hope']
    _base_bop_folder = Path('/mnt/personal/jelint19/data/bop')
    compute_missing_depths(_base_bop_folder, _relevant_datasets)
