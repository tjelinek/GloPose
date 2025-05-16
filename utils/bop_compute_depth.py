from torchvision.utils import save_image
from tqdm import tqdm

from data_providers.frame_provider import PrecomputedFrameProvider
from data_providers.metric3d import *
from tracker_config import TrackerConfig
from utils.bop_challenge import get_pinhole_params


@torch.no_grad()
def compute_missing_depths_metric3d(base_bop_folder: Path, relevant_datasets: List[str], batch_size: int = 8):
    metric3d = metric3d_vit_large(pretrain=True).cuda().eval()
    config = TrackerConfig()

    datasets = [d for d in base_bop_folder.iterdir() if d.name in relevant_datasets]
    for dataset in tqdm(datasets, desc='Datasets'):
        for split in tqdm(list(dataset.iterdir()), desc=f'Splits ({dataset.name})', leave=False):
            if not split.is_dir() and 'train' not in str(split.name):
                continue

            all_items = [x for x in split.iterdir() if x.is_dir()]
            if len(all_items) == 0 or (all_items[0] / 'depth').exists():
                continue

            for scene in tqdm(all_items, desc=f'Scenes ({split.name})', leave=False):
                print(f'Processing: Dataset={dataset.name}, Split={split.name}, Scene={scene.name}')
                rgb_folder = scene / 'rgb'
                if not scene.is_dir() or not rgb_folder.exists():
                    continue

                json_file_path = scene / 'scene_camera.json'
                pinhole_params = get_pinhole_params(json_file_path)

                first_image_pinhole_params = pinhole_params[min(pinhole_params.keys())]
                last_pinhole_params = first_image_pinhole_params

                new_depth_folder = scene / 'depth_metric3d'
                new_depth_folder.mkdir(exist_ok=True)

                all_images = sorted(rgb_folder.iterdir())
                frame_provider = PrecomputedFrameProvider(config, all_images)

                batch_imgs = []
                batch_K = []
                batch_paths = []

                for i in tqdm(range(frame_provider.sequence_length), desc='Frames', leave=False):
                    pinhole_params_i = pinhole_params.get(i)
                    image_name = frame_provider.get_n_th_image_name(i).stem
                    depth_image_path = new_depth_folder / (image_name + '.png')

                    if depth_image_path.exists():
                        continue

                    if pinhole_params_i is None:
                        pinhole_params_i = last_pinhole_params
                    else:
                        last_pinhole_params = pinhole_params_i
                    cam_K = pinhole_params_i.intrinsics.squeeze()

                    image = frame_provider.next_image_255(i)

                    batch_imgs.append(image)
                    batch_K.append(cam_K)
                    batch_paths.append(depth_image_path)

                    if len(batch_imgs) == batch_size or i == frame_provider.sequence_length - 1:
                        imgs_tensor = torch.cat(batch_imgs, dim=0).cuda()
                        Ks_tensor = torch.stack(batch_K, dim=0).cuda()
                        pred_depths = infer_depth_using_metric3d(imgs_tensor, Ks_tensor, metric3d)  # (B, 1, H, W)

                        for d, path in zip(pred_depths, batch_paths):
                            save_image(d, str(path))

                        batch_imgs.clear()
                        batch_K.clear()
                        batch_paths.clear()


if __name__ == '__main__':

    _relevant_datasets = ['handal', 'hope']
    _base_bop_folder = Path('/mnt/personal/jelint19/data/bop')
    compute_missing_depths_metric3d(_base_bop_folder, _relevant_datasets)
