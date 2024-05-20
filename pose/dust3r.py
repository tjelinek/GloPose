import sys
from typing import List, Tuple

import torch
import numpy as np
import PIL
from PIL import Image

sys.path.append('repositories/dust3r')

from repositories.dust3r.dust3r.inference import inference
from repositories.dust3r.dust3r.model import AsymmetricCroCo3DStereo
from repositories.dust3r.dust3r.utils.image import load_images, ImgNorm, _resize_pil_image
from repositories.dust3r.dust3r.image_pairs import make_pairs
from repositories.dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from repositories.dust3r.dust3r.utils.geometry import find_reciprocal_matches, xy_grid


def tensors_for_dust3r(image_tensors: List[torch.Tensor], size: int, square_ok: bool = False, verbose: bool = True):
    """
    Process a list of image tensors to prepare them for DUSt3R.

    Args:
    - image_tensors: List of image tensors (PyTorch).
    - size: Target size for resizing.
    - square_ok: Whether square images are acceptable without further cropping.
    - verbose: Whether to print verbose output.

    Returns:
    - List of dictionaries with processed image data and metadata.
    """

    imgs = []
    for idx, tensor in enumerate(image_tensors):
        img = PIL.Image.fromarray(tensor.mul(255).byte().permute(1, 2, 0).numpy(force=True))
        W1, H1 = img.size
        if size == 224:
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not square_ok and W == H:
                halfh = 3 * halfw // 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - processing tensor {idx} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=idx, instance=str(idx)))
    assert imgs, 'no images found in the provided tensors'
    if verbose:
        print(f' (Processed {len(imgs)} images)')
    return imgs


# model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
# you can put the path to a local checkpoint in model_name if needed
# model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to('cuda')

def get_matches_using_dust3r(imgs: List[torch.Tensor], size) -> Tuple[torch.Tensor, torch.Tensor]:
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = tensors_for_dust3r(imgs, 512)
    print(images[0]['img'].shape)

    images = load_images(['repositories/dust3r/croco/assets/Chateau1.png',
                          'repositories/dust3r/croco/assets/Chateau2.png'], size=512)
    # images = load_images([
    #     '/mnt/personal/jelint19/results/FlowTracker/GoogleScannedObjects/Squirrel/gt_imgs/gt_img_0_1.png',
    #     '/mnt/personal/jelint19/results/FlowTracker/GoogleScannedObjects/Squirrel/gt_imgs/gt_img_0_10.png',
    # ], size=512)
    print(images[0]['img'].shape)
    # breakpoint()
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # visualize reconstruction
    # scene.show()

    # find 2D-2D matches between the two images
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    matches_im0_torch = torch.from_numpy(matches_im0).cuda()
    matches_im1_torch = torch.from_numpy(matches_im1).cuda()

    # visualize a few matches
    # n_viz = 10
    # match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    #
    # H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    # img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img = np.concatenate((img0, img1), axis=1)
    # pl.figure()
    # pl.imshow(img)
    # cmap = pl.get_cmap('jet')
    # for i in range(n_viz):
    #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    # pl.show(block=True)

    return matches_im0_torch, matches_im1_torch
