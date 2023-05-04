import os
import torch

from benchmark.benchmark_loader import *
from benchmark.loaders_helpers import *
import argparse

import sys

sys.path.insert(0, '../track6d/')
from shapefromblur import *
from utils import load_config

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tbd_path", default='/cluster/home/denysr/scratch/dataset/TbD', required=False)
	parser.add_argument("--tbd3d_path", default='/cluster/home/denysr/scratch/dataset/TbD-3D', required=False)
	parser.add_argument("--falling_path", default='/cluster/home/denysr/scratch/dataset/falling_objects', required=False)
	parser.add_argument("--verbose", default=True)
	parser.add_argument("--visualization_path", default='/cluster/home/denysr/scratch/eval', required=False)
	parser.add_argument("--save_visualization", default=False, required=False)
	parser.add_argument("--dataset", required=True)
	parser.add_argument("--config", default=None, required=False)
	return parser.parse_args()

def main():
	args = parse_args()
	g_resolution_x = int(640/2)
	g_resolution_y = int(480/2)

	config_path = args.config
	if config_path is None:
		if args.dataset == 'f':
			config_path = "../configs/config_falling.yaml"
		elif args.dataset == '3d':
			config_path = "../configs/config_tbd3d.yaml"
		elif args.dataset == 'tbd':
			config_path = "../configs/config_tbd.yaml"
	config = load_config(config_path)

	print(config)
	print(args)
	sfb = ShapeFromBlur(config=config)

	def deblur_sfb(I,B,bbox_tight,nsplits,radius,obj_dim):
		best_model = sfb.apply(I, B, None, bbox_tight, nsplits)
		est_hs_crop = rev_crop_resize(best_model["renders"][0,0].transpose(2,3,1,0), sfb.bbox, np.zeros((I.shape[0],I.shape[1],4)))
		est_hs = rgba2hs(est_hs_crop, B)
		est_traj = renders2traj(torch.from_numpy(best_model["renders"][0]), 'cpu')[0].T
		est_traj = rev_crop_resize_traj(est_traj, sfb.bbox, (g_resolution_x, g_resolution_y))
		return est_hs, est_traj, best_model["value"]

	args.add_traj = False	
	args.method_name = 'SfB'
	run_benchmark(args, deblur_sfb)


if __name__ == "__main__":
    main()

# bsub -G ls_polle -R "rusage[mem=20000, ngpus_excl_p=1, scratch=10000]" -R "select[gpu_mtotal0>=9000]" -W 24:00 python example_sfb.py --dataset f
# bsub -G ls_polle -R "rusage[mem=20000, ngpus_excl_p=1, scratch=10000]" -R "select[gpu_mtotal0>=9000]" -W 120:00 python example_sfb.py --dataset tbd