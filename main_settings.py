import os
from pathlib import Path

dataset_folder = '/mnt/lascar/rozumden/dataset/'
tmp_folder = '/home.stud/rozumden/tmp/'

if not os.path.exists(dataset_folder):
	dataset_folder = '/cluster/scratch/denysr/dataset/'
	tmp_folder = '/cluster/scratch/denysr/eval/'

# TODO: put your paths here:
if not os.path.exists(dataset_folder):
	dataset_folder = str(Path('~/data/').expanduser())
	tmp_folder = str(Path('./tmp/').expanduser())

g_tbd_folder = dataset_folder+'TbD/'
g_tbd3d_folder = dataset_folder+'TbD-3D/'
g_falling_folder = dataset_folder+'falling_objects/'
g_wildfmo_folder = dataset_folder+'wildfmo/'
g_youtube_folder = dataset_folder+'youtube/'

g_syn_folder = dataset_folder+'synthetic/'
g_bg_folder = dataset_folder+'vot2018.zip'

# TODO: download S2DNet weights:
# wget https://www.dropbox.com/s/hnv51iwu4hn82rj/s2dnet_weights.pth -P /cluster/scratch/denysr/dataset/
g_ext_folder = os.path.join(dataset_folder, 's2dnet_weights.pth')

g_resolution_x = int(640/2)
g_resolution_y = int(480/2)

g_use_selfsupervised_timeconsistency = True
g_timeconsistency_type = 'ncc'

