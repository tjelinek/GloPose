import os
from pathlib import Path

dataset_folder = '/mnt/lascar/rozumden/dataset/'
tmp_folder = '/home.stud/rozumden/tmp/'

if not os.path.exists(dataset_folder):
	dataset_folder = '/cluster/scratch/denysr/dataset/'
	tmp_folder = '/cluster/scratch/denysr/eval/'

# TODO: put your paths here:
if not os.path.exists(dataset_folder):
	dataset_folder = str(Path('/mnt/personal/jelint19/data/').expanduser())
	tmp_folder = str(Path('/mnt/personal/jelint19/results/FlowTracker/').expanduser())

# TODO: download S2DNet weights:
# wget https://www.dropbox.com/s/hnv51iwu4hn82rj/s2dnet_weights.pth -P /cluster/scratch/denysr/dataset/
g_ext_folder = os.path.join(dataset_folder, 's2dnet_weights.pth')



