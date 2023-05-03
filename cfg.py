import sys
from pathlib import Path

sys.path.append('GMA/core')
sys.path.append('OSTrack/lib')
sys.path.append('track6d')

FLOW_OUT_DEFAULT_DIR = Path('./data/flow_out')
DEVICE = 'cuda'
