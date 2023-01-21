import sys
from pathlib import Path

sys.path.append('')
sys.path.append('GMA/core')

print(sys.path)

FLOW_OUT_DEFAULT_DIR = Path('../data/flow_out')
DEVICE = 'cuda'
