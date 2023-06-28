from argparse import Namespace

import torch
import sys

from flow import export_flow_from_files

sys.path.append('GMA')
# sys.path.append('GMA/core')

from cfg import *

from GMA.core.network import RAFTGMA


def get_flow_model_gma():
    args = Namespace(model='checkpoints/gma-sintel.pth', model_name='GMA', path=None, num_heads=1, position_only=False,
                     position_and_content=False, mixed_precision=True)

    model = torch.nn.DataParallel(RAFTGMA(args=args))
    checkpoint_path = Path('GMA/checkpoints/gma-sintel.pth')
    model.load_state_dict(torch.load(checkpoint_path))

    print(f"Loaded checkpoint at {checkpoint_path}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model


model = get_flow_model_gma()

if __name__ == "__main__":
    path_to_dataset = Path("data/360photo/original/concept/09")
    export_flow_from_files(path_to_dataset, model)
