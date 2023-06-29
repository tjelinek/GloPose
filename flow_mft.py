from pathlib import Path

import sys
import torch
from argparse import Namespace

from cfg import DEVICE
from flow import export_flow_from_files

sys.path.append('RAFT-occl-uncertainty/RAFT')

from RAFT.core.raft import RAFT


def get_flow_model_mft():
    args = Namespace(model='/home/jelint19/datagrid/mnt/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/'
                           '50000_raft-things-sintel-occlusion-uncertainty.pth', model_name='RAFT', path=None,
                     mixed_precision=True, alternate_corr=False, small=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model


model = get_flow_model_mft()

if __name__ == "__main__":
    path_to_dataset = Path("data/360photo/original/concept/09")
    # path_to_dataset = Path("RAFT/demo-frames")
    export_flow_from_files(path_to_dataset, model)
