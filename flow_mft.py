from pathlib import Path

import sys
import torch
from argparse import Namespace

from cfg import DEVICE
from flow import export_flow_from_files

# sys.path.append('RAFTocclUncertainty/RAFT')
# sys.path.append('RAFTocclUncertainty')

from RAFT.core.raft import RAFT


def get_flow_model_mft():
    args = Namespace(model='RAFTocclUncertainty/RAFT/models/'
                           'raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth',
                     model_name='RAFT', path=None,
                     mixed_precision=True, alternate_corr=False, small=False)
    raft_kwargs = {
        'occlusion_module': 'separate_with_uncertainty',
        'restore_ckpt': 'RAFTocclUncertainty/RAFT/models',
        'small': False,
        'mixed_precision': False,
    }

    raft_params = Namespace(**raft_kwargs)
    model = torch.nn.DataParallel(RAFT(raft_params))
    model.load_state_dict(torch.load(args.model))

    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model
