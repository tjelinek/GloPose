import sys
import torch
from argparse import Namespace

from cfg import DEVICE

sys.path.append('MFT')

from MFT.RAFT.core.raft import RAFT


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__.update(kwargs)


def get_flow_model_mft():
    args = Namespace(model='MFT/RAFT/models/'
                           'raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth',
                     model_name='MFT', path=None,
                     mixed_precision=False, alternate_corr=False, small=False)

    raft_params = {
        'occlusion_module': 'separate_with_uncertainty',
        'restore_ckpt': 'MFT/RAFT/models',
        'small': False,
        'mixed_precision': False,
    }

    raft_params = AttrDict(**raft_params)
    model = torch.nn.DataParallel(RAFT(raft_params))
    model.load_state_dict(torch.load(args.model))

    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model
