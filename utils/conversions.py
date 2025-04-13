from kornia.geometry import Se3
from pycolmap import Rigid3d


def Se3_to_Rigid3d(Se3_pose: Se3) -> Rigid3d:
    R_np = Se3_pose.rotation.matrix().numpy(force=True)
    t_np = Se3_pose.t.numpy(force=True)
    Rigid3d_pose = Rigid3d(R_np, t_np)

    return Rigid3d_pose
