import pickle
from pathlib import Path

import networkx as nx
import pycolmap
import torch
import torchvision.utils as vutils
from dataclasses import dataclass
from kornia.geometry import Se3, Quaternion

from data_structures.data_graph import DataGraph
from data_structures.keyframe_buffer import FrameObservation


@dataclass
class ViewGraphNode:
    Se3_cam2obj: Se3
    observation: FrameObservation
    colmap_db_image_id: int
    colmap_db_image_name: str


class ViewGraph:
    def __init__(self, device: str):
        self.view_graph = nx.DiGraph()
        self.device: str = device

    def add_node(self, node_id, se3_cam2obj, observation, colmap_db_image_id, colmap_db_image_name):
        """Adds a node with ViewGraphNode attributes."""
        self.view_graph.add_node(node_id, data=ViewGraphNode(se3_cam2obj, observation, colmap_db_image_id,
                                                             colmap_db_image_name))

    def get_node_data(self, frame_idx) -> ViewGraphNode:
        """Returns the ViewGraphNode data for a given node ID."""
        if frame_idx in self.view_graph:
            return self.view_graph.nodes[frame_idx]["data"]
        else:
            raise KeyError(f"Node {frame_idx} not found in the graph.")

    def save(self, save_dir: Path, save_images: bool = False, overwrite: bool = True, to_cpu: bool = False):
        """Saves the graph structure and associated images/segmentations to disk."""
        graph_path = save_dir / Path("graph.pkl")

        if graph_path.exists() and overwrite:
            if graph_path.is_dir():
                graph_path.rmdir()
            if graph_path.is_file():
                graph_path.unlink()

        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.is_file()

        if to_cpu:
            self.send_to_device('cpu')

        with open(graph_path, "wb") as f:
            pickle.dump(self, f)
            print(f"Written view graph to {str(graph_path)}")

        if save_images:
            image_save_dir = save_dir / "images"
            segmentations_save_dir = save_dir / "segmentations"

            image_save_dir.mkdir(exist_ok=True)
            segmentations_save_dir.mkdir(exist_ok=True)

            # Save images and segmentations separately
            for node_id, data in self.view_graph.nodes(data=True):
                node_data: ViewGraphNode = self.get_node_data(node_id)

                image = node_data.observation.observed_image  # Shape: (1, C, H, W)
                segmentation = node_data.observation.observed_segmentation  # Shape: (1, 1, H, W)

                img_path = image_save_dir / f"{node_id}_image.png"
                seg_path = segmentations_save_dir / f"{node_id}_seg.png"

                vutils.save_image(image, str(img_path))
                vutils.save_image(segmentation.float(), str(seg_path))

    def send_to_device(self, device):
        """Sends the graph to a given device."""
        self.device = device
        for node_id, data in self.view_graph.nodes(data=True):
            node_data: ViewGraphNode = data["data"]
            node_data.observation = node_data.observation.send_to_device(device)
            node_data.Se3_cam2obj = node_data.Se3_cam2obj.to(device)


def load_view_graph(load_dir: Path, device='cuda') -> ViewGraph:
    """Loads the graph structure and associated images/segmentations from disk."""
    graph_path = load_dir / "graph.pkl"
    with open(graph_path, "rb") as f:
        view_graph: ViewGraph = pickle.load(f)
        view_graph.send_to_device(device)

    return view_graph


def view_graph_from_datagraph(structure: nx.DiGraph, data_graph: DataGraph, colmap_reconstruction:
                              pycolmap.Reconstruction) -> ViewGraph:
    all_image_names = [str(data_graph.get_frame_data(i).image_filename)
                       for i in range(len(data_graph.G.nodes))]

    view_graph = ViewGraph(data_graph.storage_device)

    for image_id, image in colmap_reconstruction.images.items():
        frame_index = all_image_names.index(image.name)

        image_t_obj2cam = torch.tensor(image.cam_from_world.translation)[None]
        image_q_obj2cam_xyzw = torch.tensor(image.cam_from_world.rotation.quat)[None]
        image_q_obj2cam_wxyz = image_q_obj2cam_xyzw[:, [3, 0, 1, 2]]

        gt_Se3_obj2cam = Se3(Quaternion(image_q_obj2cam_wxyz), image_t_obj2cam)
        gt_Se3_cam2obj = gt_Se3_obj2cam.inverse()

        frame_observation = data_graph.get_frame_data(frame_index).frame_observation

        view_graph.add_node(frame_index, gt_Se3_cam2obj, frame_observation, image_id, image.name)

    # TODO this causes errors when COLMAP does not register an image
    # for u, v in structure.edges:
    #     if u not in view_graph.view_graph.nodes and v not in view_graph.view_graph.nodes:
    #         view_graph.view_graph.add_edge(u, v)

    return view_graph
