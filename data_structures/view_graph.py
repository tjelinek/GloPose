import pickle
from pathlib import Path

import networkx as nx
import torchvision.utils as vutils
from dataclasses import dataclass
from kornia.geometry import Se3
from data_structures.keyframe_buffer import FrameObservation


@dataclass
class ViewGraphNode:
    Se3_obj2cam: Se3
    observation: FrameObservation


class ViewGraph:
    def __init__(self):
        self.view_graph = nx.DiGraph()

    def add_node(self, node_id, se3_obj2cam, observation):
        """Adds a node with ViewGraphNode attributes."""
        self.view_graph.add_node(node_id, data=ViewGraphNode(se3_obj2cam, observation))

    def get_node_data(self, frame_idx) -> ViewGraphNode:
        """Returns the ViewGraphNode data for a given node ID."""
        if frame_idx in self.view_graph:
            return self.view_graph.nodes[frame_idx]["data"]
        else:
            raise KeyError(f"Node {frame_idx} not found in the graph.")

    def save(self, save_dir: Path, save_images: bool = False):
        """Saves the graph structure and associated images/segmentations to disk."""
        graph_path = save_dir / "graph.pkl"
        graph_path.mkdir(exist_ok=True)
        with open(graph_path, "wb") as f:
            pickle.dump(self, f)

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


def view_graph_from_pickle(self, load_dir: Path):
    """Loads the graph structure and associated images/segmentations from disk."""
    graph_path = load_dir / "graph.pkl"
    with open(graph_path, "rb") as f:
        self.view_graph = pickle.load(f)
