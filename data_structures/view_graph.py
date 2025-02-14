import os
import pickle
from pathlib import Path

import networkx as nx
import cv2
import torch
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

    def save(self, save_dir):
        """Saves the graph structure and associated images/segmentations to disk."""
        os.makedirs(save_dir, exist_ok=True)
        graph_path = os.path.join(save_dir, "graph.pkl")

        # Save images and segmentations separately
        for node_id, data in self.view_graph.nodes(data=True):
            node_data: ViewGraphNode = self.get_node_data(node_id)
            img_path = os.path.join(save_dir, f"{node_id}_image.png")
            seg_path = os.path.join(save_dir, f"{node_id}_seg.png")

            # Save image and segmentation
            cv2.imwrite(img_path, node_data.observation.observed_image)
            cv2.imwrite(seg_path, node_data.observation.observed_segmentation)

            # Convert Se3_obj2cam to a serializable format (e.g., tensor or numpy array)
            data["data"] = {
                "Se3_obj2cam": node_data.Se3_obj2cam.matrix().cpu(),
                "image_path": img_path,
                "seg_path": seg_path
            }

        # Save the modified graph as a pickle file
        with open(graph_path, "wb") as f:
            pickle.dump(self.view_graph, f)


def from_pickle(self, load_dir: Path):
    """Loads the graph structure and associated images/segmentations from disk."""
    graph_path = os.path.join(load_dir, "graph.pkl")
    with open(graph_path, "rb") as f:
        self.view_graph = pickle.load(f)

    # Reload images and segmentations into the graph nodes
    for node_id, data in self.view_graph.nodes(data=True):
        node_data = data["data"]
        se3_obj2cam = Se3(torch.tensor(node_data["Se3_obj2cam"]))
        image = cv2.imread(node_data["image_path"])
        segmentation = cv2.imread(node_data["seg_path"], cv2.IMREAD_GRAYSCALE)

        # Restore the original ViewGraphNode
        self.view_graph.nodes[node_id]["data"] = ViewGraphNode(
            se3_obj2cam, FrameObservation(image, segmentation)
        )
