import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import networkx as nx
import pycolmap
import torch
import torchvision.utils as vutils
from einops import rearrange
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from kornia.geometry import Se3, Quaternion
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from data_structures.data_graph import DataGraph
from data_structures.keyframe_buffer import FrameObservation
from pose.colmap_utils import merge_two_databases, merge_colmap_reconstructions


@dataclass
class ViewGraphNode:
    Se3_obj2cam: Se3
    observation: FrameObservation
    colmap_db_image_id: int
    colmap_db_image_name: str
    dino_descriptor: torch.Tensor


class ViewGraph:
    def __init__(self, object_id: int | str, colmap_db_path: Path,
                 colmap_output_path: Path, device: str):
        self.view_graph = nx.DiGraph()
        self.object_id: int | str = object_id
        self.colmap_db_path: Path = colmap_db_path
        self.colmap_reconstruction_path: Path = colmap_output_path
        self.device: str = device

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        cfg_dir = (Path(__file__).parent.parent / 'repositories' / 'cnos' / 'configs').resolve()
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cnos_cfg = compose(config_name="run_inference")

        sys.path.append('./repositories/cnos')
        from src.model.dinov2 import CustomDINOv2
        self.dino_descriptor: CustomDINOv2 = instantiate(cnos_cfg.model.descriptor_model).to(self.device)
        self.dino_descriptor.model = self.dino_descriptor.model.to(self.device)
        self.dino_descriptor.model.device = self.device

    def add_node(self, node_id, Se3_obj2cam, observation, colmap_db_image_id, colmap_db_image_name):
        """Adds a node with ViewGraphNode attributes."""
        from src.model.utils import Detections
        image_tensor = observation.observed_image
        segmentation_mask = observation.observed_segmentation.squeeze(1).to(self.device)
        segmentation_bbox = masks_to_boxes(segmentation_mask)
        image_np = rearrange((image_tensor * 255).to(torch.uint8), '1 c h w -> h w c').numpy(force=True)
        detections = Detections({'masks': segmentation_mask, 'boxes': segmentation_bbox})
        dino_descriptor = self.dino_descriptor(image_np, detections).squeeze()
        self.view_graph.add_node(node_id, data=ViewGraphNode(Se3_obj2cam, observation, colmap_db_image_id,
                                                             colmap_db_image_name, dino_descriptor))

    def get_node_data(self, frame_idx) -> ViewGraphNode:
        """Returns the ViewGraphNode data for a given node ID."""
        if frame_idx in self.view_graph:
            return self.view_graph.nodes[frame_idx]["data"]
        else:
            raise KeyError(f"Node {frame_idx} not found in the graph.")

    def get_concatenated_descriptors(self) -> torch.Tensor:
        view_graph_descriptors = []
        for node_idx in sorted(self.view_graph.nodes):
            node = self.get_node_data(node_idx)
            node_descriptors = node.dino_descriptor
            view_graph_descriptors.append(node_descriptors)

        view_graph_descriptors = torch.stack(view_graph_descriptors)

        return view_graph_descriptors

    def save_viewgraph(self, save_dir: Path, colmap_reconstruction: pycolmap.Reconstruction,
                       save_images: bool = False, overwrite: bool = True, to_cpu: bool = False):
        """Saves the graph structure and associated images/segmentations to disk."""
        graph_path = save_dir / Path("graph.pkl")

        if save_dir.exists() and overwrite:
            shutil.rmtree(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        reconstruction_path = save_dir / 'reconstruction' / '0'
        reconstruction_path.mkdir(exist_ok=True, parents=True)
        colmap_reconstruction.write(str(reconstruction_path))
        self.colmap_reconstruction_path = reconstruction_path

        new_db_path = save_dir / self.colmap_db_path.name
        if self.colmap_db_path != new_db_path:
            shutil.copy(self.colmap_db_path, new_db_path)
        self.colmap_db_path = new_db_path

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
            node_data.Se3_obj2cam = node_data.Se3_obj2cam.to(device)
            node_data.dino_descriptor = node_data.dino_descriptor.to(device)


def load_view_graph(load_dir: Path, device='cuda') -> ViewGraph:
    """Loads the graph structure and associated images/segmentations from disk."""
    graph_path = load_dir / "graph.pkl"

    sys.path.append('./repositories/cnos')

    with open(graph_path, "rb") as f:
        view_graph: ViewGraph = pickle.load(f)
        view_graph.send_to_device(device)

    return view_graph


def view_graph_from_datagraph(structure: nx.DiGraph, data_graph: DataGraph,
                              colmap_reconstruction: pycolmap.Reconstruction, colmap_db_path,
                              colmap_output_path, object_id: int | str) -> ViewGraph:
    all_image_names = [str(data_graph.get_frame_data(i).image_filename)
                       for i in range(len(data_graph.G.nodes))]

    view_graph = ViewGraph(object_id, colmap_db_path, colmap_output_path,
                           data_graph.storage_device)

    for image_id, image in colmap_reconstruction.images.items():
        frame_index = all_image_names.index(image.name)

        image_t_obj2cam = torch.tensor(image.cam_from_world.translation)[None]
        image_q_obj2cam_xyzw = torch.tensor(image.cam_from_world.rotation.quat)[None]
        image_q_obj2cam_wxyz = image_q_obj2cam_xyzw[:, [3, 0, 1, 2]]

        Se3_obj2cam = Se3(Quaternion(image_q_obj2cam_wxyz), image_t_obj2cam)

        frame_observation = data_graph.get_frame_data(frame_index).frame_observation

        view_graph.add_node(frame_index, Se3_obj2cam, frame_observation, image_id, image.name)

    # TODO this causes errors when COLMAP does not register an image
    # for u, v in structure.edges:
    #     if u not in view_graph.view_graph.nodes and v not in view_graph.view_graph.nodes:
    #         view_graph.view_graph.add_edge(u, v)

    return view_graph


def merge_two_view_graphs(viewgraph1_folder: Path, viewgraph2_folder: Path, merged_folder: Path):

    if merged_folder.exists():
        shutil.rmtree(merged_folder)
    merged_folder.mkdir(parents=True, exist_ok=True)

    view_graph1 = load_view_graph(viewgraph1_folder, device='cpu')
    view_graph2 = load_view_graph(viewgraph2_folder, device='cpu')

    colmap_db1_path = view_graph1.colmap_db_path
    colmap_db2_path = view_graph2.colmap_db_path

    merged_db_path: Path = merged_folder / "database.db"
    db1_image_rename_dict, db2_image_rename_dict = merge_two_databases(colmap_db1_path, colmap_db2_path, merged_db_path)

    merged_db = pycolmap.Database(str(merged_db_path))

    viewgraph1_node_relabel_mapping = relabel_viewgraph_nodes(merged_db, view_graph1, db1_image_rename_dict)
    viewgraph2_node_relabel_mapping = relabel_viewgraph_nodes(merged_db, view_graph2, db2_image_rename_dict)

    copy_relabeled_images(viewgraph1_folder, viewgraph1_node_relabel_mapping, merged_folder)
    copy_relabeled_images(viewgraph2_folder, viewgraph2_node_relabel_mapping, merged_folder)

    merged_reconstruction_path = merged_folder / 'reconstruction'

    reconstruction1 = pycolmap.Reconstruction(str(view_graph1.colmap_reconstruction_path))
    reconstruction2 = pycolmap.Reconstruction(str(view_graph2.colmap_reconstruction_path))

    merged_reconstruction = merge_colmap_reconstructions(reconstruction1, reconstruction2)

    merged_viewgraph = ViewGraph(view_graph1.object_id, merged_db_path, merged_reconstruction_path, view_graph1.device)

    merged_viewgraph_G = nx.compose(view_graph1.view_graph, view_graph2.view_graph)
    merged_viewgraph.view_graph = merged_viewgraph_G
    merged_viewgraph.save_viewgraph(merged_folder, merged_reconstruction, save_images=True, overwrite=False,
                                    to_cpu=True)


def copy_relabeled_images(source_viewgraph_folder, viewgraph_node_relabel_mapping, target_viewgraph_folder):
    viewgraph_img_folder = source_viewgraph_folder / 'images'
    viewgraph_seg_folder = source_viewgraph_folder / 'segmentations'
    merged_img_folder = target_viewgraph_folder / 'images'
    merged_seg_folder = target_viewgraph_folder / 'segmentations'

    merged_img_folder.mkdir(parents=True, exist_ok=True)
    merged_seg_folder.mkdir(parents=True, exist_ok=True)

    for old_img_id, new_img_id in viewgraph_node_relabel_mapping.items():
        old_image_path = viewgraph_img_folder / f'{old_img_id}_image.png'
        old_seg_path = viewgraph_seg_folder / f'{old_img_id}_seg.png'

        new_image_path = merged_img_folder / f'{new_img_id}_image.png'
        new_seg_path = merged_seg_folder / f'{new_img_id}_seg.png'

        shutil.copy(old_image_path, new_image_path)
        shutil.copy(old_seg_path, new_seg_path)


def relabel_viewgraph_nodes(merged_db: pycolmap.Database, view_graph: ViewGraph,
                            db_image_rename_dict: Dict[str, str] = None) -> Dict[Any, Any]:
    all_merged_images = {image.name: image for image in merged_db.read_all_images()}
    viewgraph_node_relabel_mapping = {}
    image_rename_mapping = {}
    for node_id in view_graph.view_graph.nodes:
        node = view_graph.get_node_data(node_id)
        old_image_name = node.colmap_db_image_name
        new_image_name = db_image_rename_dict[old_image_name]

        merged_db_image = all_merged_images[new_image_name]
        new_image_colmap_id = merged_db_image.image_id

        node.colmap_db_image_id = new_image_colmap_id
        node.colmap_db_image_name = new_image_name

        viewgraph_node_relabel_mapping[node_id] = new_image_colmap_id
        image_rename_mapping[old_image_name] = new_image_name

    view_graph.view_graph = nx.relabel_nodes(view_graph.view_graph, viewgraph_node_relabel_mapping)

    return viewgraph_node_relabel_mapping


def load_view_graphs_by_object_id(view_graph_save_paths: Path, onboarding_type: str, device) -> Dict[Any, ViewGraph]:

    view_graphs: Dict[Any, ViewGraph] = {}
    total_dirs = sum(1 for d in view_graph_save_paths.iterdir() if d.is_dir())
    for i, view_graph_dir in tqdm(enumerate(view_graph_save_paths.iterdir()), total=total_dirs,
                                  desc="Loading view graphs"):
        if view_graph_dir.is_dir():
            if onboarding_type == 'static':
                if not view_graph_dir.stem.endswith('_both'):
                    continue
            elif onboarding_type == 'dynamic':
                if not view_graph_dir.stem.endswith('_dynamic'):
                    continue
            else:
                raise ValueError(f"Unknown onboarding type {onboarding_type}")

            view_graph: ViewGraph = load_view_graph(view_graph_dir, device=device)
            view_graphs[view_graph.object_id] = view_graph

    return view_graphs
