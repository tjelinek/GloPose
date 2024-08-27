import copy
import sys
from bisect import insort
from copy import deepcopy
from functools import reduce
from typing import Tuple, List, Dict, Union, TypeVar, get_origin, get_args, Optional
from itertools import chain

import numpy as np
import torch
import torchvision.transforms.functional as TF
import networkx as nx

from dataclasses import dataclass, field, replace, is_dataclass

from auxiliary_scripts.cameras import Cameras
from flow import flow_unit_coords_to_image_coords, flow_image_coords_to_unit_coords

sys.path.append('repositories/MFT_tracker')
from repositories.MFT_tracker.MFT.results import FlowOUTrackingResult
from repositories.MFT_tracker.MFT.MFTIQ import chain_results

ObservationType = TypeVar('ObservationType', bound='Observation')
FlowObservationType = TypeVar('FlowObservationType', bound='BaseFlowObservation')


@dataclass
class Observation:

    def trim_bounding_box(self, bounding_box: Tuple[int, int, int, int]):
        copy = deepcopy(self)

        for attr_name, attr_type in self.__annotations__.items():
            to_trim = getattr(self, attr_name)
            if not issubclass(attr_type, torch.Tensor) or to_trim is None:
                continue
            trimmed = to_trim[:, :, :, bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
            setattr(copy, attr_name, trimmed)

        return copy

    def assert_same_length(self) -> Optional[int]:

        global_shape = None

        for attr_name, attr_type in self.__annotations__.items():
            value = getattr(self, attr_name)
            if value is not None:
                if issubclass(attr_type, torch.Tensor):
                    attr_length = value.shape[1]
                elif get_origin(attr_type) is list:
                    attr_length = len(value)
                else:
                    attr_length = None

                if attr_length is not None and global_shape is None:
                    global_shape = attr_length
                if attr_length is not None and global_shape is not None:
                    assert attr_length == global_shape

        return global_shape

    def get_memory_size(self):
        memory_used = 0.0
        for attr_name, attr_type in self.__annotations__.items():
            if issubclass(attr_type, torch.Tensor):
                attr = getattr(self, attr_name)
                memory_used += attr.numel() * attr.element_size()

        return memory_used

    @staticmethod
    def concatenate(*observations):

        assert all(type(observation) is type(observations[0]) for observation in observations)
        observation_type = type(observations[0])
        concatenated_observations = observation_type()

        concatenated_length = None

        for attr_name, attr_type in observation_type.__annotations__.items():
            check_for_none = (getattr(observation, attr_name) is None for observation in observations)
            if any(check_for_none):
                assert all(check_for_none)
            else:
                if issubclass(attr_type, torch.Tensor):
                    concatenated_attr = torch.cat([getattr(observation, attr_name) for observation in observations],
                                                  dim=1)

                    attr_length = concatenated_attr.shape[1]

                    setattr(concatenated_observations, attr_name, concatenated_attr)
                elif get_origin(attr_type) is list:
                    concatenated_list = list(chain.from_iterable([getattr(observation, attr_name)
                                                                  for observation in observations]))
                    setattr(concatenated_observations, attr_name, concatenated_list)

                    attr_length = len(concatenated_list)
                else:
                    attr_length = None

                if concatenated_length is None:
                    concatenated_length = attr_length
                elif attr_length is not None:
                    assert concatenated_length == attr_length

        return concatenated_observations

    def filter_frames(self: ObservationType, frames_indices: List) -> ObservationType:

        new_observation = type(self)()

        for attr_name, attr_type in self.__annotations__.items():
            value = getattr(self, attr_name)
            if value is not None:
                if issubclass(attr_type, torch.Tensor):
                    setattr(new_observation, attr_name, value[:, frames_indices, :])
                elif get_origin(attr_type) is list:
                    setattr(new_observation, attr_name, [value[idx] for idx in frames_indices])

        return new_observation

    def send_to_device(self: ObservationType, device: str) -> ObservationType:

        if not is_dataclass(self):
            raise ValueError("send_to_device should only be used with dataclass instances")

        new_fields = {}

        for attr_name, attr_type in self.__annotations__.items():
            value = getattr(self, attr_name)
            # Check if the attribute type is torch.Tensor
            if value is not None and isinstance(value, torch.Tensor):
                new_fields[attr_name] = value.to(device)
            else:
                new_fields[attr_name] = value

        new_observation = replace(self, **new_fields)

        return new_observation

    def cuda(self: ObservationType):
        return self.send_to_device('cuda')

    def cpu(self: ObservationType):
        return self.send_to_device('cpu')


@dataclass
class FrameObservation(Observation):
    observed_image: torch.Tensor = None
    observed_image_features: torch.Tensor = None
    observed_segmentation: torch.Tensor = None


@dataclass
class BaseFlowObservation(Observation):
    observed_flow: torch.Tensor = None
    observed_flow_segmentation: torch.Tensor = None
    observed_flow_occlusion: torch.Tensor = None
    coordinate_system: str = 'unit'  # Either 'unit' or 'image'
    flow_source_frames: list[int] = field(default_factory=list)
    flow_target_frames: list[int] = field(default_factory=list)

    def cast_unit_coords_to_image_coords(self):
        if self.coordinate_system != 'unit':
            raise ValueError("Current coordinate system must be 'unit' to cast to 'image' coordinates.")

        observed_flow_image_coords = flow_unit_coords_to_image_coords(self.observed_flow)

        new_instance = replace(self, observed_flow=observed_flow_image_coords, coordinate_system='image')
        return new_instance

    def cast_image_coords_to_unit_coords(self):
        if self.coordinate_system != 'image':
            raise ValueError("Current coordinate system must be 'image' to cast to 'unit' coordinates.")

        observed_flow_unit_coords = flow_image_coords_to_unit_coords(self.observed_flow)

        new_instance = replace(self, observed_flow=observed_flow_unit_coords, coordinate_system='unit')
        return new_instance

    @staticmethod
    def chain(*flow_observations: FlowObservationType) -> FlowObservationType:

        unique_lengths = set(flow_observation.assert_same_length() for flow_observation in flow_observations)
        assert len(unique_lengths) == 1 and next(iter(unique_lengths)) == 1

        coordinate_system = flow_observations[0].coordinate_system
        assert all(observation.coordinate_system == coordinate_system for observation in flow_observations)
        # At the moment, implemented for single flow only

        for prev_obs, next_obs in zip(flow_observations[:-2], flow_observations[1:]):
            assert isinstance(prev_obs, FlowObservation)
            assert isinstance(next_obs, FlowObservation)

            assert prev_obs.flow_target_frames == next_obs.flow_source_frames

        if coordinate_system == 'unit':
            flow_observations_mft_type = [obs.cast_unit_coords_to_image_coords().to_mft_format()
                                          for obs in flow_observations]
        else:
            flow_observations_mft_type = [obs.to_mft_format() for obs in flow_observations]

        chained_flow_mft_type = reduce(chain_results, flow_observations_mft_type)

        chained_flow: FlowObservationType = (
            type(flow_observations[0])(observed_flow=chained_flow_mft_type.flow[None, None],
                                       observed_flow_occlusion=chained_flow_mft_type.occlusion[None, None],
                                       observed_flow_uncertainty=chained_flow_mft_type.sigma[None, None],
                                       observed_flow_segmentation=flow_observations[0].observed_flow_segmentation,
                                       coordinate_system='image'))

        if coordinate_system == 'unit':
            chained_flow = chained_flow.cast_image_coords_to_unit_coords()

        return chained_flow

    def to_mft_format(self) -> FlowOUTrackingResult:

        flow_uncertainty = getattr(self, 'observed_flow_uncertainty', None)
        if flow_uncertainty is not None:
            flow_uncertainty = flow_uncertainty[0, 0]

        # At the moment, implemented for single flow only
        mft_format_observation = FlowOUTrackingResult(flow=self.observed_flow[0, 0],
                                                      occlusion=self.observed_flow_occlusion[0, 0],
                                                      sigma=flow_uncertainty)

        return mft_format_observation


@dataclass
class FlowObservation(BaseFlowObservation):
    # Needs to be redeclared so as it contained in __annotations__.items() used for generic manipulations
    observed_flow: torch.Tensor = None
    observed_flow_segmentation: torch.Tensor = None
    observed_flow_occlusion: torch.Tensor = None
    observed_flow_uncertainty: torch.Tensor = None
    coordinate_system: str = 'unit'  # Either 'unit' or 'image'
    flow_source_frames: list[int] = field(default_factory=list)
    flow_target_frames: list[int] = field(default_factory=list)


@dataclass
class SyntheticFlowObservation(BaseFlowObservation):
    # Needs to be redeclared so as it contained in __annotations__.items() used for generic manipulations
    observed_flow: torch.Tensor = None
    observed_flow_segmentation: torch.Tensor = None
    observed_flow_occlusion: torch.Tensor = None
    coordinate_system: str = 'unit'  # Either 'unit' or 'image'
    flow_source_frames: list[int] = field(default_factory=list)
    flow_target_frames: list[int] = field(default_factory=list)


@dataclass
class MultiCameraObservation:
    cameras_observations: Dict[Cameras, Union[FrameObservation, FlowObservationType]] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'MultiCameraObservation':
        observation_types = {type(obs) for obs in kwargs.values()}
        if len(observation_types) != 1:
            raise ValueError("Mixed observation types are not supported.")

        cameras_observations_kwargs = {
            Cameras[key.upper()]: value for key, value in kwargs.items() if key.upper() in Cameras.__members__
        }
        return cls(cameras_observations=cameras_observations_kwargs)

    def stack(self) -> Union[FrameObservation, FlowObservationType]:
        observation_types = {type(obs) for obs in self.cameras_observations.values()}

        if len(observation_types) > 1:
            raise ValueError("Mixed observation types are not supported.")

        observation_type = observation_types.pop()

        sorted_observations = sorted(self.cameras_observations.items(), key=lambda item: item[0].value)
        observations = [obs for _, obs in sorted_observations]

        if issubclass(observation_type, FrameObservation) or issubclass(observation_type, BaseFlowObservation):
            return observation_type.concatenate(*observations)
        else:
            raise ValueError("Unknown observation type encountered.")


class KeyframeBuffer:

    def __init__(self, storage_device='cuda', output_device='cuda'):
        self._storage_device = storage_device
        self._output_device = output_device

        self.G: nx.DiGraph = nx.DiGraph()

        # Contains ordinals of a frame that we optimize for shape etc.
        self.keyframes: list = []
        # Contains ordinals of a frame that we may not optimize but serve as a keyframe source
        self.flow_frames: list = []

    @staticmethod
    def merge(buffer1, buffer2):
        """
        Concatenates and merges two KeyframeBuffer instances while sorting the keyframes. Assumes buffer1.keyframes
        and buffer2.keyframes are disjoint.

        Args:
            buffer1 (KeyframeBuffer): The first KeyframeBuffer instance.
            buffer2 (KeyframeBuffer): The second KeyframeBuffer instance.

        Returns:
            Tuple[KeyframeBuffer, List[int], List[int]]: A tuple containing the merged KeyframeBuffer instance,
            indices of buffer1 keyframes in the merged buffer, and indices of buffer2 keyframes in the merged buffer.

        """
        all_keyframes = list(sorted(set(buffer1.keyframes) | set(buffer2.keyframes)))
        all_flow_frames = list(sorted(set(buffer1.flow_frames) | set(buffer2.flow_frames)))

        indices_buffer1 = []
        indices_buffer2 = []

        merged_buffer = KeyframeBuffer()
        merged_buffer.keyframes = all_keyframes
        merged_buffer.flow_frames = all_flow_frames
        merged_buffer.G = nx.compose(buffer1.G, buffer2.G)

        indices_buffer1.extend([buffer1.keyframes.index(k) for k in all_keyframes if k in buffer1.keyframes])
        indices_buffer2.extend([buffer2.keyframes.index(k) for k in all_keyframes if k in buffer2.keyframes])

        return merged_buffer, indices_buffer1, indices_buffer2

    def get_memory_size(self):
        memory_used = 0.0
        for _, attrs in self.G.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, Observation):
                    memory_used += value.get_memory_size()

        for _, _, attrs in self.G.edges(data=True):
            for key, value in attrs.items():
                if isinstance(value, Observation):
                    memory_used += value.get_memory_size()

        return memory_used

    def add_new_keyframe_observation(self, frame_observation: FrameObservation, keyframe_index: int) -> None:

        self.G.add_node(keyframe_index, observations=frame_observation.send_to_device(self._storage_device))
        insort(self.keyframes, keyframe_index)

    def add_new_flow(self, observed_flow: torch.Tensor, observed_flows_segmentation: torch.Tensor,
                     observed_flows_occlusion: torch.Tensor, observed_flows_uncertainty: torch.Tensor,
                     source_frame: int, target_frame: int) -> None:

        flow_observation = FlowObservation(observed_flow=observed_flow,
                                           observed_flow_segmentation=observed_flows_segmentation,
                                           observed_flow_uncertainty=observed_flows_uncertainty,
                                           observed_flow_occlusion=observed_flows_occlusion,
                                           flow_source_frames=[source_frame],
                                           flow_target_frames=[target_frame])

        self.add_new_flow_observation(flow_observation, source_frame, target_frame)

    def add_new_flow_observation(self, flow_observation: FlowObservation, source_frame: int, target_frame: int) -> None:
        if source_frame not in self.G:
            raise ValueError("source_frame not in the graph")
        elif target_frame not in self.G:
            raise ValueError("target_frame not in the graph")

        if source_frame not in self.flow_frames:
            insort(self.flow_frames, source_frame)

        assert len(flow_observation.flow_source_frames) != 0 and len(flow_observation.flow_target_frames) != 0

        self.G.add_edge(source_frame, target_frame,
                        flow_observations=flow_observation.send_to_device(self._storage_device))

    def get_flows_between_frames(self, source_frame: int, target_frame: int) -> FlowObservation:

        return self.G.get_edge_data(source_frame, target_frame)['flow_observations'].send_to_device(self._output_device)

    def get_observations_for_all_keyframes(self, bounding_box: Tuple[int, int, int, int] = None) -> FrameObservation:
        vertices = list(filter(lambda vertex: vertex[0] in self.keyframes, self.G.nodes(data=True)))
        vertices.sort(key=lambda vertex: vertex[0])

        vertices_observations = [vertex[1]['observations'] for vertex in vertices]
        concatenated_observations = FrameObservation.concatenate(*vertices_observations)

        if bounding_box is not None:
            concatenated_observations = concatenated_observations.trim_bounding_box(bounding_box)

        return concatenated_observations.send_to_device(self._output_device)

    def get_observations_for_keyframe(self, keyframe,
                                      bounding_box: Tuple[int, int, int, int] = None) -> FrameObservation:
        observations = self.G.nodes[keyframe]['observations']

        if bounding_box is not None:
            observations = observations.trim_bounding_box(bounding_box)

        return observations.send_to_device(self._output_device)

    def get_flows_observations(self, bounding_box=None):
        arcs = list(self.G.edges(data=True))
        arcs.sort(key=lambda edge: edge[:2:-1])  # Sort by the target frame, then by the source frame

        flow_observations = [arc[2]['flow_observations'] for arc in arcs]
        concatenated_tensors = FlowObservation.concatenate(*flow_observations)

        if bounding_box is not None:
            concatenated_tensors = (concatenated_tensors.trim_bounding_box(bounding_box).
                                    send_to_device(self._output_device))

        return concatenated_tensors

    def trim_keyframes(self, max_keyframes):
        if len(self.keyframes) > max_keyframes:
            # Keep only those last ones
            keep_keyframes = np.zeros(len(self.keyframes), dtype=bool)
            keep_keyframes[-max_keyframes:] = True

            return self.keep_selected_keyframes(keep_keyframes)
        else:
            return KeyframeBuffer()

    def remove_frames(self, frames_to_remove: List):
        remaining_keyframes = sorted(list(set(self.keyframes) - set(frames_to_remove)))
        remaining_flow_frames = sorted(list(set(self.flow_frames) - set(frames_to_remove)))

        self.G.remove_nodes_from(frames_to_remove)
        self.keyframes = copy.deepcopy(remaining_keyframes)
        self.flow_frames = copy.deepcopy(remaining_flow_frames)

    def remove_edges(self, edges_to_remove: List[Tuple[int, int]]):
        self.G.remove_edges_from(edges_to_remove)

    def keep_selected_keyframes(self, keep_keyframes):
        not_keep_keyframes = ~ keep_keyframes
        kept_keyframes = (np.array(sorted(self.keyframes))[keep_keyframes]).tolist()
        deleted_keyframes = (np.array(sorted(self.keyframes))[not_keep_keyframes]).tolist()

        kept_graph: nx.DiGraph = self.G.subgraph(kept_keyframes + self.flow_frames).copy()
        deleted_graph: nx.DiGraph = self.G.subgraph(deleted_keyframes + self.flow_frames).copy()

        deleted_buffer = KeyframeBuffer()
        kept_graph.remove_nodes_from([node for node in set(self.flow_frames) & set(kept_graph.nodes)
                                      if kept_graph.out_degree(node) == 0])
        deleted_graph.remove_nodes_from([node for node in set(self.flow_frames) & set(deleted_graph.nodes)
                                         if deleted_graph.out_degree(node) == 0])

        deleted_buffer.flow_frames = sorted(set(self.flow_frames) & set(deleted_graph.nodes))
        deleted_buffer.keyframes = sorted(set(self.keyframes) & set(deleted_graph.nodes))
        deleted_buffer.G = deleted_graph

        self.flow_frames = sorted(set(self.flow_frames) & set(kept_graph.nodes))
        self.keyframes = sorted(set(self.keyframes) & set(kept_graph.nodes))
        self.G = kept_graph

        return deleted_buffer


def generate_rotated_observations(observation: FrameObservation, degree: int):
    if 360 % degree != 0:
        raise ValueError("Degree must divide 360 evenly.")

    num_rotations = 360 // degree
    rotated_observations = []
    degrees = []

    for i in range(num_rotations):
        rotation_degree = i * degree
        degrees.append(rotation_degree)

        # Rotate the image, features, and segmentation
        rotated_image = TF.rotate(observation.observed_image[0, 0], rotation_degree)[None, None]
        rotated_image_features = TF.rotate(observation.observed_image_features[0, 0], rotation_degree)[None, None]
        rotated_segmentation = TF.rotate(observation.observed_segmentation[0, 0], rotation_degree)[None, None]

        # Create a new FrameObservation with the rotated tensors
        rotated_observation = FrameObservation(rotated_image, rotated_image_features, rotated_segmentation)
        rotated_observations.append(rotated_observation)

    return rotated_observations, degrees
