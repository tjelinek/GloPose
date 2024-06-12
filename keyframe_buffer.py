from bisect import insort
from copy import deepcopy
from typing import Tuple, List, Dict, Union, TypeVar

import numpy as np
import torch
import networkx as nx

from dataclasses import dataclass, field, replace

from auxiliary_scripts.cameras import Cameras
from flow import flow_unit_coords_to_image_coords, flow_image_coords_to_unit_coords


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

        for attr_name, attr_type in observation_type.__annotations__.items():
            check_for_none = (getattr(observation, attr_name) is None for observation in observations)
            if any(check_for_none):
                assert all(check_for_none)
            elif not issubclass(attr_type, torch.Tensor):
                continue
            else:
                concatenated_attr = torch.cat([getattr(observation, attr_name) for observation in observations], dim=1)
                setattr(concatenated_observations, attr_name, concatenated_attr)

        return concatenated_observations

    def filter_frames(self: ObservationType, frames_indices: List) -> ObservationType:

        new_observation = type(self)()

        for attr_name, attr_type in self.__annotations__.items():
            value = getattr(self, attr_name)
            if value is not None and issubclass(attr_type, torch.Tensor):
                setattr(new_observation, attr_name, value[:, frames_indices, :])

        return new_observation

    def send_to_device(self: ObservationType, device: str) -> ObservationType:

        new_observation = type(self)()

        for attr_name, attr_type in self.__annotations__.items():
            value: torch.Tensor = getattr(self, attr_name)
            if value is not None and issubclass(attr_type, torch.Tensor):
                setattr(new_observation, attr_name, value.to(device))

        return new_observation


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


@dataclass
class FlowObservation(BaseFlowObservation):
    # Needs to be redeclared so as it contained in __annotations__.items() used for generic manipulations
    observed_flow: torch.Tensor = None
    observed_flow_segmentation: torch.Tensor = None
    observed_flow_occlusion: torch.Tensor = None
    observed_flow_uncertainty: torch.Tensor = None
    coordinate_system: str = 'unit'  # Either 'unit' or 'image'


@dataclass
class SyntheticFlowObservation(BaseFlowObservation):
    # Needs to be redeclared so as it contained in __annotations__.items() used for generic manipulations
    observed_flow: torch.Tensor = None
    observed_flow_segmentation: torch.Tensor = None
    observed_flow_occlusion: torch.Tensor = None
    coordinate_system: str = 'unit'  # Either 'unit' or 'image'


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


@dataclass
class KeyframeBuffer:
    G: nx.DiGraph = field(default_factory=nx.DiGraph)

    # Contains ordinals of a frame that we optimize for shape etc.
    keyframes: list = field(default_factory=list)
    # Contains ordinals of a frame that we may not optimize but serve as a keyframe source
    flow_frames: list = field(default_factory=list)

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

        self.G.add_node(keyframe_index, observations=frame_observation)
        insort(self.keyframes, keyframe_index)

    def add_new_flow(self, observed_flow: torch.Tensor, observed_flows_segmentation: torch.Tensor,
                     observed_flows_occlusion: torch.Tensor, observed_flows_uncertainty: torch.Tensor,
                     source_frame: int, target_frame: int) -> None:

        flow_observation = FlowObservation(observed_flow=observed_flow,
                                           observed_flow_segmentation=observed_flows_segmentation,
                                           observed_flow_uncertainty=observed_flows_uncertainty,
                                           observed_flow_occlusion=observed_flows_occlusion)

        self.add_new_flow_observation(flow_observation, source_frame, target_frame)

    def add_new_flow_observation(self, flow_observation: FlowObservation, source_frame: int, target_frame: int) -> None:
        if source_frame not in self.G:
            raise ValueError("source_frame not in the graph")
        elif target_frame not in self.G:
            raise ValueError("target_frame not in the graph")

        if source_frame not in self.flow_frames:
            insort(self.flow_frames, source_frame)
        self.G.add_edge(source_frame, target_frame, flow_observations=flow_observation)

    def get_flows_between_frames(self, source_frame: int, target_frame: int) -> FlowObservation:

        return self.G.get_edge_data(source_frame, target_frame)['flow_observations']

    def get_observations_for_all_keyframes(self, bounding_box: Tuple[int, int, int, int] = None) -> FrameObservation:
        vertices = list(filter(lambda vertex: vertex[0] in self.keyframes, self.G.nodes(data=True)))
        vertices.sort(key=lambda vertex: vertex[0])

        vertices_observations = [vertex[1]['observations'] for vertex in vertices]
        concatenated_observations = FrameObservation.concatenate(*vertices_observations)

        if bounding_box is not None:
            concatenated_observations = concatenated_observations.trim_bounding_box(bounding_box)

        return concatenated_observations

    def get_observations_for_keyframe(self, keyframe,
                                      bounding_box: Tuple[int, int, int, int] = None) -> FrameObservation:
        observations = self.G.nodes[keyframe]['observations']

        if bounding_box is not None:
            observations = observations.trim_bounding_box(bounding_box)

        return observations

    def get_flows_observations(self, bounding_box=None):
        arcs = list(self.G.edges(data=True))
        arcs.sort(key=lambda edge: edge[:2:-1])  # Sort by the target frame, then by the source frame

        flow_observations = [arc[2]['flow_observations'] for arc in arcs]
        concatenated_tensors = FlowObservation.concatenate(*flow_observations)

        if bounding_box is not None:
            concatenated_tensors = concatenated_tensors.trim_bounding_box(bounding_box)

        return concatenated_tensors

    def trim_keyframes(self, max_keyframes):
        if len(self.keyframes) > max_keyframes:
            # Keep only those last ones
            keep_keyframes = np.zeros(len(self.keyframes), dtype=bool)
            keep_keyframes[-max_keyframes:] = True

            return self.keep_selected_keyframes(keep_keyframes)
        else:
            return KeyframeBuffer()

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
