from bisect import insort

from typing import Tuple, List

import numpy as np
import torch
import networkx as nx

from dataclasses import dataclass, field


@dataclass
class Observation:

    def trim_bounding_box(self, bounding_box: Tuple[int, int, int, int]):
        for attr_name, attr_type in self.__annotations__.items():
            to_trim = getattr(self, attr_name)
            trimmed = to_trim[:, :, :, bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
            setattr(self, attr_name, trimmed)


@dataclass
class FrameObservation(Observation):
    observed_image: torch.Tensor = None
    observed_image_features: torch.Tensor = None
    observed_segmentation: torch.Tensor = None

    @staticmethod
    def concatenate(*observations):
        concatenated_observations = FrameObservation()

        for attr_name, attr_type in FrameObservation.__annotations__.items():
            to_concatenate = [getattr(observation, attr_name) for observation in observations]
            concatenated_attr = torch.cat(to_concatenate, dim=1)
            setattr(concatenated_observations, attr_name, concatenated_attr)

        return concatenated_observations


@dataclass
class FlowObservation(Observation):
    observed_flow: torch.Tensor = None
    observed_flow_segmentation: torch.Tensor = None
    observed_flow_occlusion: torch.Tensor = None
    observed_flow_uncertainty: torch.Tensor = None

    @staticmethod
    def concatenate(*observations: 'FlowObservation') -> 'FlowObservation':
        concatenated_observations = FlowObservation()

        for attr_name, attr_type in FlowObservation.__annotations__.items():
            concatenated_attr = torch.cat([getattr(observation, attr_name) for observation in observations], dim=1)
            setattr(concatenated_observations, attr_name, concatenated_attr)
        return concatenated_observations


@dataclass
class KeyframeBuffer:
    G: nx.DiGraph = field(default_factory=nx.DiGraph)

    # Contains ordinals of a frame that we optimize for shape etc.
    keyframes: list = field(default_factory=list)
    # Contains ordinals of a frame that we may not optimize but serve as a keyframe source
    flow_frames: list = field(default_factory=list)

    @staticmethod
    def merge(buffer1, buffer2):
        # TODO
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

    def add_new_keyframe(self, observed_image, observed_image_features, observed_image_segmentation, keyframe_index):

        frame_observation = FrameObservation(observed_image=observed_image,
                                             observed_image_features=observed_image_features,
                                             observed_segmentation=observed_image_segmentation)

        self.add_new_keyframe_observation(frame_observation, keyframe_index)

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

    def get_flows_from_frame(self, source_frame: int) -> FlowObservation:

        outgoing_arcs = list(self.G.edges(source_frame))
        outgoing_arcs.sort(key=lambda edge: edge[:2:-1])  # Sort by target frame
        outgoing_arcs_observations = [arc['flow_observations'] for arc in outgoing_arcs]

        concatenated_tensors = FlowObservation.concatenate(*outgoing_arcs_observations)

        return concatenated_tensors

    def get_flows_between_frames(self, source_frame: int, target_frame: int) -> FlowObservation:

        return self.G.get_edge_data(source_frame, target_frame)['flow_observations']

    def get_observations_for_all_keyframes(self, bounding_box: Tuple[int, int, int, int] = None) -> FrameObservation:
        vertices = list(filter(lambda vertex: vertex[0] in self.keyframes, self.G.nodes(data=True)))
        vertices.sort(key=lambda vertex: vertex[0])

        vertices_observations = [vertex[1]['observations'] for vertex in vertices]
        concatenated_observations = FrameObservation.concatenate(*vertices_observations)

        if bounding_box is not None:
            concatenated_observations.trim_bounding_box(bounding_box)

        return concatenated_observations

    def get_observations_for_keyframe(self, keyframe,
                                      bounding_box: Tuple[int, int, int, int] = None) -> FrameObservation:
        observations = self.G.nodes[keyframe]['observations']

        if bounding_box is not None:
            observations = observations.trim_bounding_box(bounding_box)

        return observations

    def get_flow_frames_for_keyframe(self, keyframe):
        return sorted(self.G.predecessors(keyframe))

    def get_flows_observations(self, bounding_box=None):
        arcs = list(self.G.edges(data=True))
        arcs.sort(key=lambda edge: edge[:2:-1])  # Sort by the target frame, then by the source frame

        flow_observations = [arc[2]['flow_observations'] for arc in arcs]
        concatenated_tensors = FlowObservation.concatenate(*flow_observations)

        if bounding_box is not None:
            concatenated_tensors.trim_bounding_box(bounding_box)

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

    def stochastic_update(self, max_keyframes):
        N = len(self.keyframes)
        if len(self.keyframes) > max_keyframes:
            keep_keyframes = np.full(N, False)  # Create an array of length N initialized with False
            indices = np.random.choice(N, max_keyframes, replace=False)  # Randomly select keep_keyframes indices
            keep_keyframes[indices] = True  # Set the selected indices to True

            return self.keep_selected_keyframes(keep_keyframes)
        else:
            return KeyframeBuffer()  # No items removed, return an empty buffer
