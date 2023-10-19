import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from typing import Set, Tuple, List

from tracking6d import Observations


@dataclass
class KeyframeBuffer:
    keyframes: list = None  # Ordinal of a frame that we optimize for shape etc.
    flow_frames: list = None  # Ordinal of a frame that we may not optimize but serve as a keyframe source
    # Pairs (f1, f2), where f1 is in flow_keyframes, and f2 in keyframes
    flow_arcs: Set[Tuple[int, int]] = field(default_factory=set)
    images: torch.Tensor = None
    prev_images: torch.Tensor = None
    images_feat: torch.Tensor = None
    segments: torch.Tensor = None
    observed_flows: List[torch.Tensor] = None
    observed_flows_segmentations: List[torch.Tensor] = None
    observed_flows_occlusions: List[torch.Tensor] = None
    observed_flows_uncertainties: List[torch.Tensor] = None

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
        if buffer1.keyframes is None and buffer2.keyframes is None:
            return KeyframeBuffer(), [], []
        elif buffer1.keyframes is None or (buffer1.keyframes is not None and len(buffer1.keyframes) == 0):
            return copy.deepcopy(buffer2), [], [k for k in
                                                range(len(buffer2.keyframes))] if buffer2.keyframes is not None else []
        elif buffer2.keyframes is None or (buffer2.keyframes is not None and len(buffer2.keyframes) == 0):
            return copy.deepcopy(buffer1), [k for k in
                                            range(len(buffer1.keyframes))] if buffer1.keyframes is not None else [], []

        all_keyframes = sorted(set(buffer1.keyframes + buffer2.keyframes))
        all_flow_frames = sorted(set(buffer1.flow_frames + buffer2.flow_frames))
        all_flow_arcs = buffer1.flow_arcs | buffer2.flow_arcs

        merged_buffer = KeyframeBuffer()
        merged_buffer.keyframes = all_keyframes
        merged_buffer.flow_frames = all_flow_frames
        merged_buffer.flow_arcs = all_flow_arcs

        indices_buffer1 = []
        indices_buffer2 = []

        for attr_name, attr_type in merged_buffer.__annotations__.items():
            merged_attr = None
            if attr_type is list:
                merged_attr = [getattr(buffer1, attr_name)[buffer1.keyframes.index(k)] if k in buffer1.keyframes else
                               getattr(buffer2, attr_name)[buffer2.keyframes.index(k)]
                               for k in all_keyframes]
            elif attr_type is torch.Tensor:
                attr1 = getattr(buffer1, attr_name)
                attr2 = getattr(buffer2, attr_name)
                merged_attr = torch.cat(
                    [attr1[:, buffer1.keyframes.index(k)].unsqueeze(1) if k in buffer1.keyframes else
                     attr2[:, buffer2.keyframes.index(k)].unsqueeze(1)
                     for k in all_keyframes], dim=1)
            setattr(merged_buffer, attr_name, merged_attr)

        # Track indices of keyframes in the merged buffer
        indices_buffer1.extend([buffer1.keyframes.index(k) for k in all_keyframes if k in buffer1.keyframes])
        indices_buffer2.extend([buffer2.keyframes.index(k) for k in all_keyframes if k in buffer2.keyframes])

        return merged_buffer, indices_buffer1, indices_buffer2

    def append_new_keyframe(self, observed_image, observed_image_features, observed_image_segmentation,
                            previously_observed_image, keyframe_index):
        self.images = torch.cat((self.images, observed_image), 1)
        self.images_feat = torch.cat((self.images_feat, observed_image_features), 1)
        self.segments = torch.cat((self.segments, observed_image_segmentation), 1)
        self.prev_images = torch.cat((self.prev_images, previously_observed_image), dim=1)

        self.observed_flows += [torch.empty(1, 0, 2, *self.prev_images.shape[-2:]).cuda()]
        self.observed_flows_segmentations += [torch.empty(1, 0, 1, *self.prev_images.shape[-2:]).cuda()]
        self.observed_flows_occlusions += [torch.empty(1, 0, 1, *self.prev_images.shape[-2:]).cuda()]
        self.observed_flows_uncertainties += [torch.empty(1, 0, 1, *self.prev_images.shape[-2:]).cuda()]

        self.keyframes += [keyframe_index]
        self.flow_frames += [keyframe_index - 1]

    def add_new_flow(self, observed_flows: torch.Tensor, observed_flows_segmentations: torch.Tensor,
                     observed_flows_occlusions: torch.Tensor, observed_flows_uncertainties: torch.Tensor,
                     source_frame: int, target_frame: int) -> None:
        if source_frame not in self.flow_frames:
            raise ValueError("Requested frame must be in frames sources self.flow_keyframes_sources")
        frame_source_idx = self.flow_frames.index(source_frame)
        self.flow_arcs |= {(source_frame, target_frame)}

        self.observed_flows[frame_source_idx] = \
            torch.cat([self.observed_flows[frame_source_idx], observed_flows], dim=1)
        self.observed_flows_segmentations[frame_source_idx] = \
            torch.cat([self.observed_flows_segmentations[frame_source_idx], observed_flows_segmentations], dim=1)
        self.observed_flows_occlusions[frame_source_idx] = \
            torch.cat([self.observed_flows_occlusions[frame_source_idx], observed_flows_occlusions], dim=1)
        self.observed_flows_uncertainties[frame_source_idx] = \
            torch.cat([self.observed_flows_uncertainties[frame_source_idx], observed_flows_uncertainties], dim=1)

    def get_flows_from_frame(self, source_frame: int):
        source_frame_idx = self.flow_frames.index(source_frame)
        observed_flows = self.observed_flows[source_frame_idx]
        observed_flows_segmentations = self.observed_flows_segmentations[source_frame_idx]
        observed_flows_occlusions = self.observed_flows_occlusions[source_frame_idx]
        observed_flows_uncertainties = self.observed_flows_uncertainties[source_frame_idx]

        return observed_flows, observed_flows_segmentations, observed_flows_uncertainties, observed_flows_occlusions

    def get_observations(self, bounding_box: Tuple[int, int, int, int] = None) -> Observations:

        observed_flows, observed_flows_segmentations, observed_flows_uncertainties, \
            observed_flows_occlusions = self.get_flows_observations(bounding_box)

        observed_images = self.images
        observed_segmentations = self.segments
        if bounding_box is not None:
            observed_images, observed_segmentations = self.trim_bounding_box(bounding_box, observed_images,
                                                                             observed_segmentations)
        observations = Observations(
            observed_images=observed_images.clone(),
            observed_segmentations=observed_segmentations.clone(),
            observed_flows=observed_flows.clone(),
            observed_flows_segmentations=observed_flows_segmentations.clone(),
            observed_flows_occlusions=observed_flows_occlusions.clone(),
            observed_flows_uncertainties=observed_flows_uncertainties.clone()
        )

        return observations

    def get_flows_observations(self, bounding_box=None):
        observed_flows = torch.cat(self.observed_flows, dim=1)
        observed_flows_segmentations = torch.cat(self.observed_flows_segmentations, dim=1)
        observed_flows_occlusions = torch.cat(self.observed_flows_occlusions, dim=1)
        observed_flows_uncertainties = torch.cat(self.observed_flows_uncertainties, dim=1)

        if bounding_box is not None:
            trimmed = self.trim_bounding_box(bounding_box, observed_flows, observed_flows_occlusions,
                                             observed_flows_segmentations, observed_flows_uncertainties)
            observed_flows, observed_flows_occlusions, observed_flows_segmentations, \
                observed_flows_uncertainties = trimmed

        return observed_flows, observed_flows_segmentations, observed_flows_uncertainties, observed_flows_occlusions

    @staticmethod
    def trim_bounding_box(bounding_box: Tuple[int, int, int, int], *args):
        trimmed_args = []
        for arg in args:
            trimmed_arg = arg[:, :, :, bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
            trimmed_args.append(trimmed_arg)
        return tuple(trimmed_args)

    def trim_keyframes(self, max_keyframes):
        if len(self.keyframes) > max_keyframes:
            # Keep only those last ones
            keep_keyframes = np.zeros(len(self.keyframes), dtype=bool)
            keep_keyframes[-max_keyframes:] = True

            return self.keep_selected_keyframes(keep_keyframes)
        else:
            return KeyframeBuffer()

    def __copy_selected_keyframes_data_to_buffer(self, keyframe_buffer_instance, to_be_copied_attributes):
        for attr_name, attr_type in keyframe_buffer_instance.__annotations__.items():
            if attr_type is list:
                modified_attr = (np.array(getattr(self, attr_name))[to_be_copied_attributes]).tolist()
                setattr(keyframe_buffer_instance, attr_name, modified_attr)
            elif attr_type is torch.Tensor:
                modified_attr = getattr(self, attr_name)[:, to_be_copied_attributes]
                setattr(keyframe_buffer_instance, attr_name, modified_attr)

    def keep_selected_keyframes(self, keep_keyframes):
        not_keep_keyframes = ~ keep_keyframes

        # Get the deleted keyframes
        deleted_buffer = KeyframeBuffer()
        self.__copy_selected_keyframes_data_to_buffer(deleted_buffer, not_keep_keyframes)

        flow_frames_arcs_new = []
        for source_keyframe, target_keyframe in self.flow_arcs:

            source_keyframe_idx = self.flow_frames.index(source_keyframe)
            target_keyframe_idx = self.keyframes.index(target_keyframe)

            if not keep_keyframes[target_keyframe_idx]:
                attributes_to_modify = [
                    self.observed_flows,
                    self.observed_flows_segmentations,
                    self.observed_flows_uncertainties,
                    self.observed_flows_occlusions,
                ]

                for flow_attribute in attributes_to_modify:
                    flow_attribute_at_idx = flow_attribute[source_keyframe_idx]
                    modified_attr = torch.cat(
                        (flow_attribute_at_idx[:, :target_keyframe_idx],
                         flow_attribute_at_idx[:, target_keyframe_idx + 1:]), dim=1)
                    flow_attribute[source_keyframe_idx] = modified_attr
            else:
                flow_frames_arcs_new.append((source_keyframe, target_keyframe))

        self.flow_arcs = set(flow_frames_arcs_new)

        # This will retain only those observations corresponding to keyframes where keep_keyframes is one
        self.__copy_selected_keyframes_data_to_buffer(self, keep_keyframes)

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
