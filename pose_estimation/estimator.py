import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pycolmap
import torch
from kornia.geometry import Se3
from torchvision.transforms import functional as TF

from configs.glopose_config import PoseEstimationConfig
from data_providers.flow_provider import MatchingProvider
from data_structures.types import Detection, PoseEstimate, ObjectId
from data_structures.view_graph import ViewGraph
from onboarding.colmap_utils import get_image_Se3_world2cam, create_database_cache, PYCOLMAP4
from onboarding.frame_filter import compute_matching_reliability
from onboarding.reconstruction import unique_keypoints_from_matches
from visualizations.pose_estimation_visualizations import PoseEstimatorLogger


class PoseEstimator:

    def __init__(self, matching_provider: MatchingProvider, config: PoseEstimationConfig,
                 cache_folder: Path, device: str = 'cuda'):
        self.matching_provider = matching_provider
        self.config = config
        self.cache_folder = cache_folder
        self.device = device

    def estimate_poses(self, detections: list[Detection], view_graphs: dict[ObjectId, ViewGraph],
                       image: torch.Tensor, camera_K: np.ndarray,
                       pose_logger: PoseEstimatorLogger | None = None) -> list[PoseEstimate]:
        """Estimate 6DoF poses for each detection by registering into the corresponding ViewGraph's reconstruction.

        Args:
            detections: Detected objects in this image, each with object_id and mask.
            view_graphs: Mapping from object_id to ViewGraph (with COLMAP reconstruction).
            image: Query image tensor (C, H, W).
            camera_K: 3x3 camera intrinsics matrix.
            pose_logger: Optional logger for visualization.

        Returns:
            List of PoseEstimate for detections where registration succeeded.
        """
        results = []
        for det in detections:
            view_graph = view_graphs.get(det.object_id)
            if view_graph is None:
                continue
            if not getattr(view_graph, 'reconstruction_success', True):
                continue

            pose = self._estimate_single_pose(
                image, camera_K, view_graph, det.mask, pose_logger
            )
            if pose is not None:
                results.append(PoseEstimate(
                    object_id=det.object_id,
                    score=det.score,
                    Se3_obj2cam=pose,
                ))
        return results

    def _estimate_single_pose(self, query_img: torch.Tensor, camera_K: np.ndarray,
                               view_graph: ViewGraph, query_segmentation: torch.Tensor | None,
                               pose_logger: PoseEstimatorLogger | None) -> Se3 | None:
        """Match query against ViewGraph templates, register into COLMAP reconstruction, extract pose."""

        path_to_colmap_db = view_graph.colmap_db_path
        path_to_reconstruction = view_graph.colmap_reconstruction_path

        # Work on a temporary copy of the COLMAP database
        tmp_dir = self.cache_folder / 'pose_estimation_tmp' / str(view_graph.object_id)
        if tmp_dir.exists() and tmp_dir.is_dir():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)

        try:
            cache_db_file = tmp_dir / path_to_colmap_db.name
            shutil.copy(path_to_colmap_db, cache_db_file)
            return self._register_query_image(
                query_img, camera_K, view_graph, query_segmentation,
                cache_db_file, path_to_reconstruction, pose_logger
            )
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def _register_query_image(self, query_img: torch.Tensor, camera_K: np.ndarray,
                               view_graph: ViewGraph, query_segmentation: torch.Tensor | None,
                               cache_db_file: Path, path_to_reconstruction: Path,
                               pose_logger: PoseEstimatorLogger | None) -> Se3 | None:
        """Core registration: match against templates, write to COLMAP DB, run incremental mapper."""

        database = pycolmap.Database.open(str(cache_db_file))

        h, w = query_img.shape[-2:]
        f_x = float(camera_K[0, 0])
        f_y = float(camera_K[1, 1])
        c_x = float(camera_K[0, 2])
        c_y = float(camera_K[1, 2])

        n_cameras = database.num_cameras() if callable(database.num_cameras) else database.num_cameras
        new_camera_id = n_cameras + 1
        new_camera = pycolmap.Camera(
            camera_id=new_camera_id, model=pycolmap.CameraModelId.PINHOLE,
            width=w, height=h, params=[f_x, f_y, c_x, c_y]
        )

        n_images = database.num_images() if callable(database.num_images) else database.num_images
        new_image_id = n_images + 1
        new_database_image = pycolmap.Image(
            image_id=new_image_id, camera_id=new_camera_id, name='tmp_target'
        )

        database.write_camera(new_camera, use_camera_id=True)
        database.write_image(new_database_image, use_image_id=True)

        # Match query image against a subset of ViewGraph templates
        matching_edges, matching_edges_certainties = self._match_against_templates(
            query_img, query_segmentation, view_graph, database, new_image_id, pose_logger
        )

        if len(matching_edges) == 0:
            return None

        # Build unified keypoint/match tables for COLMAP
        from data_providers.matching_provider_sift import SparseMatchingProvider
        if isinstance(self.matching_provider, SparseMatchingProvider):
            self._write_matches_direct(matching_edges, database, new_image_id)
        else:
            self._write_matches_to_database(matching_edges, matching_edges_certainties,
                                             database, new_image_id)

        # Verify new matches via two-view geometry
        # Only verify pairs involving the new image (don't re-verify existing pairs)
        self._verify_new_matches(database, new_image_id)

        # Run incremental mapper to register the query image
        return self._run_mapper(database, new_camera, new_database_image, new_image_id,
                                path_to_reconstruction)

    def _match_against_templates(self, query_img: torch.Tensor, query_segmentation: torch.Tensor | None,
                                  view_graph: ViewGraph, database: pycolmap.Database,
                                  new_image_id: int, pose_logger: PoseEstimatorLogger | None,
                                  ) -> Tuple[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                                             Dict[Tuple[int, int], torch.Tensor]]:
        """Match query image against ViewGraph templates, filter by reliability."""

        matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = {}

        nodes = list(view_graph.view_graph.nodes())
        n = len(nodes)
        max_templates = self.config.max_templates_to_match
        if n > max_templates:
            k = max(1, n // max_templates)
            while n // k < max_templates:
                k -= 1
        else:
            k = 1
        selected_nodes = nodes[::k]

        for frame_idx in selected_nodes:
            view_graph_node = view_graph.get_node_data(frame_idx)
            db_img_id = view_graph_node.colmap_db_image_id

            template_image = view_graph_node.observation.observed_image.to(self.device).squeeze()
            template_segmentation = view_graph_node.observation.observed_segmentation.to(self.device).squeeze()

            template_keypoints_xy_np = database.read_keypoints(db_img_id).astype(np.int32)
            template_keypoints_yx = torch.tensor(template_keypoints_xy_np, device=self.device)[:, [1, 0]]

            template_keypoints_mask = torch.zeros_like(template_segmentation)
            template_keypoints_mask[template_keypoints_yx[:, 0], template_keypoints_yx[:, 1]] = 1.
            segmentation_template_points = template_segmentation * template_keypoints_mask

            if self.config.black_background:
                template_image = template_image * template_segmentation

            # Resize query to match template resolution
            query_img_resized = TF.resize(query_img, list(template_image.shape[-2:]))
            if query_segmentation is not None:
                query_seg_resized = TF.resize(
                    query_segmentation[None], list(template_segmentation.shape[-2:])
                ).squeeze()
                if self.config.black_background:
                    query_img_resized = query_img_resized * query_seg_resized.to(torch.float)
            else:
                query_seg_resized = None

            # For sparse matchers, pass the object segmentation (not keypoint mask) and don't
            # filter by foreground — SIFT needs all detected keypoints for reliable matching.
            # For dense matchers, the keypoint mask selects flow at database keypoint locations.
            from data_providers.matching_provider_sift import SparseMatchingProvider
            is_sparse = isinstance(self.matching_provider, SparseMatchingProvider)

            db_img_pts_xy, query_img_pts_xy, certainties = (
                self.matching_provider.get_source_target_points(
                    template_image, query_img_resized, None,
                    template_segmentation if is_sparse else segmentation_template_points,
                    query_seg_resized,
                    as_int=True, only_foreground_matches=not is_sparse
                )
            )

            if db_img_pts_xy.shape[0] == 0:
                continue

            # For sparse matchers (SIFT), snap template match points to nearest database
            # keypoints so they connect to existing 3D points in the reconstruction.
            db_img_pts_xy, query_img_pts_xy, certainties = self._snap_to_database_keypoints(
                db_img_pts_xy, query_img_pts_xy, certainties,
                template_keypoints_xy_np, max_distance=10.0
            )

            if db_img_pts_xy.shape[0] == 0:
                continue

            reliability = compute_matching_reliability(
                db_img_pts_xy, certainties, segmentation_template_points,
                self.config.min_certainty_threshold
            )

            if pose_logger is not None:
                pose_logger.visualize_pose_matching_rerun(
                    db_img_pts_xy, query_img_pts_xy, certainties,
                    template_image, query_img_resized, reliability,
                    self.config.flow_reliability_threshold,
                    self.config.min_certainty_threshold,
                    viewgraph_image_segment=template_segmentation,
                    query_image_segment=query_seg_resized
                )
                pose_logger.rerun_sequence_id += 1

            print(f'Mean certainty: {certainties.mean().item():.4f}, Reliability: {reliability:.4f}')
            if reliability >= self.config.flow_reliability_threshold:
                matching_edges[(new_image_id, db_img_id)] = (query_img_pts_xy, db_img_pts_xy)
                matching_edges_certainties[(new_image_id, db_img_id)] = certainties

        return matching_edges, matching_edges_certainties

    @staticmethod
    def _verify_new_matches(database: pycolmap.Database, new_image_id: int) -> None:
        """Verify matches involving the new image by writing TwoViewGeometry entries.

        Instead of re-running match_exhaustive (which would re-verify ALL pairs),
        we mark new pairs as verified with CALIBRATED configuration.
        """
        for img in database.read_all_images():
            if img.image_id == new_image_id:
                continue
            matches = database.read_matches(new_image_id, img.image_id)
            if len(matches) == 0:
                continue
            tvg = pycolmap.TwoViewGeometry()
            tvg.inlier_matches = matches
            tvg.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
            database.write_two_view_geometry(new_image_id, img.image_id, tvg)

    def _write_matches_direct(self, matching_edges, database: pycolmap.Database,
                               new_image_id: int) -> None:
        """Write matches directly using snapped database keypoint indices.

        For sparse matchers (SIFT), the template match points have been snapped to known
        database keypoint positions. We can directly look up their indices in the existing
        keypoint arrays instead of going through unique_keypoints_from_matches.
        """
        # Collect all query points across all template matches
        all_query_pts = []
        for (query_id, db_img_id), (query_pts, template_pts) in matching_edges.items():
            all_query_pts.append(query_pts)

        if not all_query_pts:
            return

        # Deduplicate query keypoints
        all_query_pts_cat = torch.cat(all_query_pts, dim=0)
        unique_query_pts, inverse_indices = torch.unique(all_query_pts_cat, dim=0, return_inverse=True)

        # Write query keypoints
        query_kps_np = unique_query_pts.numpy(force=True).astype(np.float32)
        database.write_keypoints(new_image_id, query_kps_np)

        # For each edge, find match indices
        offset = 0
        for (query_id, db_img_id), (query_pts, template_pts) in matching_edges.items():
            n = query_pts.shape[0]
            query_kp_indices = inverse_indices[offset:offset + n]
            offset += n

            # Find the database keypoint indices for snapped template points
            db_keypoints = database.read_keypoints(db_img_id)
            template_pts_np = template_pts.numpy(force=True).astype(np.float32)

            # Match snapped points to database keypoints by coordinate lookup
            db_kp_indices = []
            for pt in template_pts_np:
                # Find exact or nearest match in database keypoints
                dists = np.sqrt(((db_keypoints - pt) ** 2).sum(axis=1))
                db_kp_indices.append(dists.argmin())

            db_kp_indices = np.array(db_kp_indices, dtype=np.uint32)
            query_kp_indices_np = query_kp_indices.numpy(force=True).astype(np.uint32)

            matches = np.stack([query_kp_indices_np, db_kp_indices], axis=1)
            if matches.shape[0] > 0:
                database.write_matches(new_image_id, db_img_id, matches)

    @staticmethod
    def _snap_to_database_keypoints(template_pts_xy: torch.Tensor, query_pts_xy: torch.Tensor,
                                     certainties: torch.Tensor, db_keypoints_xy: np.ndarray,
                                     max_distance: float = 10.0
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Snap matched template points to their nearest database keypoints.

        SIFT detects its own keypoints which are at different locations from the COLMAP database
        keypoints. To establish 2D-3D correspondences, we replace each SIFT template point with
        the nearest database keypoint (if within max_distance pixels).
        """
        if db_keypoints_xy.shape[0] == 0 or template_pts_xy.shape[0] == 0:
            return template_pts_xy[:0], query_pts_xy[:0], certainties[:0]

        # Compute pairwise distances between SIFT template points and database keypoints
        template_pts_np = template_pts_xy.cpu().numpy().astype(np.float32)
        db_kps = db_keypoints_xy.astype(np.float32)

        # (N_matches, N_db_kps) distance matrix
        diff = template_pts_np[:, None, :] - db_kps[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))

        # For each match, find nearest database keypoint
        nearest_idx = dists.argmin(axis=1)
        nearest_dist = dists[np.arange(len(dists)), nearest_idx]

        # Keep only matches close enough to a database keypoint
        keep = nearest_dist < max_distance
        if keep.sum() == 0:
            device = template_pts_xy.device
            return (torch.zeros((0, 2), device=device, dtype=template_pts_xy.dtype),
                    torch.zeros((0, 2), device=device, dtype=query_pts_xy.dtype),
                    torch.zeros(0, device=device, dtype=certainties.dtype))

        # Replace template points with exact database keypoint locations
        snapped_template_pts = torch.tensor(
            db_kps[nearest_idx[keep]], device=template_pts_xy.device, dtype=template_pts_xy.dtype)

        return snapped_template_pts, query_pts_xy[keep], certainties[keep]

    def _write_matches_to_database(self, matching_edges, matching_edges_certainties,
                                    database: pycolmap.Database, new_image_id: int) -> None:
        """Compute unique keypoints and write matches into the COLMAP database."""

        keypoints, edge_match_indices = unique_keypoints_from_matches(
            matching_edges, database, eliminate_one_to_many_matches=True,
            matching_edges_certainties=matching_edges_certainties, device=self.device
        )

        all_image_ids = {img.image_id for img in database.read_all_images()}
        matched_images_ids = {node for edge in edge_match_indices.keys() for node in edge}
        non_matched_images_ids = all_image_ids - matched_images_ids
        non_matched_keypoints = {img_id: database.read_keypoints(img_id) for img_id in non_matched_images_ids}

        old_keypoints = {}
        for colmap_image_id in set(keypoints.keys()) - {new_image_id}:
            old_keypoints[colmap_image_id] = database.read_keypoints(colmap_image_id)

        database.clear_keypoints()
        for colmap_image_id in sorted(keypoints.keys()):
            if colmap_image_id == new_image_id:
                keypoints_np = keypoints[colmap_image_id].numpy(force=True).astype(np.float32)
            else:
                keypoints_np = old_keypoints[colmap_image_id]
            database.write_keypoints(colmap_image_id, keypoints_np)

        for (colmap_image_u, colmap_image_v), match_indices in edge_match_indices.items():
            match_indices_np = match_indices.numpy(force=True).reshape(-1, 2).astype(np.uint32)
            if match_indices_np.shape[0] == 0:
                continue
            keypoints_u = database.read_keypoints(colmap_image_u)
            keypoints_v = database.read_keypoints(colmap_image_v)
            valid_mask = ((match_indices_np[:, 0] < len(keypoints_u)) &
                          (match_indices_np[:, 1] < len(keypoints_v)))
            valid_matches = match_indices_np[valid_mask]
            if valid_matches.shape[0] > 0:
                database.write_matches(colmap_image_u, colmap_image_v, valid_matches)

        for img_id, kps in non_matched_keypoints.items():
            database.write_keypoints(img_id, kps)

    @staticmethod
    def _run_mapper(database: pycolmap.Database, new_camera: pycolmap.Camera,
                    new_database_image: pycolmap.Image, new_image_id: int,
                    path_to_reconstruction: Path) -> Se3 | None:
        """Load existing reconstruction, register the query image, extract pose."""

        database_cache = create_database_cache(database)

        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(str(path_to_reconstruction))
        reconstruction.add_camera(new_camera)

        if PYCOLMAP4:
            # pycolmap 4.0: images need Frame→Rig chain
            cam = new_camera
            rig_id = cam.camera_id
            if rig_id not in reconstruction.rigs:
                rig = pycolmap.Rig(rig_id=rig_id)
                rig.add_ref_sensor(cam.sensor_id)
                reconstruction.add_rig(rig)

            new_database_image.frame_id = new_image_id
            frame = pycolmap.Frame(frame_id=new_image_id, rig_id=rig_id)
            frame.add_data_id(new_database_image.data_id)
            reconstruction.add_frame(frame)

        reconstruction.add_image(new_database_image)

        mapper = pycolmap.IncrementalMapper(database_cache)
        mapper.begin_reconstruction(reconstruction)
        mapper_options = pycolmap.IncrementalMapperOptions()

        print(f"Registering image #{new_image_id}")
        print(f"=> Image sees {mapper.observation_manager.num_visible_points3D(new_image_id)} / "
              f"{mapper.observation_manager.num_observations(new_image_id)} points")

        success = mapper.register_next_image(mapper_options, new_image_id)

        if success:
            print(f"Successfully registered image {new_image_id} into the reconstruction.")
            tri_options = (mapper_options.triangulation if hasattr(mapper_options, 'triangulation')
                           else pycolmap.IncrementalTriangulatorOptions())
            mapper.triangulate_image(tri_options, new_image_id)
            registered_image = reconstruction.images[new_image_id]
            return get_image_Se3_world2cam(registered_image, 'cpu')
        else:
            print(f"Failed to register image {new_image_id}.")
            return None
