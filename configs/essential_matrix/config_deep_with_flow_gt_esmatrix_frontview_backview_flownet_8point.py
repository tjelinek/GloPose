from tracker_config import TrackerConfig


class TrackerConfig8P(TrackerConfig):
    def __init__(self):
        super(TrackerConfig8P, self).__init__()

        self.run_main_optimization_loop = False
        self.optimize_shape = False

        self.ransac_essential_matrix_algorithm = None

        self.ransac_use_gt_occlusions_and_segmentation = False
        self.ransac_dilate_occlusion = False
        self.ransac_erode_segmentation = True

        self.ransac_feed_only_inlier_flow = False
        self.ransac_replace_mft_flow_with_gt_flow = False
        self.ransac_feed_gt_flow_add_gaussian_noise = False
        self.ransac_feed_gt_flow_add_gaussian_noise_use_mft_errors = False
        self.ransac_distant_pixels_sampling = False
        self.ransac_confidences_from_occlusion = False
        self.ransac_inlier_pose_method = '8point'

        self.refine_pose_using_numerical_optimization = False


def get_config() -> TrackerConfig:

    return TrackerConfig8P()


