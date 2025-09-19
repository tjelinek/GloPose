#!/usr/bin/env python3
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from bop_toolkit_lib import pycoco_utils
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout


def evaluate_bop_coco(
        result_filename,
        results_path,
        eval_path=None,
        datasets_path=None,
        ann_type="bbox",
        bbox_type="amodal",
        targets_filename="test_targets_bop19.json",
        save_results=True
):
    """
    Evaluate BOP COCO results directly without calling external script.

    Args:
        result_filename: Name of the result file to evaluate
        results_path: Path to results directory
        eval_path: Path to evaluation output directory (optional)
        datasets_path: Path to datasets (optional, uses config default)
        ann_type: Annotation type ('bbox' or 'segm')
        bbox_type: Bbox type ('modal' or 'amodal')
        targets_filename: Target filename
        save_results: Whether to save results to file

    Returns:
        dict: Dictionary containing all evaluation metrics
    """

    # Import config for default paths if not provided
    from bop_toolkit_lib import config
    if datasets_path is None:
        datasets_path = config.datasets_path
    if eval_path is None:
        eval_path = config.eval_path

    print(f"Evaluating: {result_filename}")
    print(f"Annotation type: {ann_type}")

    # Parse info about the method and dataset from filename
    result_name, method, dataset, split, split_type, _ = inout.parse_result_filename(result_filename)
    print(f"Dataset: {dataset}, Split: {split}, Split type: {split_type}")
    print("-" * 50)

    # Load dataset parameters
    print(f"Loading dataset parameters from: {datasets_path}")
    dp_split = dataset_params.get_split_params(datasets_path, dataset, split, split_type)
    print(f"Base path from dataset params: {dp_split.get('base_path', 'NOT FOUND')}")

    model_type = "eval"
    dp_model = dataset_params.get_model_params(datasets_path, dataset, model_type)

    # Load and check results
    results_path_full = os.path.join(results_path, result_filename)

    print("Checking COCO results format...")
    check_passed, check_msg = inout.check_coco_results(results_path_full, ann_type=ann_type)
    if not check_passed:
        raise ValueError(f"COCO results check failed: {check_msg}")

    print("Loading COCO results...")
    coco_results = inout.load_json(results_path_full, keys_to_int=True)

    # Load estimation targets
    targets_path = os.path.join(dp_split["base_path"], targets_filename)
    print(f"Looking for targets file at: {targets_path}")

    if not os.path.exists(targets_path):
        print(f"Targets file not found at: {targets_path}")
        print(f"Base path: {dp_split['base_path']}")
        print(f"Contents of base path:")
        if os.path.exists(dp_split["base_path"]):
            for item in os.listdir(dp_split["base_path"]):
                print(f"  {item}")
        else:
            print("  Base path does not exist!")

        # Try alternative locations
        alternative_paths = [
            os.path.join(datasets_path, dataset, targets_filename),
            os.path.join(dp_split["base_path"], "..", targets_filename),
        ]

        targets_path = None
        for alt_path in alternative_paths:
            print(f"Trying alternative path: {alt_path}")
            if os.path.exists(alt_path):
                targets_path = alt_path
                print(f"Found targets file at: {targets_path}")
                break

        if targets_path is None:
            raise FileNotFoundError(f"Could not find targets file {targets_filename} in any expected location")

    targets = inout.load_json(targets_path)

    print("Organizing estimation targets...")
    targets_org = {}
    for target in targets:
        targets_org.setdefault(target["scene_id"], {}).setdefault(target["im_id"], {})

    print("Organizing estimation results...")
    results_org = {}
    for result in coco_results:
        if (ann_type == "bbox" and result.get("bbox")) or (
                ann_type == "segm" and result.get("segmentation")
        ):
            results_org.setdefault(result["scene_id"], []).append(result)

    if not results_org:
        print(f"No valid COCO results for annotation type: {ann_type}")
        return {res_type: -1.0 for res_type in get_result_types()} | {"average_time_per_image": -1.0}

    print("Merging COCO annotations and predictions...")

    # Merge COCO scene annotations and results
    for i, scene_id in enumerate(targets_org):
        tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)

        scene_coco_ann_path = dp_split[tpath_keys["scene_gt_coco_tpath"]].format(scene_id=scene_id)
        if ann_type == "bbox" and bbox_type == "modal":
            scene_coco_ann_path = scene_coco_ann_path.replace("scene_gt_coco", "scene_gt_coco_modal")

        scene_coco_ann = inout.load_json(scene_coco_ann_path, keys_to_int=True)
        scene_coco_results = results_org.get(scene_id, [])

        # Filter target image IDs
        target_img_ids = list(targets_org[scene_id].keys())
        scene_coco_ann["images"] = [
            img for img in scene_coco_ann["images"] if img["id"] in target_img_ids
        ]
        scene_coco_ann["annotations"] = [
            ann for ann in scene_coco_ann["annotations"] if ann["image_id"] in target_img_ids
        ]
        scene_coco_results = [
            res for res in scene_coco_results if res["image_id"] in target_img_ids
        ]

        if i == 0:
            dataset_coco_ann = scene_coco_ann
            dataset_coco_results = scene_coco_results
        else:
            dataset_coco_ann, image_id_offset = pycoco_utils.merge_coco_annotations(
                dataset_coco_ann, scene_coco_ann
            )
            dataset_coco_results = pycoco_utils.merge_coco_results(
                dataset_coco_results, scene_coco_results, image_id_offset
            )

    # Check consistent timings
    _, _, times, times_available = inout.check_consistent_timings(coco_results, "image_id")

    # Initialize COCO ground truth API
    print("Initializing COCO evaluation...")
    # COCO() expects a file path, but we have a dict, so we need to save it temporarily
    # or use the alternative initialization method
    import tempfile
    import json

    # Save the merged annotations to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(dataset_coco_ann, tmp_file)
        tmp_ann_path = tmp_file.name

    try:
        cocoGt = COCO(tmp_ann_path)
    except Exception as e:
        # Clean up temp file if COCO initialization fails
        os.unlink(tmp_ann_path)
        raise e

    if ann_type == "segm":
        pycoco_utils.ensure_rle_binary(dataset_coco_results, cocoGt)

    try:
        cocoDt = cocoGt.loadRes(dataset_coco_results)

        # Run evaluation
        print("Running COCO evaluation...")
        cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
        cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # Extract scores
        res_types = get_result_types()
        coco_scores = {res_types[i]: stat for i, stat in enumerate(cocoEval.stats)}
        coco_scores["average_time_per_image"] = np.mean(list(times.values())) if times_available else -1.0

    except IndexError as e:
        print(f"Error during evaluation: {e}")
        print("This might be due to empty results or mismatched scene_id/image_id pairs")
        res_types = get_result_types()
        coco_scores = {res_type: -1.0 for res_type in res_types}
        coco_scores["average_time_per_image"] = -1.0

    finally:
        # Clean up temporary file
        if 'tmp_ann_path' in locals():
            try:
                os.unlink(tmp_ann_path)
            except OSError:
                pass  # File might already be deleted

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS:")
    print("=" * 50)
    for key, value in coco_scores.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:8.4f}")
        else:
            print(f"{key:20s}: {value}")

    # Save results if requested
    if save_results and eval_path:
        os.makedirs(os.path.join(eval_path, result_name), exist_ok=True)
        scores_filename = f"scores_bop22_coco_{ann_type}.json"
        if ann_type == "bbox" and bbox_type == "modal":
            scores_filename = scores_filename.replace(".json", "_modal.json")

        final_scores_path = os.path.join(eval_path, result_name, scores_filename)
        inout.save_json(final_scores_path, coco_scores)
        print(f"\nResults saved to: {final_scores_path}")

    return coco_scores


def get_result_types():
    """Return the standard COCO evaluation metric names."""
    return [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"
    ]


# Example usage
if __name__ == "__main__":
    try:
        # Your parameters
        result_filename = "FlowTemplates_lmo-test_ufm_c0975r05@None@detection_thresh_05.json"
        results_path = "/mnt/personal/jelint19/results/PoseEstimation/"
        eval_path = "/mnt/personal/jelint19/results/PoseEstimation/bop_eval"

        # Run evaluation
        metrics = evaluate_bop_coco(
            result_filename=result_filename,
            results_path=results_path,
            eval_path=eval_path,
            ann_type="bbox",
            targets_filename="test_targets_bop19.json"
        )

        # Access individual metrics
        if metrics:
            ap = metrics.get('AP', -1)
            ap50 = metrics.get('AP50', -1)
            ap75 = metrics.get('AP75', -1)
            avg_time = metrics.get('average_time_per_image', -1)

            print(f"\nKey metrics extracted:")
            print(f"AP: {ap:.4f}")
            print(f"AP@0.5: {ap50:.4f}")
            print(f"AP@0.75: {ap75:.4f}")
            print(f"Avg time/image: {avg_time:.4f}s")
        else:
            print("Evaluation returned no metrics!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()