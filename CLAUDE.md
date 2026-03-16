# CLAUDE.md - GloPose

## Project Overview

GloPose is a model-free 6DoF object pose estimation system following
the [BOP benchmark paradigm](https://bop.felk.cvut.cz/home/). It takes a video/image stream of a novel object and
produces posed 3D reconstructions and detection-ready representations, which are then used to detect and estimate 6DoF
poses of the object in novel images.

## Setup

- **Environment:** Conda (`environment.yml`), env name `glopose`, Python 3.13
- **Install:** `conda env create -f environment.yml && conda activate glopose`
- **Submodules:** `repositories/` contains git submodules (cnos, mast3r, SAM2, vggt, ho3d) — some installed as
  editable pip packages

## Running

- **Web UI:** `python app.py` (Gradio on localhost:7860)
- **Dataset scripts:** `python run_HANDAL.py --config configs/base_config.py --sequences SEQ --experiment EXP` (
  similarly for run_HO3D.py, run_NAVI.py, run_BOP_classic_onboarding.py, etc.)
- **Batch jobs:** `python scripts/job_runner.py`
- **Check job status:** `python scripts/check_experiment_status.py [--show-missing] [config_names...]`
- **Collect results:** `python scripts/collect_experiment_results.py [--per-sequence] [--csv out.csv] [--dataset name] [config_names...]`

## Testing

No formal test suite. Validation is done by running dataset-specific scripts and evaluation utilities
(`eval/eval_onboarding.py`, `eval/eval_reconstruction.py`, `eval/eval_point_cloud.py`,
`eval/eval_bop_detection.py`, `utils/bop_challenge.py`).

## Architecture

### Target architecture: three modules

The system has three independent modules that communicate through well-defined data interfaces:

```
┌─────────────────────────────────────────────────────────────────────┐
│  A. ONBOARDING                                                      │
│  Input:  image stream / video of a novel object                     │
│  Steps:  segmentation → keyframe selection → SfM reconstruction     │
│  Output: OnboardingResult                                           │
│          ├── ViewGraph (posed keyframes + DINOv2 descriptors)       │
│          ├── 3D model (COLMAP reconstruction / mesh)                │
│          └── object_id                                              │
├─────────────────────────────────────────────────────────────────────┤
│  B. DETECTION                                                       │
│  B1. Representation building (offline)                              │
│      Input:  OnboardingResult (ViewGraph / condensed templates)     │
│      Output: DetectionModel                                         │
│              ├── TemplateBank (images, descriptors, masks)          │
│              ├── statistical params (whitening, CSLS, thresholds)   │
│              └── object_id                                          │
│                                                                     │
│  B2. Inference (online)                                             │
│      Input:  novel image + DetectionModel                           │
│      Output: List[Detection]                                        │
│              ├── bbox, segmentation mask, confidence score           │
│              └── object_id, matched_template_id                     │
├─────────────────────────────────────────────────────────────────────┤
│  C. POSE ESTIMATION                                                 │
│  Input:  List[Detection] + OnboardingResult (posed keyframes/model) │
│  Steps:  template matching → PnP / flow-based alignment             │
│  Output: List[PoseEstimate]                                         │
│          ├── Se3 (6DoF pose)                                        │
│          ├── confidence score                                       │
│          └── object_id                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Current state vs target

| Module             | Current location                                                                                           | Status                                                                                                                         |
|--------------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| A. Onboarding      | `onboarding/pipeline.py` (orchestrator), `onboarding/reconstruction.py` (SfM), `onboarding/frame_filter.py`, `data_providers/` | Working. Evaluation extracted to `eval/` module. `OnboardingPipeline` still tightly coupled to all providers.                  |
| B1. Representation | `detection/representation.py` (condensation), `data_structures/view_graph.py` (ViewGraph)                  | Working. `TemplateBank` dataclass in `data_structures/template_bank.py`. No clean `build_detection_model()` interface yet.     |
| B2. Detection      | `detection/detector.py` (`BOPChallengePosePredictor`), `detection/nms.py` (`DetectionContainer`)           | Working. Runtime deps vendored (no Hydra/cnos at runtime). Config via `DetectionConfig`. Tangled with BOP I/O.                |
| C. Pose estimation | `pose_estimation/estimator.py` (`PoseEstimator`)                                                           | Working. Matches query against ViewGraph templates, registers into COLMAP reconstruction, extracts 6DoF pose. Not yet evaluated on RCI. |

### Key data types (interface boundaries)

- **`ViewGraph`** (`data_structures/view_graph.py`): Per-object posed keyframes + COLMAP reconstruction. Primary
  onboarding output.
- **`TemplateBank`** (`data_structures/template_bank.py`): Condensed detection-ready templates with descriptors and statistical
  params. Bridge between onboarding and detection.
- **`Detection`** (`data_structures/types.py`): Our own scored bbox + mask type at the module boundary.
  `DetectionContainer` (`detection/nms.py`) provides NMS methods, created via `adapters.cnos_adapter.make_detections()`.
- **`PoseEstimate`** (`data_structures/types.py`): 6DoF pose estimate with confidence. Output of pose estimation module.
- **`DataGraph`** (`data_structures/data_graph.py`): Internal to onboarding — tracks per-frame data and cross-frame
  relationships during processing.

### Dataset formats & comparison methods

Detailed documentation of all dataset folder layouts, annotation schemas, image naming conventions,
camera intrinsics formats, GT structures, and external method APIs lives in
[`docs/dataset_formats.md`](docs/dataset_formats.md).

### Key directories

- `adapters/` — External repository wrappers and vendored code: `cnos_adapter.py` (descriptor extractor protocol),
  `dino_descriptor.py` (vendored DINOv2/v3 model), `dino_utils.py` (vendored utilities), `sam2_adapter.py` (SAM2)
- `configs/` — Python-based config files (not YAML), loaded via `utils.general.load_config()`
- `onboarding/` — OnboardingPipeline, SfM reconstruction, frame filtering, COLMAP utils
- `detection/` — template condensation (representation building), detector (inference), scoring, NMS (`nms.py`)
- `pose_estimation/` — `PoseEstimator`: flow-based matching against ViewGraph templates → COLMAP registration → 6DoF pose
- `eval/` — Evaluation module: `eval_onboarding.py` (wiring + GT mesh unit mapping), `eval_reconstruction.py`
  (per-keyframe/sequence/dataset CSV stats + AUC), `eval_point_cloud.py` (3D reconstruction quality: accuracy,
  completeness, F-score; pose AUC adapted from VGGT)
- `data_providers/` — Frame, flow, depth, and matching providers (abstract + implementations)
- `data_structures/` — ViewGraph, DataGraph, KeyframeBuffer, Detection, PoseEstimate, observations
- `models/` — Mesh rendering (Kaolin), feature encoding
- `utils/` — Dataset sequences, math (SE(3)), image I/O, results logging, mask/bbox utils
- `visualizations/` — Flow and pose visualization helpers, shared rerun utilities (`rerun_utils.py`)
- `scripts/` — Dataset downloaders, evaluation, job runners, experiment status/results collection
- `repositories/` — External dependency submodules (do not edit)

### Key patterns

- **Provider pattern:** Abstract base classes (FrameProvider, FlowProvider, MatchingProvider) with multiple backends
- **Graph-based data:** DataGraph (frames + temporal edges), ViewGraph (3D template library)
- **Dataclass configs:** `GloPoseConfig` in `configs/glopose_config.py` is the top-level config, composed of sub-configs: `PathsConfig`, `RunConfig`, `InputConfig`, `OnboardingConfig`, `CondensationConfig`, `DetectionConfig`, `PoseEstimationConfig`, `VisualizationConfig`, `RendererConfig`
- **Narrowed config passing:** Lower-level components receive only the sub-config(s) they need, not the full
  `GloPoseConfig`. For example: `RenderingKaolin` and `Encoder` take `RendererConfig`; frame filters and matching
  providers take `OnboardingConfig` + `device`; `PoseEstimator` takes `PoseEstimationConfig`;
  evaluation functions take `RunConfig` + `BaseBOPConfig`.
  Orchestrators (`OnboardingPipeline`, `FrameProviderAll`, `BOPChallengePosePredictor`, `run_*.py` scripts) still
  hold the full `GloPoseConfig` and pass the relevant sub-configs down.
- **Observation types:** FrameObservation, FlowObservation encapsulate per-frame and cross-frame data

### Cross-cutting concerns

- **DINOv2 descriptors** are centralized behind `adapters/cnos_adapter.py` (`DescriptorExtractor` protocol,
  `GloPoseDescriptorExtractor` implementation). The descriptor model is vendored in `adapters/dino_descriptor.py`
  (`CustomDINOv2` as `nn.Module`, loaded via `descriptor_from_config()` — no Hydra dependency).
  All consumers use `create_descriptor_extractor()` or pass a `DescriptorExtractor` instance.
  `sys.path` manipulation for cnos is kept solely for pickle compatibility with old ViewGraph caches.
- **Detection config** — all matching/postprocessing parameters (thresholds, NMS, OOD) are fields in
  `DetectionConfig` dataclass. No Hydra config loading at runtime. `BOPChallengePosePredictor` reads
  `self.config.detection.*` directly.
- **NMS** is handled by `DetectionContainer` (`detection/nms.py`), created via `make_detections()`.
  Replaces cnos `Detections` class. Our own `Detection` type in `data_structures/types.py` is used at
  module boundaries.
- **CNOS baseline** — original CNOS can be run via `scripts/run_cnos_baseline.py` for comparison.
  ViewGraph export to CNOS format is available as `export_viewgraphs_to_cnos_format()` in
  `data_structures/view_graph.py`.
- **BOP dataset conventions** (folder layout, splits, annotations) are known by `detection/detector.py`,
  `detection/representation.py`, and many `run_*.py` scripts. Should be encapsulated in a BOP data adapter.
- **Experiment management:**
  - `experiment_tracker.md` — central log of all experiments with status and results tables
  - `scripts/job_runner.py` exposes `get_configurations()`, `get_sequences()`, `get_results_root()` for
    use by status/results scripts
  - `scripts/check_experiment_status.py` — compares expected vs actual results per experiment
  - `scripts/collect_experiment_results.py` — aggregates reconstruction metrics across experiments
  - `skills/experiment/SKILL.md` — project-local Claude skill for the `/experiment` workflow

## Code Conventions

- **Type hints:** Used extensively, modern union syntax (`X | Y`)
- **Naming:** snake_case for functions/variables, PascalCase for classes
- **Imports:** Relative within package (`from configs.glopose_config import GloPoseConfig`; or specific sub-configs
  like `from configs.glopose_config import RendererConfig, OnboardingConfig`), absolute for externals
- **Strings:** f-strings preferred
- **No linter/formatter configured** — match existing code style when making changes
- **Dataclasses** over plain dicts for structured data

## Key Dependencies

PyTorch, Kornia (geometry/camera), Kaolin (mesh rendering), pycolmap, SAM2, RoMa/UFM (optical flow), DINOv2 (via torch.hub),
NetworkX, Gradio, Rerun SDK, wandb, scipy (KDTree for point cloud eval), trimesh (mesh loading)

- **pycolmap API docs**: [`docs/pycolmap_api.md`](docs/pycolmap_api.md) (downloaded from
  [colmap.github.io/pycolmap](https://colmap.github.io/pycolmap/pycolmap.html))

## Things to Know

- Configs are **Python files**, not YAML/JSON — they define variables that get loaded dynamically
- The `repositories/` directory contains external code — do not modify these
- Results go to `{results_root}/{experiment}/{dataset}/{sequence}/` with COLMAP database, reconstructions, and pose
  estimates
- Rerun SDK is used for 3D visualization logging alongside disk output
- Many operations are GPU-intensive; device selection is configurable via `GloPoseConfig.run.device`
- Hardcoded RCI paths are scattered across some dataset generators and scripts. The `GloPoseConfig.paths`
  sub-config (`PathsConfig`) centralizes results, cache, and dataset paths.
- **Per-dataset data paths** are explicit fields in `GloPoseConfig.paths`:
  - `bop_data_folder` → `/mnt/data/vrg/public_datasets/bop/` (handal, hope, tless, lmo, icbin, etc.)
  - `ho3d_data_folder` → `/mnt/personal/jelint19/data/HO3D/`
  - `navi_data_folder` → `/mnt/personal/jelint19/data/NAVI/navi_v1.5/`
  - `behave_data_folder` → `/mnt/personal/jelint19/data/BEHAVE/`
  - `tum_rgbd_data_folder` → `/mnt/personal/jelint19/data/SLAM/tum_rgbd/`
  - `google_scanned_objects_data_folder` → `/mnt/personal/jelint19/data/GoogleScannedObjects/`
  - `default_data_folder` → legacy fallback, still used by `run_HANDAL.py` (native format, not on RCI)
- **Test sequence selection:** When proposing RCI test commands, pick ~5 sequences uniformly spaced across the full
  dataset — not just the first few. For HANDAL (40 objects): `obj_000001`, `obj_000009`, `obj_000017`, `obj_000025`,
  `obj_000033`. This catches issues that only appear on specific object geometries or sequence lengths.

## Development Environment

### RCI (Remote Computing Infrastructure)

The project runs on an RCI GPU cluster. Claude has **no direct access** to run code on RCI — only syntax-check locally.

**Running on RCI:**

```bash
~/run_gpu.sh -p glotracker -d gpu      # default gpufast partition, 1 GPU, 256G RAM
~/run_gpu.sh -p glotracker -d h200     # h200fast partition, 1024G RAM
~/run_gpu.sh -p glotracker -d amd      # amdgpufast partition, 512G RAM
# Add -n 2 for multi-GPU
```

This calls `srun` with the selected partition and drops into an interactive bash shell.

**Modules loaded** (from `~/scripts/modules_glotracker_login4.txt`):
PyTorch-Lightning/2.4.0 (CUDA 12.4.0), torchvision/0.19.0, maturin/1.5.1, scikit-image/0.23.2, git/2.42.0, CMake/3.29.3,
Eigen/3.4.0, Boost/1.83.0, FLANN/1.9.2, METIS/5.1.0, CeresSolver/2.2.0, glog/0.6.0, SQLite/3.43.1, CGAL/5.6.1,
Qt5/5.15.13, FreeImage/3.18.0, glew/2.2.0, GMP/6.3.0, MPFR/4.2.1

**RCI project home:** `~/projects/GloPose/` (on RCI filesystem)

### sshfs Mount (read-only for Claude)

The RCI personal folder is sshfs-mounted locally:

- **Mount point:** `/mnt/personal/jelint19/`
- **Claude can read** this directory to inspect results, caches, logs, weights, data
- **Claude must NEVER write** to this directory

**Directory layout on the mount:**

```
/mnt/personal/jelint19/
├── cache/          # Flow caches, descriptor caches, SAM caches, view graphs
│   ├── UFM_cache/, SAM_cache/, sift_cache/, view_graph_cache/
│   ├── detections_cache/, detections_templates_cache/
│   └── dinov2_cache/, dinov3_cache/
├── results/        # Experiment outputs
│   ├── FlowTracker/    # Main onboarding results
│   ├── PoseEstimation/ # Detection/pose results
│   └── condensation/   # Template condensation results
├── data/           # Datasets (HANDAL, HO3D, NAVI, BEHAVE, bop, GoogleScannedObjects, etc.)
├── weights/        # Pre-trained model weights
│   ├── RoMa/, SegmentAnything2/, DINO/, DepthAnythingV2/
│   └── Dust3r/, FastSAM/, MFT/, RAFT/, XMem/
├── logs/           # Job logs
└── tmp/            # Temporary files (Gradio temp dir, etc.)
```

**RCI run script also available locally (read-only):** `/home/tom/rci_home/run_gpu.sh`

### Local Machine Limitations

- RoMa and UFM models are too large for the local GPU
- Only passthrough onboarding (no dense matching) can run locally
- Code changes are made locally, then synced to RCI for execution

---

## Known Issues & Architectural Notes

### External Repo Integration

cnos integration is centralized in `adapters/cnos_adapter.py` — the sole location for `sys.path` manipulation
and cnos imports. Scoring functions are vendored in `detection/scoring.py`, small utilities in `utils/mask_utils.py`
and `utils/bbox_utils.py`. No other files import cnos internals directly.

SAM2 video predictor is wrapped in `adapters/sam2_adapter.py` — the sole location for SAM2 imports and
Hydra config initialization. `SAM2SegmentationProvider` in `data_providers/frame_provider.py` imports
`build_video_predictor` and `mask_to_sam_prompt` from the adapter.

---

## TODO

### Phase 2: Module B — Detection

Goal: clear separation between offline representation building (B1) and online inference (B2).

#### 2.2 Separate representation building from inference

- [ ] Define clean `build_detection_model(onboarding_result: OnboardingResult, ...) -> DetectionModel` interface
- [ ] Condensation algorithms (Hart's CNN, imblearn) stay here
- [ ] Statistical metadata computation (whitening, CSLS, Mahalanobis) stays here

#### 2.4 Clean up BOP coupling

BOP path utilities extracted to `utils/bop_data.py`. Remaining:

- [ ] Detection module should accept images and models, not know about BOP folder layout

---

### Phase 3: Module C — Pose estimation

Goal: given detections and an onboarding result, produce 6DoF poses.

#### 3.2 Evaluation

- [ ] Extend `eval/` with remaining evaluation types:
    - [ ] `evaluate_detections(detections, ground_truth) -> DetectionMetrics` (BOP COCO)
    - [ ] `evaluate_poses(pose_estimates, ground_truth) -> PoseMetrics` (BOP 6DoF)
    - [x] `evaluate_reconstruction(reconstruction, ground_truth) -> ReconstructionMetrics`
        — implemented in `eval/eval_point_cloud.py`: accuracy, completeness, overall, F-score@1/2/5mm,
        pose AUC@5/10/30°. Wired into CSV pipeline via `eval/eval_onboarding.py`.
- [x] `eval_bop_detection.py` moved from `utils/` to `eval/`
- [ ] Gradually migrate functions from `utils/bop_challenge.py` into `utils/bop_data.py` (clean extraction already started)

---

### Phase 4: Infrastructure improvements

These can be done in parallel with the module work.

#### 4.2 Visualization

- [ ] **Update Rerun SDK from ~0.21 to 0.30** — review breaking API changes (blueprint API, logging API,
  annotation classes, `rr.init`/`rr.spawn` signatures) and update all call sites in `results_logging.py`,
  `visualizations/pose_estimation_visualizations.py`, and `visualizations/rerun_utils.py`

---

## Paper: Onboarding Module Evaluation

Checklist for producing the experimental results for a paper on the onboarding part of the pipeline.
Target deliverables: scripts that produce LaTeX-ready tables and qualitative figures.

### Datasets

Primary evaluation datasets:
- **HANDAL** (40 objects, static + dynamic onboarding sequences, BOP format)
- **HO3D** (hand-object, GT from multi-cam rig)
- **NAVI** (novel view synthesis benchmark)
- **BOP classic** (T-LESS, LM-O, IC-BIN — static onboarding only)
- **HOPE** (household objects, static + dynamic)
- **HOT3D** (33 objects, hand-object tracking, fisheye cameras from Meta Aria/Quest3, BOP format)

---

### P1. Reliable static onboarding baseline

Goal: our pipeline reliably reconstructs objects with correct camera poses and complete reconstructions
on all static onboarding sequences across all datasets.

#### P1.1 Audit and fix current failures

- [ ] Run static onboarding on HANDAL (all 40 objects, `_up` and `_down` sequences) with RoMa matching,
  collect `reconstruction_keyframe_stats.csv` and `{dataset}_reconstruction_statistics.csv`
- [ ] Identify failure cases: sequences where reconstruction fails (`reconstruction is None`), alignment
  fails, or pose error is above 10°/10cm. Categorize root causes (too few keyframes, degenerate geometry,
  matching failures, COLMAP mapper failure)
- [ ] Fix or work around the top failure modes (e.g., adjust matcher confidence thresholds, keyframe
  selection aggressiveness, COLMAP mapper params per failure category)
- [ ] Run static onboarding on HO3D, NAVI, BOP classic (T-LESS, LM-O, IC-BIN), HOPE — same audit
- [ ] Achieve ≥95% reconstruction success rate on static sequences across all datasets

#### P1.2 Reconstruction completeness metric

3D point cloud quality metrics (accuracy, completeness, overall, F-score) are now implemented in
`eval/eval_point_cloud.py` and wired into the CSV pipeline.

---

### P2. Dynamic onboarding (no GT cam poses)

Goal: handle BOP dynamic onboarding sequences where only the first frame has a known pose.
Recover scale, align coordinate systems, and evaluate against GT.

#### P2.1 Fix dynamic onboarding evaluation path

Currently `run_on_bop_sequences()` sets `gt_Se3_world2cam=None` for dynamic sequences, which causes
alignment to skip and no evaluation to run.

- [ ] Fix `utils/experiment_runners.py`: for dynamic sequences, load at minimum the first-frame GT pose
  (already available in BOP `scene_gt.json`) so depth-based alignment can proceed
- [ ] Verify `align_reconstruction_with_pose()` in `onboarding/reconstruction.py` works end-to-end: it needs GT pose for
  first frame + depth maps. Test with predicted depth and with GT depth (where available)
- [ ] After alignment, evaluate all registered cameras against their GT poses (for sequences where
  per-frame GT exists in `scene_gt.json` but we don't use it during reconstruction)
- [ ] Add `similarity_transformation='depths'` results to the same CSV format as static onboarding

#### P2.2 Scale recovery robustness

- [ ] Compare scale recovery methods: (a) current depth-based median ratio, (b) using GT depth for
  first frame only vs predicted depth, (c) using multiple frames' depth for more robust scale estimation
- [ ] If predicted depth is too noisy for reliable scale, consider fallback: use the known object diameter
  (available in BOP `models_info.json`) as a scale reference

#### P2.3 Run dynamic onboarding across datasets

- [ ] Run on HANDAL dynamic sequences (40 objects × `_dynamic` suffix)
- [ ] Run on HOPE dynamic sequences
- [ ] Collect same metrics as static (rotation/translation error, accuracy at thresholds)

---

### P3. Ablation studies

#### P3.1 Frame selection ablation

Compare our adaptive frame filtering against fixed-interval subsampling.

**Prerequisite — passthrough configs already exist** (`configs/onboarding/passthroughs/`): every 1st, 2nd, 4th, 8th,
16th, 32nd, 64th frame.

- [ ] Run all passthrough configs on HANDAL static (all 40 objects, `_up` sequences)
  using the same matcher (RoMa) so the only variable is keyframe selection
- [ ] Run our adaptive frame filter (RoMa-based, `dense_matching` filter) on the same sequences
- [ ] Write `scripts/ablation_frame_selection.py`:
  - Reads all `{dataset}_reconstruction_statistics.csv` across experiments
  - Produces **Table: Frame selection ablation** — rows: {every-1, every-2, ..., every-64, ours-adaptive},
    columns: {mean rot err (°), mean trans err (cm), rot acc@5°, trans acc@5cm, #keyframes (mean),
    reconstruction rate (%), runtime (s)}
  - Include 3D quality columns when available: accuracy_mm, completeness_mm, overall_mm, fscore_5mm,
    pose_auc_at_5, pose_auc_at_10, pose_auc_at_30
  - Also produces a **plot**: x-axis = mean #keyframes, y-axis = pose accuracy, with a point per method

#### P3.2 ViewGraph filtering ablation

Compare our filtered ViewGraph (selective edges based on matching reliability) against a complete
(all-to-all) ViewGraph.

- [ ] Verify that `frame_filter_view_graph='dense'` config option creates all-to-all edges in the
  ViewGraph (check `onboarding/frame_filter.py` and `onboarding/pipeline.py`)
- [ ] Run with `frame_filter_view_graph='dense'` on HANDAL static sequences — same keyframes as
  our adaptive filter, but with all-to-all matching
- [ ] Run with our default filtered ViewGraph on the same sequences
- [ ] Add rows to the frame selection table or produce a separate **Table: ViewGraph density ablation**

#### P3.3 Matching method ablation

Compare dense matching (RoMa, UFM) vs sparse matching (SIFT+LightGlue).

- [ ] Run onboarding with `reconstruction_matcher='SIFT'` on HANDAL static (all 40 objects)
- [ ] Run onboarding with `reconstruction_matcher='UFM'` on HANDAL static (all 40 objects)
- [ ] Run onboarding with `reconstruction_matcher='RoMa'` on HANDAL static (all 40 objects) — may
  already exist from P1.1
- [ ] Write `scripts/ablation_matching.py`:
  - Produces **Table: Matching method ablation** — rows: {RoMa, UFM, SIFT+LightGlue},
    columns: {mean rot err, mean trans err, rot acc@5°, trans acc@5cm, #matches/pair (mean),
    reconstruction rate (%), matching time (s/pair), total time (s)}

#### P3.4 Comparison against external SfM/pose methods

Compare our COLMAP-based reconstruction pipeline against learned reconstruction methods.
For each method: run on our selected keyframes AND on every-nth subsampled frames.

**Prerequisites — write adapters/wrappers for each method:**

- [ ] Write `adapters/dust3r_adapter.py`: wrapper that takes a set of images + masks and runs
  Dust3r/Mast3r reconstruction. Output: camera poses (world2cam) + 3D point cloud.
  Mast3r is in `repositories/mast3r/` (already cloned). Needs: load model, run on image pairs,
  global alignment, extract poses.
- [ ] Write `adapters/vggt_adapter.py`: wrapper for VGGT (in `repositories/vggt/`). Takes images,
  outputs camera poses + point cloud. VGGT is a feed-forward method (single forward pass for
  all images simultaneously).
- [ ] Write `adapters/mapanything_adapter.py`: wrapper for MapAnything. If not already cloned,
  add as submodule or install. Takes images, outputs posed reconstruction.
- [ ] Each adapter must output a common format: list of `(image_id, Se3_world2cam)` + optional
  point cloud as numpy array, so we can evaluate with the same metrics.

**Runs — static scenes, with and without background:**

- [ ] For each method (Dust3r/Mast3r, VGGT, MapAnything) and each input variant:
    - [ ] Our adaptive keyframes, with background (original images)
    - [ ] Our adaptive keyframes, without background (black background outside segmentation mask)
    - [ ] Every-8th frame, with background
    - [ ] Every-8th frame, without background
  Run on HANDAL static (all 40 objects, `_up` sequences)

**Runs — dynamic scenes, without background only:**

- [ ] For each method, run on HANDAL dynamic sequences using images with background removed
  (since dynamic scenes have moving camera + changing background, background removal is mandatory)

**Evaluation:**

- [ ] Write `scripts/ablation_external_methods.py`:
  - For each method + variant, align predicted poses to GT using Kabsch (static) or depths (dynamic)
  - Produces **Table: Comparison with learned reconstruction methods (static)** —
    rows: {Ours (COLMAP+RoMa), Dust3r/Mast3r, VGGT, MapAnything} × {our keyframes, every-8th} × {bg, no-bg},
    columns: {mean rot err, mean trans err, rot acc@5°, trans acc@5cm, reconstruction rate, runtime}
  - Produces **Table: Comparison with learned reconstruction methods (dynamic)** — same but dynamic sequences,
    no-bg only

#### P3.5 Background removal ablation

- [ ] Run our pipeline (RoMa) with and without `black_background=True` on HANDAL static
- [ ] Add this as a row in the matching ablation table or a separate mini-table

#### P3.6 Track merging matches ablation

- [ ] Run onboarding with track merging matches disabled on HANDAL static (all 40 objects)
- [ ] Run onboarding with track merging matches enabled (default) on the same sequences
- [ ] Compare reconstruction quality: rotation/translation error, accuracy at thresholds,
  reconstruction rate, number of 3D points
- [ ] Add results as a row in the ablation tables or a separate mini-table

---

### P4. 3D reconstruction quality

Goal: compare our reconstructed 3D model against GT mesh models.

#### P4.1 GT model loading — DONE

GT mesh loading (`trimesh.load` + surface sampling) is in `eval/eval_point_cloud.py:sample_points_from_mesh()`.
GT model path resolution is in `eval/eval_onboarding.py:resolve_gt_model_path()` (BOP, NAVI, HO3D).
Unit mapping (BOP→mm, NAVI/HO3D/GSO→m) is in `eval/eval_onboarding.py:gt_mesh_unit_for_dataset()`.

#### P4.2 Reconstruction-to-GT alignment — DONE (via existing Kabsch)

Reconstruction points are already in GT frame after Kabsch alignment (`onboarding/reconstruction.py`
transforms `reconstruction.points3D` with the Sim3d alignment). No separate ICP step needed for
static sequences with GT poses.

- [ ] For dynamic sequences or sequences without GT poses, implement ICP fallback alignment

#### P4.3 3D distance metrics — DONE

Implemented in `eval/eval_point_cloud.py:compute_reconstruction_metrics()`:
- **Accuracy** (mm): mean NN-distance pred→GT, clamped at 20mm
- **Completeness** (mm): mean NN-distance GT→pred, clamped at 20mm
- **Overall** (mm): (accuracy + completeness) / 2
- **F-score@τ** at τ = 1mm, 2mm, 5mm
- Uses `scipy.spatial.KDTree` for efficient NN queries

Also implemented `compute_pose_auc()` (AUC@5/10/30° over rotation errors, adapted from VGGT).

All metrics are wired into the CSV pipeline at sequence, dataset, and experiment levels.

#### P4.4 Run 3D evaluation

- [ ] Run 3D evaluation on all HANDAL static objects (GT models available in BOP `models/` dir)
  — metrics are now computed automatically when `gt_model_path` exists
- [ ] Run on HO3D, HOPE, BOP classic where GT models are available
- [ ] Verify new columns in `reconstruction_sequence_stats.csv`: `accuracy_mm`, `completeness_mm`,
  `overall_mm`, `fscore_1mm`, `fscore_2mm`, `fscore_5mm`, `pose_auc_at_5`, `pose_auc_at_10`, `pose_auc_at_30`
- [ ] Write `scripts/eval_3d_reconstruction.py`:
  - Reads existing CSVs and produces **Table: 3D reconstruction quality** — rows: per-dataset aggregate,
    columns: {accuracy (mm), completeness (mm), overall (mm), F-score@1mm, F-score@2mm, F-score@5mm,
    AUC@5°, AUC@10°, AUC@30°}
  - Also run for external methods (P3.4) to include in comparison

---

### P5. Pose estimation from reconstruction (optional)

Goal: demonstrate that our onboarding result can be used for 6DoF pose estimation on novel images.

#### P5.3 Evaluate pose estimation

- [ ] Run detection + pose estimation on HANDAL BOP val sequences using default CNOS detections
- [ ] Evaluate using BOP 6DoF metrics (VSD, MSSD, MSPD via `bop_toolkit`)
- [ ] Write `scripts/eval_pose_estimation.py`:
  - Produces **Table: Pose estimation from onboarding** — rows: per-dataset,
    columns: {AR_VSD, AR_MSSD, AR_MSPD, AR (mean), detection AP50}
  - This table demonstrates end-to-end viability, not necessarily SOTA results

---

### P6. Paper deliverables

Summary of all scripts and outputs for the paper.

#### Scripts

| Script | Produces | Table/Figure |
|--------|----------|--------------|
| `scripts/ablation_frame_selection.py` | Frame selection comparison | Table 1 + Figure 2 |
| `scripts/ablation_matching.py` | Matching method comparison | Table 2 |
| `scripts/ablation_external_methods.py` | External method comparison | Table 3 (static) + Table 4 (dynamic) |
| `scripts/eval_3d_reconstruction.py` | 3D quality metrics | Table 5 |
| `scripts/eval_pose_estimation.py` | Pose estimation results | Table 6 (optional) |

#### Figures (qualitative)

- [ ] Write `scripts/paper_qualitative.py`:
  - For ~5 representative objects, render: input frames → selected keyframes → 3D reconstruction
    (COLMAP point cloud rendered from a canonical viewpoint) → GT model overlay
  - Show failure cases and challenging objects
  - Export as PDF-ready figures (matplotlib, no rerun dependency)

---

## Paper: Detection Module Evaluation

Checklist for the detection paper. Key experiment: 3-way comparison of CNOS + BOP onboarding frames
vs CNOS + our keyframes vs GloPose detection.

### D1. Verify vendored detection produces identical results

Runtime dependencies (DINOv2/v3 descriptors, NMS, detection config) are vendored. CNOS baseline
runner: `scripts/run_cnos_baseline.py`. ViewGraph export: `export_viewgraphs_to_cnos_format()`
in `data_structures/view_graph.py`.

- [ ] Run GloPose detection and original CNOS on the same test sequence, verify identical outputs
  (descriptor values, similarity scores, NMS survivors, final detections)

### D2. Divergence investigation

Trace where GloPose detection diverges from original CNOS. Compare each stage:
descriptor extraction, proposal generation, similarity scoring, NMS, and post-processing.

- [ ] Run side-by-side comparison on HANDAL val and HOPE val with matched configs
- [ ] Identify and document divergence points (if any) — are differences due to vendoring
  or to intentional changes (e.g., DetectionConfig defaults vs Hydra defaults)?
- [ ] Quantify detection AP impact of each divergence
