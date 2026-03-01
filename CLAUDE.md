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

## Testing

No formal test suite. Validation is done by running dataset-specific scripts and evaluation utilities (
`utils/eval_*.py`, `utils/bop_challenge.py`).

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
| A. Onboarding      | `onboarding/pipeline.py` (orchestrator), `onboarding/reconstruction.py` (SfM), `onboarding/frame_filter.py`, `data_providers/` | Working but monolithic. `OnboardingPipeline` is tightly coupled to all providers.                                              |
| B1. Representation | `detection/representation.py` (condensation), `view_graph.py` (ViewGraph)                                  | Working. `TemplateBank` dataclass exists in `data_structures/template_bank.py`.                                                |
| B2. Detection      | `detection/detector.py` (`BOPChallengePosePredictor`)                                                      | Working. Despite the class name, it only does detection (the pose call is commented out). Tangled with BOP I/O and evaluation. |
| C. Pose estimation | `detection/detector.py` (commented out at lines 211-216)                                                   | **Not connected.** The flow provider is initialized but never called.                                                          |

### Key data types (interface boundaries)

- **`ViewGraph`** (`data_structures/view_graph.py`): Per-object posed keyframes + COLMAP reconstruction. Primary
  onboarding output.
- **`TemplateBank`** (`data_structures/template_bank.py`): Condensed detection-ready templates with descriptors and statistical
  params. Bridge between onboarding and detection.
- **`Detections`** (from `repositories/cnos/src/model/utils.py`): Scored bboxes + masks. Detection output. Currently an
  external type — should become our own.
- **`DataGraph`** (`data_structures/data_graph.py`): Internal to onboarding — tracks per-frame data and cross-frame
  relationships during processing.

### Dataset formats & comparison methods

Detailed documentation of all dataset folder layouts, annotation schemas, image naming conventions,
camera intrinsics formats, GT structures, and external method APIs lives in
[`docs/dataset_formats.md`](docs/dataset_formats.md).

### Key directories

- `adapters/` — External repository wrappers (cnos); sole location for `sys.path` manipulation
- `configs/` — Python-based config files (not YAML), loaded via `utils.general.load_config()`
- `onboarding/` — OnboardingPipeline, SfM reconstruction, frame filtering, COLMAP utils
- `detection/` — template condensation (representation building), detector (inference), scoring
- `pose_estimation/` — pose estimator (placeholder, not yet implemented)
- `data_providers/` — Frame, flow, depth, and matching providers (abstract + implementations)
- `data_structures/` — ViewGraph, DataGraph, KeyframeBuffer, observations
- `models/` — Mesh rendering (Kaolin), feature encoding
- `utils/` — Dataset sequences, math (SE(3)), image I/O, evaluation, results logging, mask/bbox utils
- `visualizations/` — Flow and pose visualization helpers
- `scripts/` — Dataset downloaders, evaluation, job runners
- `repositories/` — External dependency submodules (do not edit)

### Key patterns

- **Provider pattern:** Abstract base classes (FrameProvider, FlowProvider, MatchingProvider) with multiple backends
- **Graph-based data:** DataGraph (frames + temporal edges), ViewGraph (3D template library)
- **Dataclass configs:** `GloPoseConfig` in `configs/glopose_config.py` is the top-level config, composed of sub-configs: `PathsConfig`, `RunConfig`, `InputConfig`, `OnboardingConfig`, `CondensationConfig`, `DetectionConfig`, `VisualizationConfig`, `RendererConfig`
- **Narrowed config passing:** Lower-level components receive only the sub-config(s) they need, not the full
  `GloPoseConfig`. For example: `RenderingKaolin` and `Encoder` take `RendererConfig`; frame filters and matching
  providers take `OnboardingConfig` + `device`; evaluation functions take `RunConfig` + `BaseBOPConfig`.
  Orchestrators (`OnboardingPipeline`, `FrameProviderAll`, `BOPChallengePosePredictor`, `run_*.py` scripts) still
  hold the full `GloPoseConfig` and pass the relevant sub-configs down.
- **Observation types:** FrameObservation, FlowObservation encapsulate per-frame and cross-frame data

### Cross-cutting concerns

- **DINOv2 descriptors** are centralized behind `adapters/cnos_adapter.py` (`DescriptorExtractor` protocol,
  `CnosDescriptorExtractor` implementation). All consumers use `create_descriptor_extractor()` or pass
  a `DescriptorExtractor` instance — no direct cnos imports outside `adapters/`.
- **BOP dataset conventions** (folder layout, splits, annotations) are known by `detection/detector.py`,
  `detection/representation.py`, and many `run_*.py` scripts. Should be encapsulated in a BOP data adapter.
- **`Detections` type** from cnos is accessed via `adapters.cnos_adapter.make_cnos_detections()` for NMS
  methods. Should eventually become our own type at the module boundary.

## Code Conventions

- **Type hints:** Used extensively, modern union syntax (`X | Y`)
- **Naming:** snake_case for functions/variables, PascalCase for classes
- **Imports:** Relative within package (`from configs.glopose_config import GloPoseConfig`; or specific sub-configs
  like `from configs.glopose_config import RendererConfig, OnboardingConfig`), absolute for externals
- **Strings:** f-strings preferred
- **No linter/formatter configured** — match existing code style when making changes
- **Dataclasses** over plain dicts for structured data

## Key Dependencies

PyTorch, Kornia (geometry/camera), Kaolin (mesh rendering), pycolmap, SAM2, RoMa/UFM (optical flow), DINOv2 (via CNOS),
NetworkX, Gradio, Rerun SDK, wandb

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

### Structural

- **`results_logging.py` (~983 lines)**: `WriteResults` has 10+ responsibilities — rerun blueprint layout (282 lines in
  `rerun_init` alone), keyframe viz, 3D camera viz, flow matching viz, image I/O, matplotlib helpers, silhouette
  rendering, math utilities. Should be 4-5 classes.
- **Duplicated visualization systems**: `results_logging.py` and `visualizations/pose_estimation_visualizations.py` have
  near-identical rerun init, blueprint setup, matching visualization logic, and overlapping annotation constants (
  `RerunAnnotations` vs `RerunAnnotationsPose`).

### External Repo Integration

cnos integration is centralized in `adapters/cnos_adapter.py` — the sole location for `sys.path` manipulation
and cnos imports. Scoring functions are vendored in `detection/scoring.py`, small utilities in `utils/mask_utils.py`
and `utils/bbox_utils.py`. No other files import cnos internals directly.

### Error Handling

- `app.py:113` has `except Exception: pass`.

---

## TODO

### Phase 1: Define module boundaries and shared types

These are prerequisites for working on modules A/B/C independently.

#### 1.1 External repo adapters

- [x] Create `adapters/cnos_adapter.py` — single location for `sys.path` manipulation and cnos imports (DINOv2
  descriptors, `Detections` type, Hydra configs). All other files import from the adapter.
    - [x] Wrap DINOv2 descriptor computation behind a `DescriptorExtractor` interface (currently scattered across
      `view_graph.py`, `detection/representation.py`, `detection/detector.py`)
    - [x] Vendor scoring functions into `detection/scoring.py`, mask/bbox utils into `utils/mask_utils.py`
      and `utils/bbox_utils.py`
    - [ ] Define our own `Detection` type; adapter converts to/from cnos `Detections` at the boundary
- [ ] Create `adapters/sam2_adapter.py` for SAM2 (currently inline in `frame_provider.py:341`)
- [ ] Evaluate whether `mast3r`, `vggt`, `ho3d` need adapters

### Phase 2: Module B — Detection

Goal: clear separation between offline representation building (B1) and online inference (B2).

#### 2.1 Split detection and pose estimation logic

`BOPChallengePosePredictor` (now in `detection/detector.py`) currently only does detection (the pose call
is commented out). Split into two modules:

- [x] Move detection code to `detection/detector.py` (done via file reorganization)
- [ ] Create `pose_estimation/estimator.py` — pose estimation (currently commented out):
    - [ ] `estimate_poses(detections, onboarding_result, image, intrinsics) -> List[PoseEstimate]`
    - [ ] Template matching → PnP or flow-based alignment
    - [ ] The flow provider is already initialized in `BOPChallengePosePredictor.__init__` — move it here
- [ ] Define shared types at the boundary: `Detection`, `PoseEstimate`

#### 2.2 Separate representation building from inference

- [x] Condensation logic moved to `detection/representation.py` (done via file reorganization)
- [ ] Define clean `build_detection_model(onboarding_result: OnboardingResult, ...) -> DetectionModel` interface
- [ ] Condensation algorithms (Hart's CNN, imblearn) stay here
- [ ] Statistical metadata computation (whitening, CSLS, Mahalanobis) stays here

#### 2.3 CNOS integration investigation

- [ ] Inspect original CNOS detector pipeline end-to-end to understand where our detection results diverge (some
  datasets show significantly worse performance). Compare: descriptor extraction, proposal generation, similarity
  scoring, NMS, and post-processing steps against the original CNOS code.
- [ ] Decide integration strategy: either (a) reliably integrate CNOS as a dependency with a clean adapter, or
  (b) vendor the specific pieces we need (descriptor matching, similarity scoring) into our own codebase and
  drop the CNOS dependency. Document the decision and rationale.

#### 2.4 Clean up BOP coupling

- [ ] BOP folder conventions (scene/image path construction, annotation loading) are scattered across
  `detection/detector.py`, `detection/representation.py`, and `run_*.py` scripts
- [ ] Extract into a `bop_data.py` adapter: `load_annotations`, `get_image_path`, `get_scene_folder`, etc.
- [ ] Detection module should accept images and models, not know about BOP folder layout

---

### Phase 3: Module C — Pose estimation

Goal: given detections and an onboarding result, produce 6DoF poses.

#### 3.1 Implement/reconnect pose estimation

- [ ] Uncomment and refactor the pose estimation call in `detection/detector.py` (lines 211-216)
- [ ] Create `pose_estimation/estimator.py`:
    - [ ] 
      `estimate_poses(detections: List[Detection], onboarding_result: OnboardingResult, image, intrinsics) -> List[PoseEstimate]`
    - [ ] Template matching → PnP or flow-based alignment
- [ ] The flow provider (RoMa/UFM) is already initialized in `BOPChallengePosePredictor.__init__` — wire it up

#### 3.2 Evaluation

- [ ] Separate evaluation from pipeline execution
- [ ] Create `evaluation/` module:
    - [ ] `evaluate_detections(detections, ground_truth) -> DetectionMetrics` (BOP COCO)
    - [ ] `evaluate_poses(pose_estimates, ground_truth) -> PoseMetrics` (BOP 6DoF)
    - [ ] `evaluate_reconstruction(reconstruction, ground_truth) -> ReconstructionMetrics`
        - [ ] Chamfer distance
        - [ ] Hausdorff distance
        - [ ] Additional metrics as needed (F-score, completeness, accuracy)
- [ ] Move `utils/eval_*.py` and `utils/bop_challenge.py` into this module

---

### Phase 4: Infrastructure improvements

These can be done in parallel with the module work.

#### 4.1 Reorganize file structure

- [x] Reorganize project layout to reflect the three-module architecture:
    - [x] `onboarding/` — `pipeline.py` (OnboardingPipeline), `frame_filter.py`, `reconstruction.py` (SfM), `colmap_utils.py`
    - [x] `detection/` — `representation.py` (condensation/B1), `detector.py` (detection inference/B2)
    - [x] `pose_estimation/` — placeholder package (not yet implemented)
    - [x] `data_structures/` — shared types (ViewGraph, TemplateBank, Detection, PoseEstimate) — unchanged
    - [ ] `adapters/` — external repo wrappers (cnos, sam2) — not yet created
- [x] Move top-level scripts (`onboarding_pipeline.py`, `condensate_templates.py`) into packages
- [x] Update all imports across the codebase
- [ ] Verify all entry points (`run_*.py`, `app.py`) still work on RCI

#### 4.2 Visualization

- [ ] **Update Rerun SDK from ~0.22 to 0.30** — review breaking API changes (blueprint API, logging API,
  annotation classes, `rr.init`/`rr.spawn` signatures) and update all call sites in `results_logging.py`,
  `visualizations/pose_estimation_visualizations.py`, and any other files using `rerun`
- [ ] Unify `RerunAnnotations` and `RerunAnnotationsPose` into a single annotation constants module
- [ ] Extract shared rerun blueprint setup into a common function
- [ ] Extract shared matching visualization logic into a shared helper
- [ ] Decompose `results_logging.py` (~983 lines) into focused classes: `KeyframeVisualizer`, `Scene3DVisualizer`,
  `MatchingVisualizer`, `ImageLogger`

#### 4.3 Web UI

- [ ] Update `app.py` to reflect the three-module architecture:
    - [ ] Tab 1: Onboarding (existing "Run on Dataset" + "Custom Input")
    - [ ] Tab 2: Detection (run detector on novel images using an onboarded model)
    - [ ] Tab 3: Pose estimation (given detections, estimate poses)
    - [ ] Tab 4: Browse results (existing)

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

Currently we only measure camera pose accuracy (rotation/translation error). We also need to measure
whether the reconstruction is *complete* (covers the whole object, not just one side).

- [ ] Implement viewpoint coverage metric: for each reconstruction, compute the angular span of camera
  poses (e.g., solid angle coverage on the viewing sphere). Report fraction of hemisphere/sphere covered.
- [ ] Add to `eval/eval_reconstruction.py`: `compute_viewpoint_coverage(reconstruction) -> float`
- [ ] Add completeness columns to `reconstruction_keyframe_stats.csv` output

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

**Prerequisite — passthrough configs already exist** (`configs/passthroughs/`): every 1st, 2nd, 4th, 8th,
16th, 32nd, 64th frame.

- [ ] Run all passthrough configs on HANDAL static (all 40 objects, `_up` sequences)
  using the same matcher (RoMa) so the only variable is keyframe selection
- [ ] Run our adaptive frame filter (RoMa-based, `dense_matching` filter) on the same sequences
- [ ] Write `scripts/ablation_frame_selection.py`:
  - Reads all `{dataset}_reconstruction_statistics.csv` across experiments
  - Produces **Table: Frame selection ablation** — rows: {every-1, every-2, ..., every-64, ours-adaptive},
    columns: {mean rot err (°), mean trans err (cm), rot acc@5°, trans acc@5cm, #keyframes (mean),
    reconstruction rate (%), runtime (s)}
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

### P4. 3D reconstruction quality (Chamfer / Hausdorff)

Goal: compare our reconstructed 3D model against GT mesh models.

#### P4.1 GT model loading

- [ ] Write `utils/gt_model_loader.py`: loads GT models from BOP dataset directories
  (`models/obj_NNNNNN.ply`). BOP models are triangle meshes in PLY format. Parse with trimesh or
  Open3D. Return as point cloud (sample surface points) or keep as mesh.
- [ ] Handle coordinate system conventions: BOP models are in mm, model origin is at object center.
  Our COLMAP reconstructions are in arbitrary scale — need Sim3d alignment first.

#### P4.2 Reconstruction-to-GT alignment

- [ ] Implement `align_reconstruction_to_gt_model()`:
  - Extract 3D points from COLMAP reconstruction (`reconstruction.points3D`)
  - Sample points from GT mesh surface (uniform sampling, e.g., 10k–100k points)
  - Run ICP (trimesh or Open3D) to align reconstruction point cloud to GT model
  - Alternatively: use known GT camera poses to transform reconstruction into GT model frame
    (if Kabsch alignment to GT poses already done, the reconstruction is already in GT frame)
- [ ] Decide alignment strategy: ICP (no GT poses needed) vs GT-pose-based (more reliable if poses
  are accurate). Likely use GT-pose-based for static sequences, ICP as fallback.

#### P4.3 3D distance metrics

- [ ] Implement `utils/eval_3d_metrics.py`:
  - `chamfer_distance(pred_points, gt_points) -> float` — mean of mean nearest-neighbor distances
    in both directions
  - `hausdorff_distance(pred_points, gt_points) -> float` — max of max nearest-neighbor distances
  - `f_score(pred_points, gt_points, threshold) -> float` — fraction of points with NN distance < threshold
  - `completeness(pred_points, gt_points, threshold) -> float` — fraction of GT points covered
  - `accuracy(pred_points, gt_points, threshold) -> float` — fraction of predicted points close to GT
  - Use KD-tree (scipy or Open3D) for efficient NN queries

#### P4.4 Run 3D evaluation

- [ ] Run 3D evaluation on all HANDAL static objects (GT models available in BOP `models/` dir)
- [ ] Run on HO3D, HOPE, BOP classic where GT models are available
- [ ] Write `scripts/eval_3d_reconstruction.py`:
  - For each sequence: load COLMAP reconstruction, load GT model, align, compute metrics
  - Produces **Table: 3D reconstruction quality** — rows: per-dataset aggregate,
    columns: {Chamfer (mm), Hausdorff (mm), F-score@1mm, F-score@5mm, completeness@5mm, accuracy@5mm}
  - Also run for external methods (P3.4) to include in comparison

---

### P5. Pose estimation from reconstruction (optional)

Goal: demonstrate that our onboarding result can be used for 6DoF pose estimation on novel images.

#### P5.1 Fix `predict_poses` in onboarding/reconstruction.py

- [ ] Remove `breakpoint()` at line 889 and dummy `Se3.identity()` return
- [ ] Complete the implementation: after registering the query image in the COLMAP reconstruction,
  extract the estimated `cam_from_world` pose from the mapper result
- [ ] Handle failure cases (registration fails, too few inliers)

#### P5.2 Wire up pose estimation in the detection pipeline

- [ ] Uncomment the `predict_poses` call in `detection/detector.py` (lines 205-210)
- [ ] For each detection: crop query image to detection bbox, run flow matching against best-matching
  ViewGraph templates, register into reconstruction, extract pose
- [ ] Output poses in BOP format (`scene_id, im_id, obj_id, score, R, t, time`)

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
