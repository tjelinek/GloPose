# CLAUDE.md - GloPose

## Project Overview

GloPose is a model-free 6DoF object pose estimation system following
the [BOP benchmark paradigm](https://bop.felk.cvut.cz/home/). It takes a video/image stream of a novel object and
produces posed 3D reconstructions and detection-ready representations, which are then used to detect and estimate 6DoF
poses of the object in novel images.

## Setup

- **Environment:** Conda (`environment.yml`), env name `glopose`, Python 3.13
- **Install:** `conda env create -f environment.yml && conda activate glopose`
- **Submodules:** `repositories/` contains git submodules (cnos, mast3r, Metric3D, SAM2, vggt, ho3d) — some installed as
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
| A. Onboarding      | `onboarding_pipeline.py` (orchestrator), `pose/glomap.py` (SfM), `pose/frame_filter.py`, `data_providers/` | Working but monolithic. `OnboardingPipeline` is tightly coupled to all providers.                                              |
| B1. Representation | `condensate_templates.py` (condensation), `view_graph.py` (ViewGraph)                                      | Working. `TemplateBank` dataclass exists but is defined in `condensate_templates.py`.                                          |
| B2. Detection      | `pose/pose_estimator.py` (`BOPChallengePosePredictor`)                                                     | Working. Despite the class name, it only does detection (the pose call is commented out). Tangled with BOP I/O and evaluation. |
| C. Pose estimation | `pose/pose_estimator.py` (commented out at lines 211-216)                                                  | **Not connected.** The flow provider is initialized but never called.                                                          |

### Key data types (interface boundaries)

- **`ViewGraph`** (`data_structures/view_graph.py`): Per-object posed keyframes + COLMAP reconstruction. Primary
  onboarding output.
- **`TemplateBank`** (`condensate_templates.py`): Condensed detection-ready templates with descriptors and statistical
  params. Bridge between onboarding and detection.
- **`Detections`** (from `repositories/cnos/src/model/utils.py`): Scored bboxes + masks. Detection output. Currently an
  external type — should become our own.
- **`DataGraph`** (`data_structures/data_graph.py`): Internal to onboarding — tracks per-frame data and cross-frame
  relationships during processing.

### Key directories

- `configs/` — Python-based config files (not YAML), loaded via `utils.general.load_config()`
- `pose/` — SfM (glomap.py), frame filtering, COLMAP utils, pose estimation
- `data_providers/` — Frame, flow, depth, and matching providers (abstract + implementations)
- `data_structures/` — ViewGraph, DataGraph, KeyframeBuffer, observations
- `models/` — Mesh rendering (Kaolin), feature encoding
- `utils/` — Dataset sequences, math (SE(3)), image I/O, evaluation, results logging
- `visualizations/` — Flow and pose visualization helpers
- `scripts/` — Dataset downloaders, evaluation, job runners
- `repositories/` — External dependency submodules (do not edit)

### Key patterns

- **Provider pattern:** Abstract base classes (FrameProvider, FlowProvider, MatchingProvider) with multiple backends
- **Graph-based data:** DataGraph (frames + temporal edges), ViewGraph (3D template library)
- **Dataclass configs:** `TrackerConfig` in `tracker_config.py` is the central config (~100+ settings)
- **Observation types:** FrameObservation, FlowObservation encapsulate per-frame and cross-frame data

### Cross-cutting concerns

- **DINOv2 descriptors** are computed in 3 places: `view_graph.py` (onboarding), `condensate_templates.py` (
  representation), `pose_estimator.py` (detection). Should be behind a single descriptor service / adapter.
- **BOP dataset conventions** (folder layout, splits, annotations) are known by `pose_estimator.py`,
  `condensate_templates.py`, and many `run_*.py` scripts. Should be encapsulated in a BOP data adapter.
- **`Detections` type** is imported from the external cnos repo. Should be our own type at the module boundary.

## Code Conventions

- **Type hints:** Used extensively, modern union syntax (`X | Y`)
- **Naming:** snake_case for functions/variables, PascalCase for classes
- **Imports:** Relative within package (`from tracker_config import TrackerConfig`), absolute for externals
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
- Many operations are GPU-intensive; device selection is configurable via `TrackerConfig`
- Hardcoded RCI paths are scattered across many files beyond `onboarding_pipeline.py` (scripts, dataset generators, pose
  estimator, etc.). The `TrackerConfig` defaults (`default_data_folder`, `default_results_folder`,
  `default_cache_folder`) centralize the main ones.

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
│   ├── RoMa/, SegmentAnything2/, Metric3D/, DINO/, DepthAnythingV2/
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

### Critical (all fixed)

- ~~**Hardcoded user paths**~~: Fixed — `onboarding_pipeline.py` now derives cache paths from
  `TrackerConfig.default_cache_folder`.
- ~~**Class name collision**~~: Fixed — renamed to `PrecomputedSIFTMatchingProvider` in `matching_provider_sift.py`.
- ~~**Config bug**~~: Fixed — `similarity_transformation: str = 'kabsch'` now uses proper type annotation.

### Structural

- **`OnboardingPipeline` tight coupling**: Constructor imports and instantiates 14+ concrete classes via `if/elif/else`
  chains. No dependency injection or factory pattern. Adding a new matcher/filter requires editing
  `onboarding_pipeline.py`.
- **`TrackerConfig` god-object**: ~47 flat fields spanning 8+ concerns (viz, input, rendering, mesh, filtering,
  matching, reconstruction, SIFT). Sub-configs exist (`BaseRomaConfig`, etc.) but most fields remain top-level.
- **`CommonFrameData` god-class** (`data_graph.py:32-72`): 20+ fields mixing input data, SIFT features, ground truth,
  filtering state, file paths, predictions, and timing.
- **`results_logging.py` (~983 lines)**: `WriteResults` has 10+ responsibilities — rerun blueprint layout (282 lines in
  `rerun_init` alone), keyframe viz, 3D camera viz, flow matching viz, image I/O, matplotlib helpers, silhouette
  rendering, math utilities. Should be 4-5 classes.
- **Duplicated visualization systems**: `results_logging.py` and `visualizations/pose_estimation_visualizations.py` have
  near-identical rerun init, blueprint setup, matching visualization logic, and overlapping annotation constants (
  `RerunAnnotations` vs `RerunAnnotationsPose`).

### Diamond Inheritance in Flow Providers

`PrecomputedRoMaFlowProviderDirect(RoMaFlowProviderDirect, PrecomputedFlowProviderDirect)` and
`PrecomputedUFMFlowProviderDirect(UFMFlowProviderDirect, PrecomputedFlowProviderDirect)` use diamond inheritance.
`PrecomputedFlowProviderDirect` duplicates `compute_flow` logic for the UFM variant. Suggested fix: replace with
composition (`FlowCache` standalone class) — reverted previous attempt due to breakage, needs careful re-implementation.

### External Repo Integration

All external repos are integrated via `sys.path.append('./repositories/...')` scattered across 10+ files. This is
CWD-dependent, pollutes namespaces, and provides no insulation from API changes. Files that touch cnos internals:
`view_graph.py`, `pose_estimation_visualizations.py`, `condensate_templates.py`, `bop_challenge.py`, `cnos_utils.py`.

### State Management

- `DataGraph` is a shared mutable hub — every pipeline stage freely mutates nodes and edges with no access control.
- `DataGraphStorage.__setattr__` silently moves tensors between devices on every assignment (action-at-a-distance).
- `app.py:62-65,151-152` has 5 module-level mutable globals shared between Gradio callbacks with no synchronization (
  except `_dataset_process_lock`).

### Error Handling

- `assert` used as runtime validation throughout (stripped with `python -O`).
- `runtime_utils.py:15-21` `exception_logger` catches `Exception` and silently continues.
- `app.py:113` has `except Exception: pass`.
- `onboarding_pipeline.py` has zero try/except — COLMAP failure is communicated only via `reconstruction is None` and
  print statements.
- `metric3d.py:15` and `render_ho3d_segmantations.py:107` have bare `except:` catching even `SystemExit`/
  `KeyboardInterrupt`.

---

## TODO

### Completed

- [x] Move hardcoded cache paths from `onboarding_pipeline.py` into `TrackerConfig` (`default_cache_folder`)
- [x] Fix `tracker_config.py` `similarity_transformation` type annotation
- [x] Rename `PrecomputedUFMFlowProviderDirect` in `matching_provider_sift.py` → `PrecomputedSIFTMatchingProvider`
- [x] Replace diamond inheritance with `FlowCache` composition
- [x] Rename `Tracker6D` → `OnboardingPipeline`, `tracker6d.py` → `onboarding_pipeline.py`
- [x] Fix operator-precedence bug in `flow_provider.py` (fixed during FlowCache refactoring)

---

### Phase 1: Define module boundaries and shared types

These are prerequisites for working on modules A/B/C independently.

#### 1.1 Define shared data types

- [ ] Create `glopose/types.py` (or `data_structures/types.py`) with the interface types:
    - [ ] `OnboardingResult`: wraps `ViewGraph` + 3D model path + `object_id`. Output of module A, input to B1 and C.
    - [ ] `DetectionModel`: wraps `TemplateBank` + `object_id`. Output of B1, input to B2.
    - [ ] `Detection`: our own type (bbox, mask, score, object_id, matched_template_id). Replace dependency on cnos
      `Detections`. Output of B2, input to C.
    - [ ] `PoseEstimate`: Se3 + confidence + object_id. Output of C.
- [ ] Move `TemplateBank` from `condensate_templates.py` into the shared types module

#### 1.2 External repo adapters

- [ ] Create `adapters/cnos_adapter.py` — single location for `sys.path` manipulation and cnos imports (DINOv2
  descriptors, `Detections` type, Hydra configs). All other files import from the adapter.
    - [ ] Wrap DINOv2 descriptor computation behind a `DescriptorExtractor` interface (currently scattered across
      `view_graph.py`, `condensate_templates.py`, `pose_estimator.py`)
    - [ ] Define our own `Detection` type; adapter converts to/from cnos `Detections` at the boundary
- [ ] Create `adapters/metric3d_adapter.py` for Metric3D
- [ ] Create `adapters/sam2_adapter.py` for SAM2 (currently inline in `frame_provider.py:341`)
- [ ] Evaluate whether `mast3r`, `vggt`, `ho3d` need adapters

#### 1.3 Config decomposition

- [ ] Split `TrackerConfig` (~47 flat fields) into per-module configs:
    - [ ] `OnboardingConfig` (frame filtering, dense matching, reconstruction, input data, SfM settings)
    - [ ] `DetectionConfig` (condensation params, descriptor model, similarity metric)
    - [ ] `PoseEstimationConfig` (already exists as `BasePoseEstimationConfig` — review if sufficient)
    - [ ] `VisualizationConfig` (rerun settings, write frequency, jpeg quality)
    - [ ] Sub-configs for providers: keep existing `BaseRomaConfig`, `BaseUFMConfig`, `BaseSiftConfig`, `BaseBOPConfig`
- [ ] Keep a top-level `GloPoseConfig` that composes all sub-configs
- [ ] Preserve backwards compatibility via `__getattr__` delegation if needed

---

### Phase 2: Module A — Onboarding

Goal: `onboarding_pipeline.py` becomes a clean onboarding pipeline that produces an `OnboardingResult`.

#### 2.1 Clean up OnboardingPipeline

- [x] ~~Rename `Tracker6D` → `OnboardingPipeline`, `tracker6d.py` → `onboarding_pipeline.py`~~
- [ ] Remove evaluation logic from `run_pipeline` (lines 248-278) into a separate evaluation step
- [x] ~~Remove `evaluate_sam` method — it's a separate workflow, not part of onboarding~~
- [x] ~~Extract the destructive `shutil.rmtree` from `__init__` into `prepare_output_folder()`, called at start
  of `run_pipeline`~~
- [ ] Have `run_pipeline` return an `OnboardingResult` rather than writing to disk as a side effect

#### 2.2 Reduce coupling

- [ ] Introduce a provider factory/registry: map config strings (`'RoMa'`, `'UFM'`, `'SIFT'`) to classes. Eliminates the
  `if/elif/else` chains in the constructor.
- [ ] Unify SIFT matching provider and RoMa/UFM flow providers under a single `MatchingProvider` interface. We only care about matches (source/target points + certainty), not raw flow warps. `FlowProviderDirect` and `SIFTMatchingProviderDirect` should share a common base with a `get_source_target_points` contract; the flow-specific internals (`compute_flow`, `_compute_raw`, warp format) become implementation details.
- [ ] `DataGraph` should be internal to onboarding — not exposed to detection or pose modules

#### 2.3 Break up CommonFrameData

- [ ] Split into per-concern structs: `FrameInput`, `SIFTFeatures`, `GroundTruth`, `FilteringState`, `FramePaths`,
  `FramePrediction`
- [ ] `CommonFrameData` can hold references to these if a single access point is still needed

#### 2.4 Decouple ViewGraph from descriptor model

- [ ] `ViewGraph.add_node` currently computes DINOv2 descriptors inline (line 42-48) — the data structure should not own
  the descriptor model
- [ ] Pass descriptors as input to `add_node`, or compute them in a separate step after construction

---

### Phase 3: Module B — Detection

Goal: clear separation between offline representation building (B1) and online inference (B2).

#### 3.1 Separate representation building from inference

- [ ] Extract condensation logic from `condensate_templates.py` into a `detection/representation.py` (or similar):
    - [ ] `build_detection_model(onboarding_result: OnboardingResult, ...) -> DetectionModel`
    - [ ] Condensation algorithms (Hart's CNN, imblearn) stay here
    - [ ] Statistical metadata computation (whitening, CSLS, Mahalanobis) stays here
- [ ] Extract detection inference from `pose_estimator.py` into `detection/detector.py`:
    - [ ] `detect(image, detection_model: DetectionModel) -> List[Detection]`
    - [ ] SAM proposal generation + descriptor matching + NMS
    - [ ] Rename or decompose `BOPChallengePosePredictor` — it only does detection despite the name

#### 3.2 Clean up BOP coupling

- [ ] BOP folder conventions (scene/image path construction, annotation loading) are scattered across
  `pose_estimator.py`, `condensate_templates.py`, and `run_*.py` scripts
- [ ] Extract into a `bop_data.py` adapter: `load_annotations`, `get_image_path`, `get_scene_folder`, etc.
- [ ] Detection module should accept images and models, not know about BOP folder layout

---

### Phase 4: Module C — Pose estimation

Goal: given detections and an onboarding result, produce 6DoF poses.

#### 4.1 Implement/reconnect pose estimation

- [ ] Uncomment and refactor the pose estimation call in `pose_estimator.py` (lines 211-216)
- [ ] Create `pose_estimation/estimator.py`:
    - [ ] 
      `estimate_poses(detections: List[Detection], onboarding_result: OnboardingResult, image, intrinsics) -> List[PoseEstimate]`
    - [ ] Template matching → PnP or flow-based alignment
- [ ] The flow provider (RoMa/UFM) is already initialized in `BOPChallengePosePredictor.__init__` — wire it up

#### 4.2 Evaluation

- [ ] Separate evaluation from pipeline execution
- [ ] Create `evaluation/` module:
    - [ ] `evaluate_detections(detections, ground_truth) -> DetectionMetrics` (BOP COCO)
    - [ ] `evaluate_poses(pose_estimates, ground_truth) -> PoseMetrics` (BOP 6DoF)
    - [ ] `evaluate_reconstruction(reconstruction, ground_truth) -> ReconstructionMetrics`
- [ ] Move `utils/eval_*.py` and `utils/bop_challenge.py` into this module

---

### Phase 5: Infrastructure improvements

These can be done in parallel with the module work.

#### 5.1 Visualization

- [ ] Unify `RerunAnnotations` and `RerunAnnotationsPose` into a single annotation constants module
- [ ] Extract shared rerun blueprint setup into a common function
- [ ] Extract shared matching visualization logic into a shared helper
- [ ] Decompose `results_logging.py` (~983 lines) into focused classes: `KeyframeVisualizer`, `Scene3DVisualizer`,
  `MatchingVisualizer`, `ImageLogger`

#### 5.2 Error handling

- [ ] Replace `assert` with `raise ValueError`/`RuntimeError` for runtime validation
- [ ] Replace bare `except:` with `except Exception:` in `metric3d.py:15` and `render_ho3d_segmantations.py:107`
- [ ] Add structured error handling for COLMAP reconstruction failures in `onboarding_pipeline.py`
- [ ] Review `exception_logger` — at minimum log the sequence that failed

#### 5.3 State management

- [ ] Review `DataGraphStorage.__setattr__` device-transfer magic — consider making it explicit
- [ ] Add synchronization to `app.py` module-level globals or refactor into a state class

#### 5.4 Web UI

- [ ] Update `app.py` to reflect the three-module architecture:
    - [ ] Tab 1: Onboarding (existing "Run on Dataset" + "Custom Input")
    - [ ] Tab 2: Detection (run detector on novel images using an onboarded model)
    - [ ] Tab 3: Pose estimation (given detections, estimate poses)
    - [ ] Tab 4: Browse results (existing)
