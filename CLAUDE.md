# CLAUDE.md - GloPose

## Project Overview

GloPose is a 6DoF object pose tracking and 3D reconstruction system for video sequences. It combines dense optical flow, Structure-from-Motion (SfM), and template-based pose estimation to track objects in videos.

## Setup

- **Environment:** Conda (`environment.yml`), env name `glopose`, Python 3.13
- **Install:** `conda env create -f environment.yml && conda activate glopose`
- **Submodules:** `repositories/` contains git submodules (cnos, mast3r, Metric3D, SAM2, vggt, ho3d) — some installed as editable pip packages

## Running

- **Web UI:** `python app.py` (Gradio on localhost:7860)
- **Dataset scripts:** `python run_HANDAL.py --config configs/base_config.py --sequences SEQ --experiment EXP` (similarly for run_HO3D.py, run_NAVI.py, run_BOP_classic_onboarding.py, etc.)
- **Batch jobs:** `python scripts/job_runner.py`

## Testing

No formal test suite. Validation is done by running dataset-specific scripts and evaluation utilities (`utils/eval_*.py`, `utils/bop_challenge.py`).

## Architecture

### Pipeline (tracker6d.py)
```
Tracker6D orchestrates:
  FrameProvider → segmentation (SAM2) → frame filtering (optical flow)
  → SfM reconstruction (COLMAP/GloMAP) → pose estimation (CNOS templates)
```

### Key directories
- `configs/` — Python-based config files (not YAML), loaded via `utils.general.load_config()`
- `pose/` — Pose estimation, SfM (glomap.py), frame filtering, COLMAP utils
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

## Code Conventions

- **Type hints:** Used extensively, modern union syntax (`X | Y`)
- **Naming:** snake_case for functions/variables, PascalCase for classes
- **Imports:** Relative within package (`from tracker_config import TrackerConfig`), absolute for externals
- **Strings:** f-strings preferred
- **No linter/formatter configured** — match existing code style when making changes
- **Dataclasses** over plain dicts for structured data

## Key Dependencies

PyTorch, Kornia (geometry/camera), Kaolin (mesh rendering), pycolmap, SAM2, RoMa/UFM (optical flow), DINOv2 (via CNOS), NetworkX, Gradio, Rerun SDK, wandb

## Things to Know

- Configs are **Python files**, not YAML/JSON — they define variables that get loaded dynamically
- The `repositories/` directory contains external code — do not modify these
- Results go to `{results_root}/{experiment}/{dataset}/{sequence}/` with COLMAP database, reconstructions, and pose estimates
- Rerun SDK is used for 3D visualization logging alongside disk output
- Many operations are GPU-intensive; device selection is configurable via `TrackerConfig`
- **Project runs on RCI** (remote computing infrastructure), not locally. Paths like `/mnt/personal/jelint19/` and `/mnt/data/vrg/` are RCI mount points. Changes cannot be tested locally — only syntax-checked.
- Hardcoded RCI paths are scattered across many files beyond `tracker6d.py` (scripts, dataset generators, pose estimator, etc.). The `TrackerConfig` defaults (`default_data_folder`, `default_results_folder`, `default_cache_folder`) centralize the main ones.

---

## Known Issues & Architectural Notes

### Critical

- **Hardcoded user paths**: `tracker6d.py:51-63` has absolute paths to `/mnt/personal/jelint19/cache/...`. Must come from config.
- **Class name collision**: `PrecomputedUFMFlowProviderDirect` exists as two unrelated classes — one in `flow_provider.py:475` (actual UFM) and one in `matching_provider_sift.py:76` (SIFT-based). Not currently a runtime bug but a maintenance hazard.
- **Config bug**: `tracker_config.py:89` — `similarity_transformation = 'kabsch'` uses `=` instead of `:`, making it a class variable instead of a dataclass field (invisible to `__init__`, `asdict()`, etc.).

### Structural

- **`Tracker6D` tight coupling**: Constructor imports and instantiates 14+ concrete classes via `if/elif/else` chains. No dependency injection or factory pattern. Adding a new matcher/filter requires editing `tracker6d.py`.
- **`TrackerConfig` god-object**: ~47 flat fields spanning 8+ concerns (viz, input, rendering, mesh, filtering, matching, reconstruction, SIFT). Sub-configs exist (`BaseRomaConfig`, etc.) but most fields remain top-level.
- **`CommonFrameData` god-class** (`data_graph.py:32-72`): 20+ fields mixing input data, SIFT features, ground truth, filtering state, file paths, predictions, and timing.
- **`results_logging.py` (~983 lines)**: `WriteResults` has 10+ responsibilities — rerun blueprint layout (282 lines in `rerun_init` alone), keyframe viz, 3D camera viz, flow matching viz, image I/O, matplotlib helpers, silhouette rendering, math utilities. Should be 4-5 classes.
- **Duplicated visualization systems**: `results_logging.py` and `visualizations/pose_estimation_visualizations.py` have near-identical rerun init, blueprint setup, matching visualization logic, and overlapping annotation constants (`RerunAnnotations` vs `RerunAnnotationsPose`).

### Diamond Inheritance in Flow Providers

`flow_provider.py` uses diamond inheritance for precomputed caching:

```
      FlowProviderDirect (ABC)
           /          \
  RoMaFlowProvider  PrecomputedFlowProvider
           \          /
  PrecomputedRoMaFlowProvider
```

Same pattern for UFM. The `PrecomputedUFMFlowProviderDirect.compute_flow` (`flow_provider.py:486-520`) is a near-copy of `PrecomputedFlowProviderDirect.compute_flow` (`flow_provider.py:161-195`) to work around MRO issues — including a shared operator-precedence bug on `or`/`and` lines (187-189 / 512-514).

### External Repo Integration

All external repos are integrated via `sys.path.append('./repositories/...')` scattered across 10+ files. This is CWD-dependent, pollutes namespaces, and provides no insulation from API changes. Files that touch cnos internals: `view_graph.py`, `pose_estimation_visualizations.py`, `condensate_templates.py`, `bop_challenge.py`, `cnos_utils.py`.

### State Management

- `DataGraph` is a shared mutable hub — every pipeline stage freely mutates nodes and edges with no access control.
- `DataGraphStorage.__setattr__` silently moves tensors between devices on every assignment (action-at-a-distance).
- `app.py:62-65,151-152` has 5 module-level mutable globals shared between Gradio callbacks with no synchronization (except `_dataset_process_lock`).

### Error Handling

- `assert` used as runtime validation throughout (stripped with `python -O`).
- `runtime_utils.py:15-21` `exception_logger` catches `Exception` and silently continues.
- `app.py:113` has `except Exception: pass`.
- `tracker6d.py` has zero try/except — COLMAP failure is communicated only via `reconstruction is None` and print statements.
- `metric3d.py:15` and `render_ho3d_segmantations.py:107` have bare `except:` catching even `SystemExit`/`KeyboardInterrupt`.

---

## TODO: Fixes

### Critical (fix first)
- [x] Move hardcoded cache paths from `tracker6d.py:51-63` into `TrackerConfig` — added `default_cache_folder` field, tracker derives paths from it. Needs validation on RCI.
- [x] Fix `tracker_config.py:89`: change `similarity_transformation = 'kabsch'` to `similarity_transformation: str = 'kabsch'`
- [x] Rename `PrecomputedUFMFlowProviderDirect` in `matching_provider_sift.py` → `PrecomputedSIFTMatchingProvider`
- [x] Fix operator-precedence bug in `flow_provider.py` — added parens around the `or` clause so `and zero_certainty_outside_segmentation` applies to both branches

### Flow provider diamond inheritance → composition
- [x] Replaced diamond inheritance with composition: extracted `FlowCache` class handling disk + datagraph caching. `RoMaFlowProviderDirect` and `UFMFlowProviderDirect` accept optional `FlowCache`. Deleted `PrecomputedFlowProviderDirect`, `PrecomputedRoMaFlowProviderDirect`, `PrecomputedUFMFlowProviderDirect`. Moved `get_source_target_points_datagraph` to base class. Fixed `PrecomputedSIFTMatchingProvider` constructor to accept `BaseSiftConfig` instead of raw int. Needs validation on RCI.

### Config decomposition
- [ ] Group `TrackerConfig` fields into sub-configs: `VisualizationConfig`, `InputConfig`, `RendererConfig`, `MeshConfig`, `FrameFilterConfig`, `ReconstructionConfig`, `SIFTConfig`
- [ ] Keep flat access via `@property` delegation or `__getattr__` if needed for backwards compatibility

### Reduce tracker6d.py coupling
- [ ] Introduce a provider registry or factory: map config string values (e.g. `'RoMa'`, `'UFM'`, `'SIFT'`) to provider classes. `Tracker6D` calls the registry instead of hard-coding `if/elif/else` chains.

### Deduplicate visualization systems
- [ ] Unify `RerunAnnotations` and `RerunAnnotationsPose` into a single annotation constants module
- [ ] Extract shared rerun blueprint setup into a common function
- [ ] Extract shared matching visualization logic (template+target concat, certainty split, correspondence logging) into a shared helper. Both `WriteResults` and `PoseEstimatorLogger` should call it.

### Decompose results_logging.py
- [ ] Extract `rerun_init` blueprint layout into a dedicated blueprint factory
- [ ] Split `WriteResults` into focused classes: `KeyframeVisualizer`, `Scene3DVisualizer`, `MatchingVisualizer`, `ImageLogger`

### External repo adapters
- [ ] Create `adapters/cnos_adapter.py` — single location for `sys.path` manipulation and cnos imports. All other files import from the adapter.
- [ ] Same for Metric3D (`adapters/metric3d_adapter.py`), SAM2, ho3d as needed

### Break up CommonFrameData
- [ ] Split into per-concern structs: e.g. `FrameInput`, `SIFTFeatures`, `GroundTruth`, `FilteringState`, `FramePaths`, `FramePrediction`
- [ ] `CommonFrameData` can hold references to these if a single access point is needed

### Error handling
- [ ] Replace `assert` with `raise ValueError`/`RuntimeError` for runtime validation
- [ ] Replace bare `except:` with `except Exception:` (or narrower) in `metric3d.py:15` and `render_ho3d_segmantations.py:107`
- [ ] Add structured error handling for COLMAP reconstruction failures in `tracker6d.py`
- [ ] Review `exception_logger` — at minimum log the sequence that failed, consider whether silent continuation is always desired

### State management
- [ ] Review `DataGraphStorage.__setattr__` device-transfer magic — consider making it explicit
- [ ] Add synchronization to `app.py` module-level globals or refactor into a state class
