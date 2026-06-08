# CLAUDE.md - GloPose

## Project Overview

GloPose is a model-free 6DoF object pose estimation system following
the [BOP benchmark paradigm](https://bop.felk.cvut.cz/home/). It takes a video/image stream of a novel object and
produces posed 3D reconstructions and detection-ready representations, which are then used to detect and estimate 6DoF
poses of the object in novel images.

## Setup

- **Environment:** Conda (`environment.yml`), env name `glopose`, Python 3.13
- **Install:** `conda env create -f environment.yml && conda activate glopose`
- **Submodules:** `repositories/` contains git submodules (cnos, mast3r, SAM2, vggt, ho3d, sam3d, map-anything) — some installed as
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
| A. Onboarding      | `onboarding/pipeline.py` (orchestrator), `onboarding/reconstruction.py` (SfM), `onboarding/frame_filter.py`, `data_providers/`, `adapters/{vggt,mast3r,sam3d}_adapter.py` | Working. Evaluation extracted to `eval/` module. Reconstruction methods: COLMAP, VGGT, Mast3r, SAM3D. Frame filters: dense_matching, RANSAC, passthrough, SIFT, max_visible. |
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
  `dino_descriptor.py` (vendored DINOv2/v3 model), `dino_utils.py` (vendored utilities), `sam2_adapter.py` (SAM2),
  `hot3d_adapter.py` (HOT3D FISHEYE624→pinhole undistortion via `hand_tracking_toolkit`),
  `sam3d_adapter.py` (SAM3D single-image 3D reconstruction), `vggt_adapter.py` (VGGT), `mast3r_adapter.py` (Mast3r),
  `map_anything_adapter.py` (Map Anything — unified wrapper for mapanything/vggt/moge/dust3r/mast3r/must3r/pow3r/pi3/anycalib/da3 backends)
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

The project runs on an RCI GPU (SLURM) cluster. **Claude CAN run code on RCI** — it has
non-interactive SSH access to the login node and submits jobs with `sbatch`.

**SSH access (Claude, non-interactive):**

```bash
ssh -i ~/ssh/ssh_fel jelint19@login3.rci.cvut.cz '<command>'
```

The private key is `~/ssh/ssh_fel` (NOT in `~/.ssh/`; the only credential RCI accepts — the default
`~/.ssh/git` key is rejected). `login3.rci.cvut.cz` is the login node. The mounts (below) are set up by
`~/rci.sh`, which also drops the user into an interactive shell. Submit jobs from the login node:
`ssh ... 'sbatch <script>'`; monitor with `squeue -j <id>` / `sacct -j <id>`. There are no SLURM
clients locally — everything goes through the login node.

**Code sync (local → RCI):** the RCI copy at `~/projects/GloPose/` is **not** a git repo; it is kept in
sync from the local repo by **PyCharm SFTP auto-upload** (`.idea/deployment.xml`) — files Claude edits
locally appear on RCI automatically within seconds. Manual fallback: write through the RW mount at
`~/rci_home/projects/GloPose/`. Always verify the edit landed on RCI (e.g. `grep` the changed line in
`~/rci_home/projects/GloPose/...`) before launching a job.

**Running a batch job — module setup gotcha:** compute nodes do **not** initialize Lmod (`MODULEPATH`
is empty, `~/.bashrc` early-returns for non-interactive shells, and SLURM strips `-l` from the shebang).
An sbatch script must set up modules explicitly before loading them. numpy/torch/pycolmap come from the
**modules**, not the venv. Working template:

```bash
#!/bin/bash
#SBATCH --partition=gpufast        # also: h200fast, amdgpufast
#SBATCH --gres=gpu:1               # gpufast REQUIRES a GPU request
#SBATCH --mem=128G --cpus-per-task=12 --time=4:00:00
#SBATCH --output=/home/jelint19/logs/%x_%j.out
#SBATCH --error=/home/jelint19/logs/%x_%j.err

export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
ml purge
ml $(<~/scripts/modules_glotracker_foss2025b.txt)   # PyTorch-Lightning/2.5.5-foss-2025b, COLMAP/3.13.0, etc.
source ~/envs/glopose/bin/activate                  # venv layered on the modules
cd ~/projects/GloPose/
python run_bop_HANDAL_onboarding.py --config configs/onboarding/roma_c0975r05.py \
  --sequences obj_000001_dynamic obj_000017_dynamic --experiment my_experiment
```

`#!/bin/bash -l`, `source /etc/profile`, and `source ~/.bashrc` all FAIL to provide `ml` on compute
nodes — only the explicit `MODULEPATH` + Lmod-init recipe above works.

**Interactive (for the user, not Claude):**

```bash
~/run_gpu.sh -p glotracker -d gpu      # gpufast partition, 1 GPU, 128G RAM
~/run_gpu.sh -p glotracker -d h200     # h200fast partition, 512G RAM
~/run_gpu.sh -p glotracker -d amd      # amdgpufast partition, 256G RAM
# Add -n 2 for multi-GPU
```

This calls `srun --pty bash --init-file ...` and drops into an interactive shell with modules + venv loaded.

**Results:** `/mnt/personal/jelint19/results/FlowTracker/<experiment>/<dataset>/<sequence>/`
(readable locally at `~/rci_data/results/...` or `/mnt/personal/jelint19/results/...`).
**SLURM logs:** write `--output`/`--error` to `~/logs/` on RCI, never into the results folder.

### sshfs Mounts (local views of RCI)

`~/rci.sh` sshfs-mounts four RCI paths locally (all via `login3.rci.cvut.cz`):

| Local mount | RCI path | Access | Contents |
|-------------|----------|--------|----------|
| `~/rci_home/` | `/home/jelint19` | **RW** | RCI home: `projects/GloPose`, `logs/`, `scripts/`, `envs/glopose` — Claude stages batch scripts here |
| `~/rci_data/` and `/mnt/personal/jelint19/` | `/mnt/personal/jelint19` | read-only by convention | results, caches, weights, data — **never clobber** |
| `~/public_datasets/` | `/mnt/data/` | read-only | BOP datasets at `vrg/public_datasets/bop/` |
| `~/rci_archive/` | `/mnt/archive/jelint19` | read-only | archive |

- **Claude may write** to `~/rci_home/` (RCI home — for batch scripts, logs dir, manual code staging).
- **Claude must NEVER clobber** `/mnt/personal/jelint19/` (results/caches/weights/data) — read only.

**Directory layout of `/mnt/personal/jelint19/` (= `~/rci_data/`):**

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

- RoMa and UFM models are too large for the local GPU — run GPU-heavy work on RCI via `sbatch` (see above)
- Only passthrough onboarding (no dense matching) can run locally
- Code is edited locally and auto-synced to RCI (PyCharm SFTP); GPU runs happen on RCI

---

## Known Issues & Architectural Notes

### External Repo Integration

cnos integration is centralized in `adapters/cnos_adapter.py` — the sole location for `sys.path` manipulation
and cnos imports. Scoring functions are vendored in `detection/scoring.py`, small utilities in `utils/mask_utils.py`
and `utils/bbox_utils.py`. No other files import cnos internals directly.

SAM2 video predictor is wrapped in `adapters/sam2_adapter.py` — the sole location for SAM2 imports and
Hydra config initialization. `SAM2SegmentationProvider` in `data_providers/frame_provider.py` imports
`build_video_predictor` and `mask_to_sam_prompt` from the adapter.

SAM3D single-image 3D reconstruction is wrapped in `adapters/sam3d_adapter.py` — the sole location for
SAM3D imports. Takes one RGB image + segmentation mask, runs SAM3D inference, converts the output mesh
to `pycolmap.Reconstruction` with all mesh vertices as 3D points. Handles PyTorch3D→OpenCV coordinate
frame conversion. Used via `reconstruction_method = 'sam3d'` with `frame_filter = 'max_visible'`
(selects the single frame with the most visible object pixels). Alignment/scale recovery is not yet
implemented (see TODO Phase 3.1).

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
- [ ] Gradually migrate functions from `utils/bop_challenge.py` into `utils/bop_data.py` (clean extraction already started)

---

### Phase 4: Infrastructure improvements

These can be done in parallel with the module work.

#### 4.3 Visualization

- [ ] **Update Rerun SDK from ~0.21 to 0.30** — review breaking API changes (blueprint API, logging API,
  annotation classes, `rr.init`/`rr.spawn` signatures) and update all call sites in `results_logging.py`,
  `visualizations/pose_estimation_visualizations.py`, and `visualizations/rerun_utils.py`

---

## Publication

The paper and its project page live **outside this repo** (each its own git repo, siblings of
`GloPose/`). Set up 2026-06-08.

### Paper title & framing

**Dense Matchers for Dense Object Reconstruction** — Tomáš Jelínek, Dmytro Mishkin, Jiří Matas
(Visual Recognition Group, CTU in Prague). WACV 2026 submission (in rebuttal). Three claimed
contributions (from `sections/Intro.tex`):
1. A simple, effective method for dense 3D object reconstruction from videos.
2. A **matchability criterion** for online sparse-keyframe selection and view relocalization.
3. A **sparse view graph** that speeds up reconstruction and beats the complete (all-to-all) edge set.

("GloPose" is the codebase name; the paper/page brand may differ and can be finalized later.)

### Project page (GitHub Pages)

- Repo `github.com/tjelinek/glopose-page` (public) → live at
  **https://tjelinek.github.io/glopose-page/**. Local clone: `/home/tom/Projects/glopose-page`.
- Static Nerfies/Academic-Project-Page template (HTML/CSS/JS, `.nojekyll`). Title/authors/abstract/
  BibTeX mirror the paper. Media sections are commented out until real assets exist; open items are
  marked `TODO(...)` in `index.html` (venue, arXiv/PDF links, teaser + 1200×630 social image, favicon).
- **Remote is SSH** (`git@github.com:tjelinek/glopose-page.git`) — the HTTPS credential helper does
  not work on the local machine. Edit `index.html` → `git push` → Pages auto-redeploys.

### Paper source (Overleaf)

- Overleaf project `68c3d833f2e4e1ed3923c814`, cloned to `/home/tom/Projects/glopose-paper`.
  Remote `origin` → `https://git@git.overleaf.com/68c3d833f2e4e1ed3923c814` (Premium git remote,
  direct — no GitHub hop). Branch `master`. Sources: `main.tex`,
  `sections/{Intro,RelatedWork,Method,Experiments}.tex` (abstract in `Intro.tex`), `main.bib`,
  figures in `imgs/` and `figs/`.
- **Workflow: `git pull` before editing**, then edit → commit → `git push` (lands in the Overleaf
  web editor immediately). Co-authors edit in the Overleaf web UI. Token is stored
  (`credential.helper store`) so sync is non-interactive. `glopose-paper/SYNC.md` documents this
  (local-only via `.git/info/exclude`, never pushed to Overleaf).
- Backup: `github.com/tjelinek/GloPosePaper` (private) — **manual** backup, not auto-synced.

### Auto-sync rule (standing authorization)

For the **project page** (`/home/tom/Projects/glopose-page`) and the **Overleaf paper**
(`/home/tom/Projects/glopose-paper`) — and **only** these two repos — commit and push changes
**automatically** after editing, without asking first. (This overrides the default "commit/push only
when asked"; it does **not** apply to the main `GloPose` code repo or any other repo, where the
default still holds.)

- **Page:** after editing, `git add -A && git commit && git push` (remote is SSH). Pages auto-deploys.
- **Paper:** `git pull` **first** (avoid clobbering co-authors' web-editor edits), then edit →
  `git commit` → `git push` (lands in Overleaf immediately). If `pull` reports conflicts, stop and
  surface them rather than force-anything.
- End commit messages with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer.
- Still **report** what was committed/pushed in the reply.

### Minimum viable paper — TODO

The narrative is reconstruction-first; each result below backs a stated contribution. Detailed
sub-steps live in the `## Paper: Onboarding Module Evaluation` checklist (P1–P6) — this is the
curated MVP subset.

**Core results (must-have):**
- [ ] **Main reconstruction result** — static onboarding on HANDAL (40 objects, `_up`) with RoMa:
  pose error, rot/trans accuracy, reconstruction success rate. → backs contribution 1. (see P1.1)
- [ ] **Ablation A — keyframe selection** — our matchability criterion vs every-N subsampling:
  accuracy vs mean #keyframes table + plot. → backs contribution 2. (see P3.1,
  `scripts/ablation_frame_selection.py`)
- [ ] **Ablation B — view-graph density** — sparse (ours) vs complete edge set: reconstruction
  quality + runtime, showing speedup *and* quality win. → backs contribution 3. (see P3.2)
- [ ] **3D reconstruction quality vs GT meshes** — accuracy / completeness / F-score, pose AUC on
  HANDAL. → strengthens contribution 1. (see P4.4, `scripts/eval_3d_reconstruction.py`)
- [ ] **One downstream-task demo** — the abstract claims the representation suffices for pose
  estimation / 6DoF tracking / self-relocalization; include at least one minimal quantitative or
  qualitative result. (see P5)

**Figures & tables (deliverables):**
- [ ] Teaser figure (also reused on the project page).
- [ ] Method/pipeline figure — `glopose-paper/figs/scheme.tex`.
- [ ] Qualitative reconstruction figure — input → selected keyframes → point cloud → GT overlay.
- [ ] Frame-selection ablation table + plot; reconstruction-quality table.

**Writing:**
- [ ] Experiments section written up from the tables above.
- [ ] Limitations + conclusion.
- [ ] Tighten Intro/Related Work/Method (drafts exist).

**Release gate (project page):**
- [ ] Add teaser + qualitative figures to `glopose-page` (un-comment media sections).
- [ ] Fill arXiv id + PDF link; switch BibTeX from `@misc` preprint to the published entry.

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

#### P4.2 Reconstruction-to-GT alignment

- [ ] For dynamic sequences or sequences without GT poses, implement ICP fallback alignment

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
