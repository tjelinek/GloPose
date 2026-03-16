---
name: experiment
description: "Use this skill when the user says /experiment, asks to 'set up an experiment', 'create an experiment', 'add an ablation', or wants to make a code change that should be run as a configurable experiment on the cluster. This skill orchestrates the full workflow: implement code, make it configurable, create a config file, add it to the job runner, update CLAUDE.md, and log it in experiment_tracker.md."
version: 1.0.0
---

# Experiment Skill

You are setting up a new experiment in the GloPose project. Every experiment follows these six steps
in order. Complete each step fully before moving to the next. Ask the user for clarification if the
experiment description is ambiguous.

## Step 1: Implement the code change

- Read and understand the relevant code before making changes.
- Implement the change the user described. This could be anything: a new feature, a bug fix,
  a new adapter, a pipeline modification, a new evaluation metric, etc.
- Keep the change minimal and focused. Do not refactor surrounding code.
- Ensure the change works with the existing pipeline (runners, providers, pipeline stages).

## Step 2: Make it configurable

- Add new fields to the appropriate config dataclass(es) in `configs/glopose_config.py`.
  Pick the sub-config where the field logically belongs:
  - `OnboardingConfig` — frame filtering, reconstruction, matching settings
  - `CondensationConfig` — template condensation
  - `DetectionConfig` — detection inference
  - `PoseEstimationConfig` — pose estimation
  - `InputConfig` / `BaseFrameProviderConfig` — frame input, background
  - `VisualizationConfig` — output/logging
  - `RendererConfig` — mesh rendering
  - `RunConfig` — runtime params (device, experiment name)
  - `PathsConfig` — data/results paths
- Set the default value to preserve existing behavior (backward-compatible).
- Wire the new config field into the code change from Step 1 — the code should read from
  the config, not use hardcoded values.
- Follow existing patterns: use Python type hints, dataclass fields with defaults.

## Step 3: Create a config file

- Create a new Python config file under `configs/`. Choose the right subdirectory:
  - `configs/onboarding/` — onboarding/reconstruction experiments
  - `configs/reconstruction/` — external reconstruction method experiments
  - `configs/components/` — component-level configs
  - `configs/matching/` — matching method configs
  - Or a new subdirectory if none fits.
- Follow the standard pattern:

```python
from configs.glopose_config import GloPoseConfig

def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()
    # Override only the fields relevant to this experiment
    cfg.sub_config.new_field = new_value
    return cfg
```

- Name the config file descriptively. Use the naming conventions from existing configs
  (e.g., `ufm_c0975r05_bbg` encodes matcher, thresholds, and background variant).
- If the experiment has multiple variants (e.g., with/without a feature, different thresholds),
  create one config file per variant.

## Step 4: Add to job runner

- Edit `scripts/job_runner.py`:
  - Add the new config name(s) to the `configurations` list in `main()`.
  - Add them commented-out with a short comment indicating what experiment group they belong to.
  - If a new dataset is needed that isn't already in the `sequences` dict, add it.
- The config name is the path relative to `configs/` without the `.py` extension
  (e.g., `'onboarding/ufm_c0975r05_bbg'`).
- Tell the user which datasets to uncomment/enable when they want to run the experiment.

## Step 5: Update CLAUDE.md

- Check if the code change is relevant to the project documentation in `CLAUDE.md`.
- Update `CLAUDE.md` if:
  - A TODO item was completed — check the box `[x]` or remove it.
  - The architecture section needs updating (new module, new data type, new pattern).
  - A new key directory, file, or convention was introduced.
  - A known issue was resolved or a new one discovered.
- Do NOT update CLAUDE.md for minor config-only changes that don't affect architecture.

## Step 6: Add entry to experiment_tracker.md

`experiment_tracker.md` at the project root has four tables. Update the relevant ones:

### Experiments table
- Always add a row here.
- **Date**: today's date (YYYY-MM-DD)
- **Experiment**: short name describing what is being tested
- **Config**: config file path(s) relative to `configs/`, comma-separated if multiple variants
- **Datasets**: which dataset(s) from the job runner this should run on
- **Paper section**: which paper section this supports (e.g., P3.1, P3.5, D1), or "N/A"
- **Status**: one of `ready` (config created, not yet run), `running`, `done`, `failed`, `blocked`
- **Notes**: brief note on what to look for in results, or any caveats

### Results tables
- Leave the results tables empty at experiment creation time (results are filled in after runs complete).
- There are three results tables — fill in the appropriate one(s) when results are available:
  - **Reconstruction Results**: recon rate, rotation/translation errors, 3D quality metrics
  - **Detection Results**: AP, AR, mAP
  - **Pose Estimation Results**: BOP metrics (AR_VSD, AR_MSSD, AR_MSPD)

## After completing all steps

Summarize what was done:
1. What code was changed and where
2. What config field(s) were added
3. What config file(s) were created
4. What was added to job_runner.py
5. Whether CLAUDE.md was updated and why (or why not)
6. The experiment_tracker.md entry

Then ask the user if they want to commit the changes.
