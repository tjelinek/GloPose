# Experiment Tracker

Tracks all experiments set up in the GloPose project.

## Experiments

| Date | Experiment | Config | Datasets | Paper section | Status | Notes |
|------|------------|--------|----------|---------------|--------|-------|
| 2026-03-15 | Baseline (UFM adaptive) | `onboarding/ufm_c0975r05` | BOP static+dynamic, classic, HO3D train, NAVI | Baseline | ready | Reference for all ablations |
| 2026-03-15 | Frame selection: every frame | `onboarding/passthroughs/every_frame` | BOP static+dynamic, classic, HO3D train, NAVI | P3.1 | ready | No subsampling |
| 2026-03-15 | Frame selection: every 2nd | `onboarding/passthroughs/every_2nd_frame` | BOP static+dynamic, classic, HO3D train, NAVI | P3.1 | ready | |
| 2026-03-15 | Frame selection: every 4th | `onboarding/passthroughs/every_4th_frame` | BOP static+dynamic, classic, HO3D train, NAVI | P3.1 | ready | |
| 2026-03-15 | Frame selection: every 8th | `onboarding/passthroughs/every_8th_frame` | BOP static+dynamic, classic, HO3D train, NAVI | P3.1 | ready | |
| 2026-03-15 | Frame selection: every 16th | `onboarding/passthroughs/every_16th_frame` | BOP static+dynamic, classic, HO3D train, NAVI | P3.1 | ready | |
| 2026-03-15 | Frame selection: every 32nd | `onboarding/passthroughs/every_32th_frame` | BOP static+dynamic, classic, HO3D train, NAVI | P3.1 | ready | |
| 2026-03-15 | Frame selection: every 64th | `onboarding/passthroughs/every_64th_frame` | BOP static+dynamic, classic, HO3D train, NAVI | P3.1 | ready | |
| 2026-03-15 | ViewGraph: dense (all-to-all) | `onboarding/ufm_c0975r05_dense` | BOP static+dynamic, classic, HO3D train, NAVI | P3.2 | ready | Compare against baseline from_matching |
| 2026-03-15 | Matching: RoMa | `onboarding/roma_c0975r05` | BOP static+dynamic, classic, HO3D train, NAVI | P3.3 | ready | Same thresholds as UFM baseline |
| 2026-03-15 | Matching: SIFT + LightGlue | `onboarding/sift_matching` | BOP static+dynamic, classic, HO3D train, NAVI | P3.3 | ready | Sparse matching comparison |
| 2026-03-15 | External: VGGT (adaptive, white bg) | `reconstruction/vggt` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: VGGT (adaptive, black bg) | `reconstruction/vggt_black_bg` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: VGGT (adaptive, original bg) | `reconstruction/vggt_original_bg` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: VGGT (every 8th, white bg) | `reconstruction/vggt_every_8th` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: VGGT (every 8th, original bg) | `reconstruction/vggt_every_8th_original_bg` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: Mast3r (adaptive, black bg) | `reconstruction/mast3r` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: Mast3r (adaptive, white bg) | `reconstruction/mast3r_white_bg` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: Mast3r (adaptive, original bg) | `reconstruction/mast3r_original_bg` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: Mast3r (every 8th, black bg) | `reconstruction/mast3r_every_8th` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | External: Mast3r (every 8th, original bg) | `reconstruction/mast3r_every_8th_original_bg` | BOP static+dynamic, classic, HO3D train, NAVI | P3.4 | ready | |
| 2026-03-15 | Background: black bg | `onboarding/ufm_c0975r05_bbg` | BOP static+dynamic, classic, HO3D train, NAVI | P3.5 | ready | Compare against original bg baseline |
| 2026-03-15 | Track merging: disabled | `onboarding/ufm_c0975r05_no_track_merging` | BOP static+dynamic, classic, HO3D train, NAVI | P3.6 | ready | Compare against baseline (enabled) |
| 2026-03-15 | RANSAC frame adding: pycolmap | `onboarding/ufm_ransac_pycolmap` | BOP static+dynamic, classic, HO3D train, NAVI | N/A | ready | RANSAC-based reliability |
| 2026-03-15 | RANSAC frame adding: MAGSAC++ | `onboarding/ufm_ransac_magsac` | BOP static+dynamic, classic, HO3D train, NAVI | N/A | ready | MAGSAC++ variant |
| 2026-03-15 | Seg filter + BA | `reconstruction/colmap_seg_filter` | BOP static+dynamic, classic, HO3D train, NAVI | N/A | ready | Filter out-of-seg points then BA |
| 2026-03-15 | Seg filter + BA + black bg | `reconstruction/colmap_seg_filter_black_bg` | BOP static+dynamic, classic, HO3D train, NAVI | N/A | ready | Combined with black background |
| 2026-03-15 | Fixed certainty threshold | `onboarding/ufm_c0975r05_fixed_threshold` | BOP static+dynamic, classic, HO3D train, NAVI | N/A | ready | Compare against Otsu (baseline) |
| 2026-03-15 | Matchability-based reliability | `onboarding/ufm_c0975r05_matchability` | BOP static+dynamic, classic, HO3D train, NAVI | N/A | ready | Matchability mask in reliability |

## Reconstruction Results

| Experiment | Dataset | Recon rate (%) | Rot err (deg) | Trans err (cm) | Rot acc@5 | Trans acc@5cm | Accuracy (mm) | Completeness (mm) | F-score@5mm | AUC@5 | AUC@10 | AUC@30 |
|------------|---------|----------------|---------------|-----------------|-----------|---------------|---------------|--------------------| ------------|-------|--------|--------|

## Detection Results

| Experiment | Dataset | AP50 | AP75 | AR | mAP |
|------------|---------|------|------|----|-----|

## Pose Estimation Results

| Experiment | Dataset | AR_VSD | AR_MSSD | AR_MSPD | AR (mean) |
|------------|---------|--------|---------|---------|-----------|
