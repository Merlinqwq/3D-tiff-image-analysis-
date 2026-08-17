# Multichannel 3D TIFF image-analysis 

## Overview

This package provides reusable Python workflows for:

1. Segmenting nuclei or another primary object from a user-selected channel.
2. Measuring the intensity of other channels inside every segmented ROI.
3. Thresholding three-dimensional target objects inside existing ROIs.
4. Measuring individual-object volume and object count per ROI.
5. Comparing two channels using intensity- and mask-based colocalization.
6. Generating journal-style statistical figures.

The code does not require a particular input-directory name. A user may select
any directory containing TIFF files. Recursive batch mode can also find
TIFF-containing directories under a selected root.

The reusable copies are stored here; historical project-specific launchers are
isolated in `legacy_project_examples` and are not part of the general interface.

## AI-use disclosure

Generative AI (OpenAI Codex) assisted with development of this repository,
including code drafting and refactoring, debugging, test development,
documentation, and execution of the analysis workflows. AI-generated changes
were reviewed and tested before inclusion. Experimental design, selection and
approval of image-analysis parameters, visual inspection of segmentation and
thresholding results, exclusion decisions, and biological interpretation remain
the responsibility of the researcher. Users should independently validate the
software and all parameters for their own images before drawing scientific
conclusions or using the results in a publication.

## Package layout

```text
Project_pipelines/
|-- README.md
|-- 01_nucleus_segmentation_and_intensity/
|   |-- nucleus_intensity.py
|   |-- audit_batch_outputs.py
|   |-- test_nucleus_intensity.py
|   `-- legacy_project_examples/
|-- 02_intensity_merging_and_figures/
|   |-- prepare_merged_data.py
|   |-- audit_merged_outputs.py
|   |-- intensity_data_analysis.py
|   |-- test_intensity_data_analysis.py
|   `-- legacy_project_examples/
|-- 03_condensate_thresholding_and_figures/
|   |-- # 3D TIFF channel thresholding.py
|   |-- background_corrected_condensate_thresholding.py
|   |-- foci_data_analysis.py
|   |-- parameter_table_template.csv
|   `-- test_foci_data_analysis.py
`-- 04_two_channel_colocalization/
    |-- two_channel_3d_colocalization.py
    |-- colocalization_data_analysis.py
    |-- # 3D TIFF channel thresholding.py
    `-- background_corrected_condensate_thresholding.py
```

The two thresholding modules are duplicated in the colocalization directory
because the colocalization script imports their TIFF, mask, and background-
correction functions. This keeps that workflow self-contained.

## Requirements

```text
Python 3.10+
numpy
pandas
scipy
scikit-image
tifffile
openpyxl
matplotlib
```

`tkinter` is needed for graphical folder selection. Arial is optional but is
required to reproduce the default journal-figure typography exactly.

## Input selection

### Select one arbitrary folder

For nucleus segmentation, run without an input argument:

```powershell
python ".\Project_pipelines\01_nucleus_segmentation_and_intensity\nucleus_intensity.py"
```

A folder-selection window appears. The chosen folder may have any name, but its
source TIFF files must be located directly inside it.

The general 3D thresholding and background-corrected scripts also open a folder
picker when `--batch-root` is omitted. They additionally prompt for XY pixel
size and Z spacing because those values are required for volume calculations.

### Select a folder from the command line

```powershell
python ".\Project_pipelines\01_nucleus_segmentation_and_intensity\nucleus_intensity.py" `
  --input-folder "D:\experiment\images"
```

### Recursive batch mode

```powershell
python ".\Project_pipelines\01_nucleus_segmentation_and_intensity\nucleus_intensity.py" `
  --batch-root "D:\experiment"
```

Batch discovery:

- Finds directories containing TIFF files directly.
- Does not require a particular directory name.
- Ignores known output directories such as `Intensity`, `Masks`,
  `condensate_thresholding`, and `colocalization_analysis`.
- If both a parent and a descendant contain TIFF files, chooses the leaf-most
  directory by default to reduce duplicate processing.
- Can be restricted to an exact directory name using `--folder-name NAME`.

## TIFF and channel requirements

- TIFF axis metadata should identify `Y` and `X` and preferably `C` and `Z`.
- If `C` is absent, the software attempts to infer a 2- or 3-channel axis.
- The segmentation channel is selected by the user with
  `--segmentation-channel`.
- Other channel names are supplied with repeatable `--channel-name` options.
- If a channel name is not supplied and cannot be inferred from folder text,
  the neutral labels `ch2`, `ch3`, and so on are used.

Example with nuclei in channel 3 and two measurement channels:

```powershell
python ".\Project_pipelines\01_nucleus_segmentation_and_intensity\nucleus_intensity.py" `
  --input-folder "D:\experiment\images" `
  --segmentation-channel 3 `
  --channel-name "1=Protein_A" `
  --channel-name "2=Protein_B"
```

Channel numbering is one-based, matching common microscopy software.

## User-configurable parameters

The values below are interfaces, not mandatory biological settings. Users must
choose parameters appropriate for their microscope, magnification, staining,
cell type, and scientific question.

### Primary-object segmentation

| Option | Purpose | Default |
|---|---|---:|
| `--segmentation-channel` | Channel used to segment nuclei/primary objects | 1 |
| `--min-area` | Minimum accepted 2D object area in pixels | 5000 |
| `--sigma` | Gaussian smoothing sigma in pixels | 1.0 |
| `--no-fill-holes` | Disable filling holes in the thresholded mask | off |
| `--clear-border` | Remove objects touching the image edge | off |
| `--no-watershed` | Disable separation of touching objects | off |
| `--watershed-min-distance` | Minimum distance between watershed markers | 65 |
| `--min-nucleus-radius` | Minimum distance-transform height for markers | 5 |
| `--expected-count` | Optional expected object count used only for QA | unset |
| `--qa-exempt-date` | Date label exempt from expected-count QA | unset |

The defaults are starting values inherited from development and should be
validated visually. The pipeline never forces segmentation to match
`--expected-count`.

### Spatial calibration

The 3D thresholding and colocalization scripts require the user to provide:

```text
--xy-pixel-size <micrometers per pixel>
--z-step <micrometers per slice>
```

These values are not inferred from acquisition dates. When graphical folder
selection is used, the user is prompted to enter them. When command-line mode
is used, both options are required.

Voxel volume is calculated as:

```text
voxel_volume_um3 = xy_pixel_size_um x xy_pixel_size_um x z_step_um
```

### Target threshold and object-size parameters

The general 3D thresholding workflow reads a CSV table relative to each selected
TIFF folder. The default relative path is:

```text
channel_analysis_results/combined_channel_object_analysis.csv
```

Use `--parameter-table RELATIVE_PATH` to choose a different relative location.
The path must be relative so every selected TIFF folder can have its own table.

Required columns:

| Column | Meaning |
|---|---|
| `channel` | One-based channel, such as `C2` or `2` |
| `threshold` | Absolute intensity threshold |
| `min_object_size_voxels` | Minimum accepted 3D object size |
| `max_object_size_voxels` | Maximum accepted 3D object size |

Optional column:

| Column | Meaning |
|---|---|
| `target` | User-defined target name; otherwise inferred or reported as `ch#` |

A template is provided at
`03_condensate_thresholding_and_figures/parameter_table_template.csv`.

No protein-specific threshold or size-filter override is applied by the
reusable code. Different selected folders may use different thresholds. Size
filters are required to match for the same named target by default; use
`--allow-varying-filters` if intentional differences are required.

---

# Workflow 1: Nucleus segmentation and intensity measurement

## Algorithm

For each TIFF:

1. Read axes and sum-project every channel across Z.
2. Smooth the selected segmentation-channel projection.
3. Calculate an Otsu threshold independently for that image.
4. Optionally fill internal holes.
5. Remove objects below the selected minimum area.
6. Optionally clear border-touching objects.
7. Optionally separate touching objects with marker-controlled watershed.
8. Filter invalid post-watershed fragments.
9. Save projections, masks, and ROI boundaries.
10. Measure other channels inside every ROI.

Example:

```powershell
python ".\Project_pipelines\01_nucleus_segmentation_and_intensity\nucleus_intensity.py" `
  --input-folder "D:\experiment\images" `
  --segmentation-channel 1 `
  --channel-name "2=Target_A" `
  --channel-name "3=Target_B" `
  --min-area 2500 `
  --sigma 1.2 `
  --watershed-min-distance 40 `
  --min-nucleus-radius 4
```

Outputs are created under the selected input directory:

```text
selected_input_folder/Intensity/
|-- nuclear_intensity_results.xlsx
|-- Projections/
|-- Masks/
`-- ROIs/
```

The workbook contains `Nuclei`, `Images`, `Settings`, and `Errors` sheets.
Every output records the parameters actually used.

---

# Workflow 2: merge and analyze intensity measurements

Prepare integrated-intensity tables from all per-folder workbooks below a
selected root:

```powershell
python ".\Project_pipelines\02_intensity_merging_and_figures\prepare_merged_data.py" `
  "D:\experiment" "D:\experiment\prepared_merge_tables"
```

The script no longer excludes named targets or dates. It groups workbooks by
their measured target columns and records source hashes in `manifest.json`.
If no eight-digit date is present in the path, acquisition date is reported as
`unknown`.

Audit prepared/merged outputs:

```powershell
python ".\Project_pipelines\02_intensity_merging_and_figures\audit_merged_outputs.py" `
  "D:\experiment\prepared_merge_tables\manifest.json"
```

Analyze any compatible merged workbook:

```powershell
python ".\Project_pipelines\02_intensity_merging_and_figures\intensity_data_analysis.py" `
  "D:\experiment\Merged_targets.xlsx" `
  "D:\experiment\figures\intensity" `
  "intensity_analysis"
```

Historical target-specific launchers are retained only in
`legacy_project_examples`; new users should call `intensity_data_analysis.py`
directly.

---

# Workflow 3: 3D target-object thresholding

This step expects existing 2D label masks from Workflow 1. Each ROI is extended
through the Z stack. Target objects are thresholded in 3D and assigned to ROIs.

## Absolute-threshold mode (for clear background image)

```powershell
python ".\Project_pipelines\03_condensate_thresholding_and_figures\# 3D TIFF channel thresholding.py" `
  --batch-root "D:\experiment" `
  --xy-pixel-size 0.10 `
  --z-step 0.30 `
  --gaussian-sigma 1.0
```

Additional options:

- `--no-fill-holes`: do not fill holes in target masks.
- `--allow-varying-filters`: permit different size filters for the same target.
- `--include-date` / `--exclude-date`: optional metadata filters when an
  eight-digit date is present in a path.
- `--folder-name`: optional directory-name restriction; no name is required.
- `--parameter-table`: alternate relative parameter-table path.

## Background-corrected shared-threshold mode (for high nucleoplasm signal image)

This mode subtracts a local background and derives a shared robust threshold
from all selected folders in each target/replicate group.

```powershell
python ".\Project_pipelines\03_condensate_thresholding_and_figures\background_corrected_condensate_thresholding.py" `
  --batch-root "D:\experiment" `
  --xy-pixel-size 0.10 `
  --z-step 0.30 `
  --target "Target_A" `
  --background-window 31 `
  --target-background-window "Target_A=41" `
  --threshold-k-mad 5 `
  --gaussian-sigma 1 `
  --threshold-sample-stride 50
```

User-adjustable options:

| Option | Meaning |
|---|---|
| `--target NAME` | Include only this target; repeatable; omit for all targets |
| `--background-window` | Default odd local-background window in pixels |
| `--target-background-window NAME=PX` | Per-target window override |
| `--threshold-k-mad` | Robust MAD multiplier used for the shared threshold |
| `--gaussian-sigma` | XY Gaussian smoothing sigma |
| `--threshold-sample-stride` | Sampling stride used while estimating threshold |
| `--allow-varying-filters` | Permit inconsistent target size filters |

This mode does not perform condensate watershed and does not fill target-mask
holes. Those methodological choices should be validated for each application.

Per-folder output:

```text
selected_input_folder/condensate_thresholding/
|-- nucleus_focus_counts.csv
|-- focus_volumes.csv
|-- parameters_used.csv
|-- errors.csv
`-- Masks/
```

`nucleus_focus_counts.csv` retains ROIs with zero accepted objects.
`focus_volumes.csv` contains one row per individual 3D object and the ROI ID to
which it belongs.

Analyze merged focus volume and counts:

```powershell
python ".\Project_pipelines\03_condensate_thresholding_and_figures\foci_data_analysis.py" `
  --root "D:\experiment" `
  --output "D:\experiment\figures\foci"
```

---

# Workflow 4: two-channel 3D colocalization

This workflow requires:

- Original multichannel TIFF stacks.
- Existing 2D ROI label masks in `Intensity/Masks`.
- Two existing 3D binary target masks in `condensate_thresholding/Masks`.

Target names, channel numbers, spatial calibration, and local-background
windows are all selected by the user:

```powershell
python ".\Project_pipelines\04_two_channel_colocalization\two_channel_3d_colocalization.py" `
  --batch-root "D:\experiment" `
  --target-a "Target_A" --channel-a 2 `
  --target-b "Target_B" --channel-b 3 `
  --xy-pixel-size 0.10 `
  --z-step 0.30 `
  --background-window-a 31 `
  --background-window-b 41
```

Measurements per ROI:

- Background-corrected whole-ROI Pearson correlation.
- Raw-intensity whole-ROI Pearson correlation for audit.
- Background-corrected Pearson correlation inside the union of both masks.
- Dice coefficient.
- Jaccard index.
- Directional volume-overlap fractions.
- Directional object-overlap counts and fractions.
- Directional nearest-centroid distances using anisotropic voxel spacing.

Outputs:

```text
selected_input_folder/colocalization_analysis/
|-- per_roi_colocalization.csv
|-- per_image_summary.csv
`-- method_parameters.csv

selected_root/two_channel_colocalization_results/
|-- per_roi_colocalization.csv
|-- per_image_summary.csv
`-- errors.csv
```

Create the journal-style Pearson figure:

```powershell
python ".\Project_pipelines\04_two_channel_colocalization\colocalization_data_analysis.py" `
  --source "D:\experiment\two_channel_colocalization_results\per_roi_colocalization.csv" `
  --output "D:\experiment\figures\colocalization" `
  --condition-a "control" --condition-b "treatment" `
  --condition-a-label "Control" --condition-b-label "Treatment"
```

If the condition options are omitted, the plotting script automatically uses
the two unique condition values present in the source CSV.

Pearson correlation is not a percentage of overlap. It should be interpreted
together with Dice, Jaccard, directional overlap, appropriate single-channel
controls, and—when required—spatial-randomization controls.

---

# Figure and statistical defaults

The supplied figure scripts currently use:

- Individual observations rather than replicate means.
- Two-sided unpaired parametric t-test with equal variance assumed.
- Arithmetic mean with SEM.
- Opaque individual points.
- Significance symbols instead of numeric p-values on the graph.
- 5 cm by 5 cm individual panels.
- PNG, PDF, and editable SVG exports.

These are analysis defaults, not universally correct experimental-design
choices. In particular, nested measurements from multiple cells within the
same biological replicate may require a hierarchical model or replicate-level
analysis. Users should choose the statistical unit before publication.

# Safety

- Source TIFF files are opened read-only.
- Generated outputs are written into named output directories below the
  selected folder or root.
- Rerunning a workflow may replace files inside that workflow's own output
  directory.
- Parameter tables, settings sheets, method tables, errors, and masks should be
  archived with final results.
- Always inspect segmentation masks visually before interpreting measurements.

# Validation

```powershell
python -m compileall -q ".\Project_pipelines"

python -m unittest discover `
  -s ".\Project_pipelines\01_nucleus_segmentation_and_intensity" `
  -p "test_*.py"

python -m unittest discover `
  -s ".\Project_pipelines\02_intensity_merging_and_figures" `
  -p "test_*.py"

python -m unittest discover `
  -s ".\Project_pipelines\03_condensate_thresholding_and_figures" `
  -p "test_*.py"
```
