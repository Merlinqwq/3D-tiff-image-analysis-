# 3D TIFF image analysis pipelines

This repository contains two Python workflows for multichannel TIFF microscopy images:

1. **3D channel thresholding** measures fixed-threshold objects in channel 2,
   channel 3, or both channels throughout a Z stack.
2. **DAPI nucleus intensity analysis** sum-projects the stack, segments nuclei
   from channel 1, separates touching nuclei, and measures target-channel signal
   inside each nuclear ROI.

Both workflows use folder pickers, write results beside the selected inputs, and
do not modify the source TIFF files.

## Installation

Install Python 3, then install the shared dependencies from the repository root:

```powershell
python -m pip install -r requirements.md
```

Tkinter is included with standard Windows Python installations. If `python`
opens the Microsoft Store, install Python from python.org and enable
**Add Python to PATH**.

## Workflow 1: fixed-threshold 3D channel objects

Script: `# 3D TIFF channel thresholding.py`

Run:

```powershell
python "# 3D TIFF channel thresholding.py"
```

At startup, the script:

1. Opens a popup for **Channel 2**, **Channel 3**, or **Both channels 2 and 3**.
2. Opens a folder picker for the input images.
3. Requests a separate fixed intensity threshold for each selected channel.

For each selected channel in every image, it:

1. Reads an image with axis order `(Z, C, Y, X)`.
2. Applies a Gaussian blur (`sigma = 1.0`) independently to each Z slice.
3. Applies the user-entered threshold.
4. Fills enclosed holes in the 3D binary objects.
5. Removes face-connected objects outside the channel-specific size range.
6. Saves the filtered 3D mask and records object counts and volumes.

### 3D input and output

- Every `.tif` and `.tiff` directly inside the selected folder is processed.
- C3 processing requires at least three channels.
- Results are written to `channel_analysis_results` inside the input folder.

For an input named `image_1.tif` with both target channels selected:

```text
selected_input_folder/
`-- channel_analysis_results/
    |-- image_1_C2_thresholded.tif
    |-- image_1_C2_objects.csv
    |-- image_1_C3_thresholded.tif
    |-- image_1_C3_objects.csv
    `-- combined_channel_object_analysis.csv
```

The combined CSV contains:

| Column | Meaning |
| --- | --- |
| `image_name` | Original TIFF filename |
| `channel` | Analyzed channel (`C2` or `C3`) |
| `threshold` | Fixed intensity threshold entered for that channel |
| `min_object_size_voxels` | Inclusive lower size limit |
| `max_object_size_voxels` | Inclusive upper size limit |
| `object_count` | Retained-object count for the image and channel |
| `object_id` | Connected-object label within the image and channel |
| `volume_voxels` | Filled object volume in voxels |
| `volume_um3` | Calibrated volume in cubic micrometers |

An image-channel combination with no retained objects receives one CSV row with
`object_count = 0`, an empty `object_id`, and zero volume. Thresholded TIFFs are
8-bit masks with background `0` and retained foreground objects `255`.

### 3D size limits and calibration

The current code-defined inclusive size limits are:

```python
MIN_OBJECT_SIZE_VOXELS = {
    2: 20,
    3: 20,
}
MAX_OBJECT_SIZE_VOXELS = {
    2: 100_000,
    3: 100_000,
}
```

Hole filling occurs before size filtering, so filled volume determines whether
an object passes. Additional settings are:

```python
GAUSSIAN_SIGMA = 1.0
RESULT_FOLDER_NAME = "channel_analysis_results"
XY_PIXEL_SIZE_UM = 0.108333
Z_STEP_UM = 0.2
```

Each voxel represents approximately `0.0023472078 um^3`. Physical volume is
`volume_voxels * voxel_volume` and is rounded to six decimal places.

## Workflow 2: DAPI nuclear intensity

Folder: `nucleus_intensity_pipeline`

Run with a folder picker:

```powershell
cd nucleus_intensity_pipeline
python nucleus_intensity.py
```

You can also double-click `run_nucleus_intensity.bat` or specify one folder:

```powershell
python nucleus_intensity.py --input-folder "D:\path\to\tiffs"
```

The folder must contain 2- or 3-channel TIFF images. Channel 1 is DAPI. For each
image, the pipeline:

1. Reads TIFF axis metadata and accepts exactly 2 or 3 channels.
2. Sum-projects Z and any other non-channel/non-spatial axes into `C,Y,X`.
3. Smooths the DAPI projection and applies global Otsu thresholding.
4. Fills mask holes and removes objects smaller than the area cutoff.
5. Uses distance-transform watershed to separate touching nuclei.
6. Applies the area cutoff again to the final watershed ROIs.
7. Measures DAPI and channel 2/3 intensity inside every nucleus.
8. Saves projections, binary/label masks, ROI boundaries, and one Excel workbook.

### Nuclear segmentation defaults

The documented defaults below match `Settings` in `nucleus_intensity.py`:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--min-area` | `5000` px | Minimum final nuclear ROI area |
| `--sigma` | `1.0` px | Gaussian smoothing sigma |
| `--watershed-min-distance` | `65` px | Minimum spacing between watershed center markers |
| `--min-nucleus-radius` | `5` px | Minimum interior distance for a valid nuclear marker |
| `--clear-border` | off | Exclude nuclei touching an image border when enabled |
| `--no-watershed` | off | Disable touching-nucleus separation when supplied |

The folder-picker interface asks for the minimum area. All segmentation controls
remain editable through command-line options. Increasing watershed minimum
distance reduces false extra markers in very close nuclear clusters.

### Nuclear output

Results are stored under `Intensity` in the selected folder:

```text
selected_input_folder/
`-- Intensity/
    |-- Projections/*_sum_projection.tif
    |-- Masks/*_nuclei_binary_mask.tif
    |-- Masks/*_nuclei_label_mask.tif
    |-- ROIs/*_roi_boundaries.csv
    `-- nuclear_intensity_results.xlsx
```

The workbook contains:

- `Nuclei`: one row per nucleus with area, position, DAPI intensity, and target
  mean/integrated/median/minimum/maximum intensity.
- `Images`: source metadata, Otsu threshold, nucleus count, and output paths.
- `Settings`: exact segmentation settings used for the run.
- `Errors`: failed images while the remainder of the batch continues.

For three-channel data, the 488 target labels channel 2 and the 568 target labels
channel 3. For two-channel data, the only non-DAPI target labels channel 2. Target
names are inferred from the condition folder above `New folder`; the known folder
typo `NOP61` is normalized to `NOP16`.

Intensities are raw sum-projected pixel units without background subtraction.
Mean intensity is appropriate for comparing nuclei of different sizes, whereas
integrated intensity includes both signal and nuclear area.

### Recursive batch mode

Process every folder named `New folder` beneath an experiment root:

```powershell
python nucleus_intensity.py --batch-root "D:\path\to\experiment" --min-area 5000
```

Exclude specified targets when needed:

```powershell
python nucleus_intensity.py --batch-root "D:\path\to\experiment" --exclude-target UBF --exclude-target SRRM1
```

Batch mode creates `nuclear_intensity_batch_summary.csv`,
`nuclear_intensity_source_integrity.csv`, and `nuclear_intensity_count_QA.csv` at
the batch root. The current project-specific QA rule expects 25 nuclei per
target/treatment and marks 20260717 as `EXEMPT_DATE`. `REVIEW` is only a warning;
the pipeline never deletes or invents ROIs to force the expected count.

## Validation

Run the nuclear pipeline tests from its folder:

```powershell
cd nucleus_intensity_pipeline
python -m unittest -v test_nucleus_intensity.py
```

Always inspect segmentation masks before biological interpretation, especially
when staining intensity, acquisition settings, or nuclear morphology changes.
