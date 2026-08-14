# 3D TIFF image Channel thresholding.
This script measures fixed-threshold 3D objects in channel 2, channel 3, or both channels of every TIFF image in a selected folder.

## Analysis workflow

At startup, the script:

1. Opens a popup with buttons for **Channel 2**, **Channel 3**, or **Both channels 2 and 3**.
2. Opens a folder picker for the input images.
3. Requests a separate fixed intensity threshold for each selected channel.

For each selected channel in every image, the script:

1. Extracts the channel from an image with axis order `(Z, C, Y, X)`.
2. Applies a Gaussian blur (`sigma = 1.0`) independently to each Z slice.
3. Applies that channel's user-entered fixed intensity threshold.
4. Fills enclosed holes in the 3D binary objects.
5. Removes face-connected 3D objects outside that channel's code-defined size range.
6. Saves the filled and size-filtered binary mask.
7. Records the retained object count and each object's filled volume.

C2 and C3 use independent lower and upper size limits.

## Installation

Install Python 3 and the required packages:

```powershell
python -m pip install numpy pandas tifffile scipy scikit-image
```

Tkinter is included with standard Windows Python installations.

## Run

```powershell
python "# 3D TIFF channel thresholding.py"
```

When both channels are selected, separate C2 and C3 threshold popups appear. Each entered threshold is applied to its channel across every input image. Object-size limits are edited directly in the script, not entered through a popup.

## Input

- TIFF axis order must be `(Z, C, Y, X)`.
- Images must contain the selected channels. Processing C3 requires at least three channels.
- Every `.tif` and `.tiff` file directly inside the selected folder is processed.

## Output

Results are saved in `channel_analysis_results` inside the selected input folder. For an input called `image_1.tif` with both channels selected:

```text
selected_input_folder/
`-- channel_analysis_results/
    |-- image_1_C2_thresholded.tif
    |-- image_1_C2_objects.csv
    |-- image_1_C3_thresholded.tif
    |-- image_1_C3_objects.csv
    `-- combined_channel_object_analysis.csv
```

The combined CSV contains all selected channels and all input images:

| Column | Meaning |
| --- | --- |
| `image_name` | Original TIFF filename |
| `channel` | Analyzed channel (`C2` or `C3`) |
| `threshold` | Fixed intensity threshold entered for that channel |
| `min_object_size_voxels` | Channel-specific inclusive lower size limit |
| `max_object_size_voxels` | Channel-specific inclusive upper size limit |
| `object_count` | Number of retained objects in that image and channel |
| `object_id` | Connected-object label within the image and channel |
| `volume_voxels` | Object volume as a foreground voxel count |
| `volume_um3` | Calibrated object volume in cubic micrometers |

An image-channel combination with no retained objects receives one row with `object_count = 0`, an empty `object_id`, and zero volume. Thresholded TIFFs are 8-bit masks with background `0` and retained foreground objects `255`.

## Channel-specific size limits

Edit the dictionaries near the top of the script:

```python
MIN_OBJECT_SIZE_VOXELS = {
    2: 600,
    3: 500,
}
MAX_OBJECT_SIZE_VOXELS = {
    2: 10_000,
    3: 100_000,
}
```

For example, changing only `MAX_OBJECT_SIZE_VOXELS[3]` changes the upper cutoff for C3 without affecting C2. Both boundaries are inclusive:

```text
minimum <= volume_voxels <= maximum
```

Hole filling occurs before size filtering. Therefore, the filled object volume determines whether an object passes its size limits. Size filtering then occurs before mask export and measurement, so masks and CSV files contain only filled objects within the selected channel's size range.

## Calibration and other settings

```python
GAUSSIAN_SIGMA = 1.0
RESULT_FOLDER_NAME = "channel_analysis_results"
XY_PIXEL_SIZE_UM = 0.108333
Z_STEP_UM = 0.2
```

Each voxel represents:

```text
0.108333 um x 0.108333 um x 0.2 um = 0.0023472078 um^3
```

Physical volume is calculated as `volume_voxels * 0.0023472078 um^3` and rounded to six decimal places.
