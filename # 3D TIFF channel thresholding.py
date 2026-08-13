"""Measure fixed-threshold objects in C2 and/or C3 of 3D TIFF images.

Expected input axis order: (Z, C, Y, X).
Object volume is reported as the number of connected foreground voxels.
"""

from pathlib import Path
from tkinter import Button, Label, Tk, Toplevel, filedialog, simpledialog

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian
from skimage.measure import label, regionprops


GAUSSIAN_SIGMA = 1.0
RESULT_FOLDER_NAME = "channel_analysis_results"
XY_PIXEL_SIZE_UM = 0.108333
Z_STEP_UM = 0.2
VOXEL_VOLUME_UM3 = XY_PIXEL_SIZE_UM * XY_PIXEL_SIZE_UM * Z_STEP_UM

# Inclusive size limits, defined independently for each channel.
MIN_OBJECT_SIZE_VOXELS = {
    2: 20,
    3: 20,
}
MAX_OBJECT_SIZE_VOXELS = {
    2: 100000,
    3: 100000,
}

CSV_COLUMNS = [
    "image_name",
    "channel",
    "threshold",
    "min_object_size_voxels",
    "max_object_size_voxels",
    "object_count",
    "object_id",
    "volume_voxels",
    "volume_um3",
]


def choose_channels(root):
    """Show a modal popup for selecting C2, C3, or both channels."""
    selected_channels = []
    dialog = Toplevel(root)
    dialog.title("Choose channels")
    dialog.resizable(False, False)

    Label(dialog, text="Which channel(s) should be processed?").pack(
        padx=24,
        pady=(18, 12),
    )

    def select(channels):
        selected_channels.extend(channels)
        dialog.destroy()

    Button(dialog, text="Channel 2", width=22, command=lambda: select([2])).pack(
        padx=24,
        pady=4,
    )
    Button(dialog, text="Channel 3", width=22, command=lambda: select([3])).pack(
        padx=24,
        pady=4,
    )
    Button(
        dialog,
        text="Both channels 2 and 3",
        width=22,
        command=lambda: select([2, 3]),
    ).pack(padx=24, pady=(4, 18))

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.grab_set()
    dialog.lift()
    dialog.focus_force()
    root.wait_window(dialog)

    if not selected_channels:
        raise SystemExit("No channel selection made.")

    return selected_channels


def choose_analysis_options():
    """Ask for channels, an input folder, and per-channel thresholds."""
    root = Tk()
    root.withdraw()
    root.update()

    try:
        channels = choose_channels(root)

        selected_folder = filedialog.askdirectory(
            parent=root,
            title="Select folder containing TIFF images",
        )
        if not selected_folder:
            raise SystemExit("No input folder selected.")

        thresholds = {}
        for channel_number in channels:
            threshold = simpledialog.askfloat(
                f"Channel {channel_number} threshold",
                f"Enter the fixed intensity threshold for channel {channel_number}:",
                parent=root,
                minvalue=0.0,
            )
            if threshold is None:
                raise SystemExit(
                    f"No threshold entered for channel {channel_number}."
                )
            thresholds[channel_number] = threshold

    finally:
        root.destroy()

    return Path(selected_folder), channels, thresholds


def find_tiff_images(input_folder):
    """Return TIFF files located directly inside the selected folder."""
    return sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def filter_objects_by_size(binary, min_object_size_voxels, max_object_size_voxels):
    """Keep 3D objects within the inclusive minimum and maximum size limits."""
    if min_object_size_voxels < 1:
        raise ValueError("Minimum object size must be at least 1 voxel.")
    if max_object_size_voxels < min_object_size_voxels:
        raise ValueError("Maximum object size cannot be smaller than minimum size.")

    labeled_objects = label(binary, connectivity=1)
    object_sizes = np.bincount(labeled_objects.ravel())

    keep_label = (
        (object_sizes >= min_object_size_voxels)
        & (object_sizes <= max_object_size_voxels)
    )
    keep_label[0] = False

    return keep_label[labeled_objects]


def measure_channel(
    image_path,
    channel_number,
    threshold,
    result_folder,
):
    """Threshold one channel, save its mask, and return one row per object."""
    image = tifffile.imread(image_path)

    if image.ndim != 4:
        raise ValueError(
            f"Expected (Z, C, Y, X), but {image_path.name} has shape {image.shape}."
        )

    channel_count = image.shape[1]
    if channel_count < channel_number:
        raise ValueError(
            f"{image_path.name} has {channel_count} channel(s); "
            f"channel {channel_number} is required."
        )

    channel_stack = image[:, channel_number - 1, :, :]
    blurred = np.stack(
        [
            gaussian(z_slice, sigma=GAUSSIAN_SIGMA, preserve_range=True)
            for z_slice in channel_stack
        ],
        axis=0,
    )

    min_object_size_voxels = MIN_OBJECT_SIZE_VOXELS[channel_number]
    max_object_size_voxels = MAX_OBJECT_SIZE_VOXELS[channel_number]

    binary = blurred > threshold
    binary = binary_fill_holes(binary)
    binary = filter_objects_by_size(
        binary,
        min_object_size_voxels,
        max_object_size_voxels,
    )
    labeled_objects = label(binary, connectivity=1)
    objects = regionprops(labeled_objects)
    object_count = len(objects)

    channel_name = f"C{channel_number}"
    mask_path = result_folder / f"{image_path.stem}_{channel_name}_thresholded.tif"
    tifffile.imwrite(
        mask_path,
        binary.astype(np.uint8) * 255,
        imagej=True,
        metadata={"axes": "ZYX"},
    )

    rows = [
        {
            "image_name": image_path.name,
            "channel": channel_name,
            "threshold": threshold,
            "min_object_size_voxels": min_object_size_voxels,
            "max_object_size_voxels": max_object_size_voxels,
            "object_count": object_count,
            "object_id": region.label,
            "volume_voxels": int(region.area),
            "volume_um3": round(float(region.area) * VOXEL_VOLUME_UM3, 6),
        }
        for region in objects
    ]

    # Preserve a record for images in which the threshold detects no objects.
    if not rows:
        rows.append(
            {
                "image_name": image_path.name,
                "channel": channel_name,
                "threshold": threshold,
                "min_object_size_voxels": min_object_size_voxels,
                "max_object_size_voxels": max_object_size_voxels,
                "object_count": 0,
                "object_id": pd.NA,
                "volume_voxels": 0,
                "volume_um3": 0.0,
            }
        )

    per_image_csv = result_folder / f"{image_path.stem}_{channel_name}_objects.csv"
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(per_image_csv, index=False)

    return rows


def main():
    input_folder, channels, thresholds = choose_analysis_options()
    image_paths = find_tiff_images(input_folder)

    if not image_paths:
        raise FileNotFoundError(
            f"No .tif or .tiff images were found in {input_folder}."
        )

    result_folder = input_folder / RESULT_FOLDER_NAME
    result_folder.mkdir(exist_ok=True)

    combined_rows = []
    for image_number, image_path in enumerate(image_paths, start=1):
        print(f"[{image_number}/{len(image_paths)}] Processing {image_path.name}")
        for channel_number in channels:
            print(f"    Channel {channel_number}")
            combined_rows.extend(
                measure_channel(
                    image_path,
                    channel_number,
                    thresholds[channel_number],
                    result_folder,
                )
            )

    combined_csv = result_folder / "combined_channel_object_analysis.csv"
    pd.DataFrame(combined_rows, columns=CSV_COLUMNS).to_csv(
        combined_csv,
        index=False,
    )

    print(f"Finished {len(image_paths)} image(s).")
    print(f"Results saved to: {result_folder}")
    print(f"Combined measurements: {combined_csv.name}")


if __name__ == "__main__":
    main()
