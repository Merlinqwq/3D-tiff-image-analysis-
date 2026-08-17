"""Count thresholded 3D target foci inside each existing nucleus ROI.

Thresholds and voxel-size filters are read from each selected folder's
channel_analysis_results/combined_channel_object_analysis.csv. Existing 2D
nucleus label masks in Intensity/Masks are extended through Z as ROIs. New
outputs are written only to condensate_thresholding.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian
from skimage.measure import label, regionprops


GAUSSIAN_SIGMA = 1.0
FILL_TARGET_HOLES = True
RESULT_FOLDER_NAME = "condensate_thresholding"
XY_PIXEL_SIZE_UM = 1.0
Z_STEP_UM = 1.0
DEFAULT_MAX_OBJECT_SIZE_VOXELS = 100000
TIFF_SUFFIXES = {".tif", ".tiff"}
OUTPUT_DIRECTORY_NAMES = {
    "intensity", "projections", "masks", "rois", "channel_analysis_results",
    RESULT_FOLDER_NAME, "colocalization_analysis",
}
PARAMETER_TABLE_RELATIVE_PATH = Path(
    "channel_analysis_results/combined_channel_object_analysis.csv"
)

TARGET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*)\s*-?\s*(488|568)(?!\d)",
    re.IGNORECASE,
)
TARGET_NAME_CORRECTIONS: dict[str, str] = {}

NUCLEUS_COLUMNS = [
    "acquisition_date", "condition", "replicate", "target_group", "source_folder",
    "image_name", "nucleus_id", "target", "channel", "threshold",
    "min_object_size_voxels", "max_object_size_voxels", "xy_pixel_size_um",
    "z_step_um", "voxel_volume_um3", "focus_count", "total_focus_volume_voxels",
    "total_focus_volume_um3", "mean_focus_volume_voxels", "mean_focus_volume_um3",
]
FOCUS_COLUMNS = [
    "acquisition_date", "condition", "replicate", "target_group", "source_folder",
    "image_name", "nucleus_id", "target", "channel", "focus_id_within_nucleus",
    "threshold", "min_object_size_voxels", "max_object_size_voxels",
    "xy_pixel_size_um", "z_step_um", "voxel_volume_um3", "volume_voxels", "volume_um3",
]
PARAMETER_COLUMNS = [
    "acquisition_date", "condition", "replicate", "target_group", "source_folder",
    "target", "channel", "threshold", "min_object_size_voxels",
    "max_object_size_voxels", "xy_pixel_size_um", "z_step_um", "voxel_volume_um3",
    "gaussian_sigma_px", "parameter_source",
]
ERROR_COLUMNS = ["source_folder", "image_name", "target", "channel", "error"]


def _target_pairs(text: str) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for match in TARGET_PATTERN.finditer(text):
        raw = match.group(1)
        name = TARGET_NAME_CORRECTIONS.get(raw.casefold(), raw)
        wavelength = int(match.group(2))
        key = (name.casefold(), wavelength)
        if key not in seen and name.casefold() != "dapi":
            pairs.append((name, wavelength))
            seen.add(key)
    return pairs


def infer_target_names(input_folder: Path, channel_count: int) -> dict[int, str]:
    candidates = [_target_pairs(input_folder.parent.name), _target_pairs(input_folder.parent.parent.name)]
    if channel_count == 2:
        for pairs in candidates:
            names = list(dict.fromkeys(name for name, _ in pairs))
            if len(names) == 1:
                return {2: names[0]}
    elif channel_count == 3:
        for pairs in candidates:
            by_wavelength = {wavelength: name for name, wavelength in pairs}
            if 488 in by_wavelength and 568 in by_wavelength:
                return {2: by_wavelength[488], 3: by_wavelength[568]}
    return {channel: f"ch{channel}" for channel in range(2, channel_count + 1)}


def _normalize_to_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    names = list(axes.upper())
    if len(names) != array.ndim:
        raise ValueError(f"TIFF axes {axes!r} do not match shape {array.shape}")
    for index in reversed(range(array.ndim)):
        if array.shape[index] == 1 and names[index] not in {"Y", "X"}:
            array = np.squeeze(array, axis=index)
            names.pop(index)
    if "C" not in names:
        candidates = [
            index for index, (name, size) in enumerate(zip(names, array.shape))
            if name not in {"Y", "X"} and size in (2, 3)
        ]
        if len(candidates) != 1:
            raise ValueError(f"Cannot identify channel axis from axes={axes}, shape={array.shape}")
        names[candidates[0]] = "C"
    if "Y" not in names or "X" not in names:
        raise ValueError(f"Cannot identify Y/X axes from {axes}")
    unknown = [name for name in names if name not in {"Z", "C", "Y", "X"}]
    if unknown:
        raise ValueError(f"Unsupported non-singleton TIFF axes: {unknown}")
    if "Z" not in names:
        array = np.expand_dims(array, axis=0)
        names.insert(0, "Z")
    return np.transpose(array, [names.index(name) for name in "ZCYX"])


def _date_from_path(path: Path) -> str:
    for part in path.parts:
        match = re.search(r"(?<!\d)(\d{8})(?!\d)", part)
        if match:
            return match.group(1)
    return "unknown"


def _condition_from_folder(folder: Path) -> str:
    for part in reversed(folder.parts):
        text = part.casefold()
        if "normoxia" in text:
            return "normoxia"
        if "hypoxia" in text:
            return "hypoxia"
    return "unspecified"


def _replicate_group_key(folder: Path) -> tuple[str, ...]:
    """Group replicate dates by biological target combination, not filename labels."""
    names = {name.casefold() for name, _ in _target_pairs(folder.parent.parent.name)}
    if not names:
        names = {row["target"].casefold() for row in parameter_rows_for_folder(folder)}
    if not names:
        names = {"unspecified"}
    return tuple(sorted(names))


def build_replicate_map(folders: list[Path]) -> dict[tuple[tuple[str, ...], str], str]:
    dates_by_group: dict[tuple[str, ...], set[str]] = {}
    for folder in folders:
        dates_by_group.setdefault(_replicate_group_key(folder), set()).add(_date_from_path(folder))
    return {
        (group, date): f"Rep{index}"
        for group, dates in dates_by_group.items()
        for index, date in enumerate(sorted(dates), start=1)
    }


def _tiff_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.casefold() in TIFF_SUFFIXES
    )


def _channel_count(path: Path) -> int:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        if "C" in axes:
            return int(series.shape[axes.index("C")])
        candidates = [
            size for name, size in zip(axes, series.shape)
            if name not in {"Y", "X"} and size in (2, 3)
        ]
        if len(candidates) == 1:
            return int(candidates[0])
    raise ValueError(f"Could not identify channel count in {path}")


def _recorded_parameters(folder: Path, targets: dict[int, str]) -> list[dict]:
    source = folder / PARAMETER_TABLE_RELATIVE_PATH
    if not source.exists():
        return []
    data = pd.read_csv(source)
    required = {"channel", "threshold", "min_object_size_voxels", "max_object_size_voxels"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {source}: {sorted(missing)}")
    rows: list[dict] = []
    for channel_text, group in data.groupby("channel"):
        channel = int(str(channel_text).upper().removeprefix("C"))
        unique = group[
            ["threshold", "min_object_size_voxels", "max_object_size_voxels"]
        ].drop_duplicates()
        if len(unique) != 1:
            raise ValueError(f"Multiple parameter sets for {channel_text} in {source}")
        values = unique.iloc[0]
        if "target" in group.columns:
            target_values = group["target"].dropna().astype(str).str.strip().unique()
            if len(target_values) > 1:
                raise ValueError(f"Multiple target names for {channel_text} in {source}")
            target = target_values[0] if len(target_values) else targets.get(channel, f"ch{channel}")
        else:
            target = targets.get(channel, f"ch{channel}")
        rows.append(
            {
                "target": target,
                "channel": channel,
                "threshold": float(values["threshold"]),
                "min_object_size_voxels": int(values["min_object_size_voxels"]),
                "max_object_size_voxels": int(values["max_object_size_voxels"]),
                "parameter_source": str(source),
            }
        )
    return rows


def parameter_rows_for_folder(folder: Path) -> list[dict]:
    images = _tiff_files(folder)
    if not images:
        raise FileNotFoundError(f"No TIFF images found directly in {folder}")
    targets = infer_target_names(folder, _channel_count(images[0]))
    rows = _recorded_parameters(folder, targets)
    if not rows:
        raise FileNotFoundError(
            "Missing parameter CSV: "
            f"{folder / PARAMETER_TABLE_RELATIVE_PATH}"
        )
    return rows


def find_analysis_folders(batch_root: Path, folder_name: str | None = None) -> list[Path]:
    """Find leaf-most TIFF folders containing the required parameter CSV."""
    root = Path(batch_root).resolve()
    candidates = [root, *(path for path in root.rglob("*") if path.is_dir())]
    folders: list[Path] = []
    for folder in candidates:
        if any(
            part.casefold() in OUTPUT_DIRECTORY_NAMES
            for part in folder.relative_to(root).parts
        ):
            continue
        if folder_name and folder.name.casefold() != folder_name.casefold():
            continue
        recorded = folder / PARAMETER_TABLE_RELATIVE_PATH
        if recorded.exists() and _tiff_files(folder):
            folders.append(folder.resolve())
    if folder_name:
        return sorted(set(folders))
    return sorted(
        folder for folder in set(folders)
        if not any(folder != other and folder in other.parents for other in folders)
    )


def _filter_folders(
    folders: list[Path], include_date: str | None, exclude_date: str | None
) -> list[Path]:
    selected: list[Path] = []
    for folder in folders:
        date = _date_from_path(folder)
        if include_date and date != include_date:
            continue
        if exclude_date and date == exclude_date:
            continue
        selected.append(folder)
    return selected


def validate_filter_consistency(folders: list[Path]) -> None:
    values: dict[str, set[tuple[int, int]]] = {}
    for folder in folders:
        for row in parameter_rows_for_folder(folder):
            values.setdefault(row["target"].casefold(), set()).add(
                (row["min_object_size_voxels"], row["max_object_size_voxels"])
            )
    conflicts = {target: filters for target, filters in values.items() if len(filters) > 1}
    if conflicts:
        raise ValueError(f"Inconsistent target size filters: {conflicts}")


def _label_mask_for_image(folder: Path, image_path: Path) -> Path:
    path = folder / "Intensity" / "Masks" / f"{image_path.stem}_nuclei_label_mask.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing nucleus label mask: {path}")
    return path


def _threshold_stack(channel_stack: np.ndarray, threshold: float) -> np.ndarray:
    binary = np.empty(channel_stack.shape, dtype=bool)
    for z_index in range(channel_stack.shape[0]):
        smoothed = gaussian(
            channel_stack[z_index], sigma=GAUSSIAN_SIGMA, preserve_range=True
        )
        binary[z_index] = smoothed > threshold
    return binary_fill_holes(binary) if FILL_TARGET_HOLES else binary


def filter_objects_by_size(binary: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
    """Apply the inclusive voxel-size filter to objects in the whole 3D image."""
    components = label(binary, connectivity=1)
    sizes = np.bincount(components.ravel())
    keep = (sizes >= min_size) & (sizes <= max_size)
    keep[0] = False
    return keep[components]


def measure_foci_in_nuclei(
    binary: np.ndarray,
    nucleus_labels: np.ndarray,
    voxel_volume_um3: float,
) -> tuple[list[dict], list[dict], np.ndarray]:
    """Assign each whole-image 3D focus to the nucleus with greatest voxel overlap."""
    if binary.ndim != 3 or nucleus_labels.ndim != 2:
        raise ValueError("Expected a 3D threshold mask and a 2D nucleus label mask")
    if binary.shape[1:] != nucleus_labels.shape:
        raise ValueError(
            f"ROI shape {nucleus_labels.shape} does not match channel YX {binary.shape[1:]}"
        )
    focus_label_mask = label(binary, connectivity=1).astype(np.uint32)
    focus_lists: dict[int, list[dict]] = {
        int(nucleus_id): [] for nucleus_id in np.unique(nucleus_labels) if nucleus_id != 0
    }
    focus_rows: list[dict] = []
    for region in regionprops(focus_label_mask):
        global_focus_id = int(region.label)
        z_indices, y_indices, x_indices = region.coords.T
        overlapping_nuclei = nucleus_labels[y_indices, x_indices]
        overlaps = np.bincount(overlapping_nuclei.astype(np.int64))
        if len(overlaps) <= 1 or overlaps[1:].max(initial=0) == 0:
            continue
        nucleus_id = int(np.argmax(overlaps[1:]) + 1)
        volume_voxels = int(region.area)
        focus_lists[nucleus_id].append(
            {
                "global_focus_id": global_focus_id,
                "volume_voxels": volume_voxels,
                "volume_um3": round(volume_voxels * voxel_volume_um3, 6),
            }
        )

    nucleus_rows: list[dict] = []
    for nucleus_id in sorted(focus_lists):
        assigned = focus_lists[nucleus_id]
        for focus_id, focus in enumerate(assigned, start=1):
            focus_rows.append(
                {
                    "nucleus_id": nucleus_id,
                    "focus_id_within_nucleus": focus_id,
                    "volume_voxels": focus["volume_voxels"],
                    "volume_um3": focus["volume_um3"],
                }
            )
        volumes = [focus["volume_voxels"] for focus in assigned]
        total_voxels = int(sum(volumes))
        count = len(volumes)
        nucleus_rows.append(
            {
                "nucleus_id": int(nucleus_id),
                "focus_count": count,
                "total_focus_volume_voxels": total_voxels,
                "total_focus_volume_um3": round(total_voxels * voxel_volume_um3, 6),
                "mean_focus_volume_voxels": float(np.mean(volumes)) if volumes else 0.0,
                "mean_focus_volume_um3": round(float(np.mean(volumes)) * voxel_volume_um3, 6)
                if volumes else 0.0,
            }
        )
    return nucleus_rows, focus_rows, focus_label_mask


def _metadata(
    folder: Path,
    target: str,
    channel: int,
    threshold: float,
    min_size: int,
    max_size: int,
    xy_pixel_size_um: float,
    z_step_um: float,
    replicate: str,
) -> dict:
    voxel_volume = xy_pixel_size_um * xy_pixel_size_um * z_step_um
    return {
        "acquisition_date": _date_from_path(folder),
        "condition": _condition_from_folder(folder),
        "replicate": replicate,
        "target_group": target,
        "source_folder": str(folder),
        "target": target,
        "channel": f"C{channel}",
        "threshold": threshold,
        "min_object_size_voxels": min_size,
        "max_object_size_voxels": max_size,
        "xy_pixel_size_um": xy_pixel_size_um,
        "z_step_um": z_step_um,
        "voxel_volume_um3": voxel_volume,
    }


def process_folder(
    folder: Path, xy_pixel_size_um: float, z_step_um: float, replicate: str
) -> dict[str, int]:
    parameters = parameter_rows_for_folder(folder)
    if not parameters:
        return {"images": 0, "nucleus_rows": 0, "foci": 0, "parameters": 0, "errors": 0}
    output = folder / RESULT_FOLDER_NAME
    masks = output / "Masks"
    output.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)
    images = _tiff_files(folder)
    nucleus_output: list[dict] = []
    focus_output: list[dict] = []
    parameter_output: list[dict] = []
    errors: list[dict] = []

    for parameter in parameters:
        meta = _metadata(
            folder, parameter["target"], parameter["channel"], parameter["threshold"],
            parameter["min_object_size_voxels"], parameter["max_object_size_voxels"],
            xy_pixel_size_um, z_step_um,
            replicate,
        )
        parameter_output.append(
            {**meta, "gaussian_sigma_px": GAUSSIAN_SIGMA, "parameter_source": parameter["parameter_source"]}
        )

    for image_index, image_path in enumerate(images, start=1):
        print(f"  [{image_index}/{len(images)}] {image_path.name}")
        raw_memmap: np.memmap | None = None
        try:
            nucleus_labels = tifffile.imread(_label_mask_for_image(folder, image_path))
            with tifffile.TiffFile(image_path) as tif:
                series = tif.series[0]
                source_axes = series.axes
                array = series.asarray(out="memmap")
                if isinstance(array, np.memmap):
                    raw_memmap = array
            zcyx = _normalize_to_zcyx(array, source_axes)
            for parameter in parameters:
                channel = parameter["channel"]
                if zcyx.shape[1] < channel:
                    raise ValueError(f"{image_path.name} has only {zcyx.shape[1]} channels; C{channel} requested")
                meta = _metadata(
                    folder, parameter["target"], channel, parameter["threshold"],
                    parameter["min_object_size_voxels"], parameter["max_object_size_voxels"],
                    xy_pixel_size_um, z_step_um,
                    replicate,
                )
                thresholded = _threshold_stack(zcyx[:, channel - 1], parameter["threshold"])
                thresholded = filter_objects_by_size(
                    thresholded,
                    parameter["min_object_size_voxels"],
                    parameter["max_object_size_voxels"],
                )
                nuclei, foci, focus_labels = measure_foci_in_nuclei(
                    thresholded, nucleus_labels, meta["voxel_volume_um3"],
                )
                image_meta = {**meta, "image_name": image_path.name}
                nucleus_output.extend({**image_meta, **row} for row in nuclei)
                focus_output.extend({**image_meta, **row} for row in foci)
                stem = f"{image_path.stem}_{parameter['target']}_C{channel}_whole_image_foci"
                tifffile.imwrite(
                    masks / f"{stem}_binary_mask.tif",
                    (focus_labels > 0).astype(np.uint8) * 255,
                    imagej=True,
                    metadata={"axes": "ZYX"},
                )
                tifffile.imwrite(
                    masks / f"{stem}_label_mask.tif",
                    focus_labels,
                    metadata={"axes": "ZYX"},
                    photometric="minisblack",
                )
        except Exception as exc:
            errors.append(
                {
                    "source_folder": str(folder), "image_name": image_path.name,
                    "target": "", "channel": "", "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if raw_memmap is not None:
                temporary_path = Path(raw_memmap.filename)
                raw_memmap._mmap.close()
                if temporary_path.resolve() != image_path.resolve():
                    temporary_path.unlink(missing_ok=True)

    pd.DataFrame(nucleus_output, columns=NUCLEUS_COLUMNS).to_csv(output / "nucleus_focus_counts.csv", index=False)
    pd.DataFrame(focus_output, columns=FOCUS_COLUMNS).to_csv(output / "focus_volumes.csv", index=False)
    pd.DataFrame(parameter_output, columns=PARAMETER_COLUMNS).to_csv(output / "parameters_used.csv", index=False)
    pd.DataFrame(errors, columns=ERROR_COLUMNS).to_csv(output / "errors.csv", index=False)
    return {
        "images": len(images), "nucleus_rows": len(nucleus_output), "foci": len(focus_output),
        "parameters": len(parameter_output), "errors": len(errors),
    }


def collect_merged_csvs(batch_root: Path) -> dict[str, Path]:
    root = Path(batch_root).resolve()
    specs = {
        "nuclei": ("nucleus_focus_counts.csv", NUCLEUS_COLUMNS, root / "Merged_condensate_nucleus_focus_counts.csv"),
        "foci": ("focus_volumes.csv", FOCUS_COLUMNS, root / "Merged_condensate_focus_volumes.csv"),
        "parameters": ("parameters_used.csv", PARAMETER_COLUMNS, root / "Merged_condensate_parameters_used.csv"),
        "errors": ("errors.csv", ERROR_COLUMNS, root / "Merged_condensate_errors.csv"),
    }
    outputs: dict[str, Path] = {}
    for key, (filename, columns, destination) in specs.items():
        sources = sorted(root.rglob(f"{RESULT_FOLDER_NAME}/{filename}"))
        frames = [pd.read_csv(source) for source in sources if source.stat().st_size > 0]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        combined.reindex(columns=columns).to_csv(destination, index=False)
        outputs[key] = destination
    return outputs


def run_batch(
    batch_root: Path, xy_pixel_size_um: float, z_step_um: float,
    include_date: str | None, exclude_date: str | None,
    folder_name: str | None = None,
    require_consistent_filters: bool = True,
) -> int:
    all_folders = find_analysis_folders(batch_root, folder_name)
    if require_consistent_filters:
        validate_filter_consistency(all_folders)
    replicate_map = build_replicate_map(all_folders)
    folders = _filter_folders(all_folders, include_date, exclude_date)
    if not folders:
        raise FileNotFoundError("No matching analysis folders were found")
    summary: list[dict] = []
    for index, folder in enumerate(folders, start=1):
        print(f"DATASET [{index}/{len(folders)}] z_step={z_step_um} um: {folder}")
        replicate = replicate_map[(_replicate_group_key(folder), _date_from_path(folder))]
        summary.append(
            {
                "folder": str(folder), "z_step_um": z_step_um, "replicate": replicate,
                **process_folder(folder, xy_pixel_size_um, z_step_um, replicate),
            }
        )
    label = include_date or (f"excluding_{exclude_date}" if exclude_date else "all_selected")
    summary_path = Path(batch_root).resolve() / f"condensate_thresholding_batch_{label}.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    merged = collect_merged_csvs(batch_root)
    print(f"Batch summary: {summary_path}")
    for key, path in merged.items():
        print(f"Merged {key}: {path}")
    return 1 if any(row["errors"] for row in summary) else 0


def choose_folder_gui() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        title="Choose a TIFF folder or a root containing TIFF folders"
    )
    root.destroy()
    return Path(selected) if selected else None


def choose_positive_float_gui(title: str, prompt: str, initial: float = 1.0) -> float | None:
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    value = simpledialog.askfloat(title, prompt, initialvalue=initial, minvalue=1e-12, parent=root)
    root.destroy()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root", type=Path,
        help="Selected TIFF folder or a root containing TIFF folders. Opens a folder picker if omitted.",
    )
    parser.add_argument("--folder-name", help="Optional exact folder-name restriction in batch mode")
    parser.add_argument(
        "--parameter-table", type=Path,
        default=PARAMETER_TABLE_RELATIVE_PATH,
        help="Parameter CSV path relative to each selected TIFF folder.",
    )
    parser.add_argument("--xy-pixel-size", type=float, help="XY pixel size in micrometers")
    parser.add_argument("--z-step", type=float, help="Z spacing in micrometers")
    parser.add_argument("--include-date", help="Process only this YYYYMMDD date")
    parser.add_argument("--exclude-date", help="Skip this YYYYMMDD date")
    parser.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA)
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument(
        "--allow-varying-filters", action="store_true",
        help="Allow the same target to use different size filters across selected folders.",
    )
    return parser.parse_args()


def main() -> int:
    global GAUSSIAN_SIGMA, FILL_TARGET_HOLES, PARAMETER_TABLE_RELATIVE_PATH
    args = parse_args()
    if args.include_date and args.exclude_date:
        raise SystemExit("Use either --include-date or --exclude-date, not both")
    if args.gaussian_sigma < 0:
        raise SystemExit("--gaussian-sigma must be >= 0")
    GAUSSIAN_SIGMA = args.gaussian_sigma
    FILL_TARGET_HOLES = not args.no_fill_holes
    if args.parameter_table.is_absolute():
        raise SystemExit("--parameter-table must be relative to each selected TIFF folder")
    PARAMETER_TABLE_RELATIVE_PATH = args.parameter_table
    using_gui = args.batch_root is None
    batch_root = args.batch_root or choose_folder_gui()
    if batch_root is None:
        return 0
    xy_pixel_size = args.xy_pixel_size
    z_step = args.z_step
    if using_gui:
        xy_pixel_size = xy_pixel_size or choose_positive_float_gui(
            "XY calibration", "XY pixel size in micrometers:"
        )
        z_step = z_step or choose_positive_float_gui(
            "Z calibration", "Z step in micrometers:"
        )
        if xy_pixel_size is None or z_step is None:
            return 0
    if xy_pixel_size is None or z_step is None:
        raise SystemExit("Provide --xy-pixel-size and --z-step in command-line mode")
    if z_step <= 0 or xy_pixel_size <= 0:
        raise SystemExit("--z-step and --xy-pixel-size must be positive")
    return run_batch(
        batch_root, xy_pixel_size, z_step, args.include_date,
        args.exclude_date, args.folder_name, not args.allow_varying_filters,
    )


if __name__ == "__main__":
    raise SystemExit(main())
