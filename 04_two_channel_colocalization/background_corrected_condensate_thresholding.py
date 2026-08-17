"""Background-corrected 3D condensate analysis for user-selected targets.

For each target/replicate group, this pipeline pools nuclear residual
intensities across the selected folders and applies one shared threshold. It
uses Gaussian smoothing, local-background subtraction, no hole filling, and no
watershed. Existing 2D nucleus masks are used as 3D ROIs.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.measure import label, regionprops


BASE_PATH = Path(__file__).resolve().with_name("# 3D TIFF channel thresholding.py")
spec = importlib.util.spec_from_file_location("recorded_threshold_pipeline", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

TARGETS: set[str] = set()
BACKGROUND_WINDOW_PX: dict[str, int] = {}
DEFAULT_BACKGROUND_WINDOW_PX = 31
SHARED_THRESHOLD_K_MAD = 5.0
THRESHOLD_SAMPLE_STRIDE = 50
GAUSSIAN_SIGMA = 1.0

NUCLEUS_COLUMNS = base.NUCLEUS_COLUMNS + [
    "segmentation_method", "background_window_px", "watershed_enabled",
]
FOCUS_COLUMNS = base.FOCUS_COLUMNS + [
    "segmentation_method", "background_window_px", "watershed_enabled",
]
PARAMETER_COLUMNS = base.PARAMETER_COLUMNS[:-1] + [
    "segmentation_method", "background_window_px", "shared_threshold_k_mad",
    "threshold_scope", "hole_filling_enabled", "watershed_enabled", "parameter_source",
]


def parameter_rows_for_folder(folder: Path) -> list[dict]:
    rows = base.parameter_rows_for_folder(folder)
    return rows if not TARGETS else [
        row for row in rows if row["target"].casefold() in TARGETS
    ]


def selected_folders(batch_root: Path, folder_name: str | None = None) -> list[Path]:
    return [
        folder for folder in base.find_analysis_folders(batch_root, folder_name)
        if parameter_rows_for_folder(folder)
    ]


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


def background_residual(
    channel_stack: np.ndarray, target: str, window_px: int | None = None,
) -> tuple[np.ndarray, int]:
    window = window_px or BACKGROUND_WINDOW_PX.get(
        target.casefold(), DEFAULT_BACKGROUND_WINDOW_PX
    )
    if window < 3 or window % 2 == 0:
        raise ValueError("Background window must be an odd integer >= 3")
    smooth = gaussian_filter(
        channel_stack.astype(np.float32, copy=False),
        sigma=(0, GAUSSIAN_SIGMA, GAUSSIAN_SIGMA),
        mode="nearest",
    )
    background = uniform_filter(smooth, size=(1, window, window), mode="nearest")
    return smooth - background, window


def filter_objects(binary: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    components = label(binary, connectivity=1)
    sizes = np.bincount(components.ravel())
    keep = (sizes >= minimum) & (sizes <= maximum)
    keep[0] = False
    return keep[components]


def segment_nuclei(
    residual: np.ndarray,
    nucleus_labels: np.ndarray,
    threshold: float,
    minimum: int,
    maximum: int,
    voxel_volume_um3: float,
) -> tuple[list[dict], list[dict], np.ndarray]:
    if residual.shape[1:] != nucleus_labels.shape:
        raise ValueError(f"ROI shape {nucleus_labels.shape} != channel YX {residual.shape[1:]}")
    whole_labels = np.zeros(residual.shape, dtype=np.uint32)
    nucleus_rows: list[dict] = []
    focus_rows: list[dict] = []
    global_focus_id = 0
    for nucleus in regionprops(nucleus_labels.astype(np.int32)):
        minr, minc, maxr, maxc = nucleus.bbox
        roi = nucleus_labels[minr:maxr, minc:maxc] == nucleus.label
        crop = residual[:, minr:maxr, minc:maxc]
        local = label(
            filter_objects((crop > threshold) & roi[None, :, :], minimum, maximum),
            connectivity=1,
        ).astype(np.uint32)
        local_regions = regionprops(local)
        remap = np.zeros(int(local.max()) + 1, dtype=np.uint32)
        volumes: list[int] = []
        for focus_id, focus in enumerate(local_regions, start=1):
            volume = int(focus.area)
            volumes.append(volume)
            global_focus_id += 1
            remap[int(focus.label)] = global_focus_id
            focus_rows.append(
                {
                    "nucleus_id": int(nucleus.label),
                    "focus_id_within_nucleus": focus_id,
                    "volume_voxels": volume,
                    "volume_um3": round(volume * voxel_volume_um3, 6),
                }
            )
        local = remap[local]
        whole_labels[:, minr:maxr, minc:maxc] = np.maximum(
            whole_labels[:, minr:maxr, minc:maxc], local
        )
        total = int(sum(volumes))
        nucleus_rows.append(
            {
                "nucleus_id": int(nucleus.label),
                "focus_count": len(volumes),
                "total_focus_volume_voxels": total,
                "total_focus_volume_um3": round(total * voxel_volume_um3, 6),
                "mean_focus_volume_voxels": float(np.mean(volumes)) if volumes else 0.0,
                "mean_focus_volume_um3": round(float(np.mean(volumes)) * voxel_volume_um3, 6)
                if volumes else 0.0,
            }
        )
    return nucleus_rows, focus_rows, whole_labels


def shared_thresholds(
    folders: list[Path], replicate_map: dict[tuple[tuple[str, ...], str], str]
) -> dict[tuple[str, str], dict]:
    samples: dict[tuple[str, str], list[np.ndarray]] = {}
    for folder in folders:
        replicate = replicate_map[(base._replicate_group_key(folder), base._date_from_path(folder))]
        for parameter in parameter_rows_for_folder(folder):
            key = (parameter["target"].casefold(), replicate)
            for image_path in base._tiff_files(folder):
                nuclei = tifffile.imread(base._label_mask_for_image(folder, image_path))
                with tifffile.TiffFile(image_path) as tif:
                    series = tif.series[0]
                    zcyx = base._normalize_to_zcyx(series.asarray(), series.axes)
                residual, _ = background_residual(
                    zcyx[:, int(parameter["channel"]) - 1], parameter["target"]
                )
                values = residual[:, nuclei > 0].ravel()[::THRESHOLD_SAMPLE_STRIDE]
                if values.size:
                    samples.setdefault(key, []).append(values)
    output: dict[tuple[str, str], dict] = {}
    for key, chunks in samples.items():
        pooled = np.concatenate(chunks)
        center = float(np.median(pooled))
        sigma = float(1.4826 * np.median(np.abs(pooled - center)))
        threshold = center + SHARED_THRESHOLD_K_MAD * sigma
        output[key] = {
            "threshold": threshold,
            "parameter_source": (
                "pooled selected-folder nuclear residuals: "
                f"median ({center:.6g}) + {SHARED_THRESHOLD_K_MAD:g}*MAD-sigma ({sigma:.6g})"
            ),
        }
        print(f"SHARED THRESHOLD target={key[0]} replicate={key[1]} threshold={threshold:.6g}")
    return output


def metadata(
    folder: Path, parameter: dict, xy_pixel_size: float,
    z_step: float, replicate: str,
) -> dict:
    meta = base._metadata(
        folder, parameter["target"], parameter["channel"], parameter["threshold"],
        parameter["min_object_size_voxels"], parameter["max_object_size_voxels"],
        xy_pixel_size, z_step, replicate,
    )
    meta.update(
        segmentation_method="local_background_residual",
        background_window_px=BACKGROUND_WINDOW_PX.get(
            parameter["target"].casefold(), DEFAULT_BACKGROUND_WINDOW_PX
        ),
        watershed_enabled=False,
    )
    return meta


def process_folder(
    folder: Path, xy_pixel_size: float, z_step: float, replicate: str,
    thresholds: dict[tuple[str, str], dict]
) -> dict[str, int]:
    parameters = parameter_rows_for_folder(folder)
    for parameter in parameters:
        values = thresholds[(parameter["target"].casefold(), replicate)]
        parameter["threshold"] = values["threshold"]
        parameter["parameter_source"] = values["parameter_source"]
    output = folder / base.RESULT_FOLDER_NAME
    masks = output / "Masks"
    output.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)
    nucleus_output, focus_output, parameter_output, errors = [], [], [], []
    for parameter in parameters:
        meta = metadata(folder, parameter, xy_pixel_size, z_step, replicate)
        parameter_output.append(
            {
                **meta,
                "gaussian_sigma_px": GAUSSIAN_SIGMA,
                "shared_threshold_k_mad": SHARED_THRESHOLD_K_MAD,
                "threshold_scope": "target_replicate_pooled_selected_folders",
                "hole_filling_enabled": False,
                "parameter_source": parameter["parameter_source"],
            }
        )
    images = base._tiff_files(folder)
    for index, image_path in enumerate(images, start=1):
        print(f"  [{index}/{len(images)}] {image_path.name}")
        try:
            nuclei = tifffile.imread(base._label_mask_for_image(folder, image_path))
            with tifffile.TiffFile(image_path) as tif:
                series = tif.series[0]
                zcyx = base._normalize_to_zcyx(series.asarray(), series.axes)
            for parameter in parameters:
                channel = int(parameter["channel"])
                meta = metadata(folder, parameter, xy_pixel_size, z_step, replicate)
                residual, _ = background_residual(zcyx[:, channel - 1], parameter["target"])
                nucleus_rows, focus_rows, labels = segment_nuclei(
                    residual, nuclei, parameter["threshold"],
                    parameter["min_object_size_voxels"], parameter["max_object_size_voxels"],
                    meta["voxel_volume_um3"],
                )
                image_meta = {**meta, "image_name": image_path.name}
                nucleus_output.extend({**image_meta, **row} for row in nucleus_rows)
                focus_output.extend({**image_meta, **row} for row in focus_rows)
                stem = f"{image_path.stem}_{parameter['target']}_C{channel}_whole_image_foci"
                tifffile.imwrite(
                    masks / f"{stem}_binary_mask.tif", (labels > 0).astype(np.uint8) * 255,
                    imagej=True, metadata={"axes": "ZYX"},
                )
                tifffile.imwrite(
                    masks / f"{stem}_label_mask.tif", labels,
                    metadata={"axes": "ZYX"}, photometric="minisblack",
                )
        except Exception as exc:
            errors.append(
                {"source_folder": str(folder), "image_name": image_path.name,
                 "target": "", "channel": "", "error": f"{type(exc).__name__}: {exc}"}
            )
    pd.DataFrame(nucleus_output, columns=NUCLEUS_COLUMNS).to_csv(output / "nucleus_focus_counts.csv", index=False)
    pd.DataFrame(focus_output, columns=FOCUS_COLUMNS).to_csv(output / "focus_volumes.csv", index=False)
    pd.DataFrame(parameter_output, columns=PARAMETER_COLUMNS).to_csv(output / "parameters_used.csv", index=False)
    pd.DataFrame(errors, columns=base.ERROR_COLUMNS).to_csv(output / "errors.csv", index=False)
    return {"images": len(images), "nucleus_rows": len(nucleus_output), "foci": len(focus_output), "errors": len(errors)}


def collect_merged(batch_root: Path) -> None:
    root = Path(batch_root).resolve()
    specs = [
        ("nucleus_focus_counts.csv", NUCLEUS_COLUMNS, "Merged_condensate_nucleus_focus_counts.csv"),
        ("focus_volumes.csv", FOCUS_COLUMNS, "Merged_condensate_focus_volumes.csv"),
        ("parameters_used.csv", PARAMETER_COLUMNS, "Merged_condensate_parameters_used.csv"),
        ("errors.csv", base.ERROR_COLUMNS, "Merged_condensate_errors.csv"),
    ]
    for filename, columns, destination in specs:
        frames = [pd.read_csv(path) for path in sorted(root.rglob(f"{base.RESULT_FOLDER_NAME}/{filename}"))]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        combined.reindex(columns=columns).to_csv(root / destination, index=False)


def run(
    batch_root: Path, xy_pixel_size: float, z_step: float,
    include_date: str | None, exclude_date: str | None,
    folder_name: str | None = None,
    require_consistent_filters: bool = True,
) -> int:
    all_folders = selected_folders(batch_root, folder_name)
    if require_consistent_filters:
        validate_filter_consistency(all_folders)
    replicate_map = base.build_replicate_map(base.find_analysis_folders(batch_root, folder_name))
    folders = base._filter_folders(all_folders, include_date, exclude_date)
    if not folders:
        raise FileNotFoundError("No matching target folders found")
    thresholds = shared_thresholds(folders, replicate_map)
    summaries = []
    for index, folder in enumerate(folders, start=1):
        replicate = replicate_map[(base._replicate_group_key(folder), base._date_from_path(folder))]
        print(f"DATASET [{index}/{len(folders)}] z_step={z_step} um: {folder}")
        summaries.append(
            {"folder": str(folder), "z_step_um": z_step, "replicate": replicate,
             **process_folder(folder, xy_pixel_size, z_step, replicate, thresholds)}
        )
    label_name = include_date or (f"excluding_{exclude_date}" if exclude_date else "all_selected")
    pd.DataFrame(summaries).to_csv(
        Path(batch_root).resolve() / f"condensate_thresholding_batch_{label_name}.csv", index=False
    )
    collect_merged(batch_root)
    return 1 if any(row["errors"] for row in summaries) else 0


def main() -> int:
    global TARGETS, DEFAULT_BACKGROUND_WINDOW_PX, SHARED_THRESHOLD_K_MAD
    global THRESHOLD_SAMPLE_STRIDE, GAUSSIAN_SIGMA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root", type=Path,
        help="Selected TIFF folder or root containing TIFF folders. Opens a picker if omitted.",
    )
    parser.add_argument("--folder-name", help="Optional exact folder-name restriction")
    parser.add_argument(
        "--parameter-table", type=Path,
        default=base.PARAMETER_TABLE_RELATIVE_PATH,
        help="Parameter CSV path relative to each selected TIFF folder.",
    )
    parser.add_argument("--target", action="append", default=[], help="Target name to include; repeatable")
    parser.add_argument("--xy-pixel-size", type=float, help="XY pixel size in micrometers")
    parser.add_argument("--z-step", type=float, help="Z spacing in micrometers")
    parser.add_argument("--background-window", type=int, default=DEFAULT_BACKGROUND_WINDOW_PX)
    parser.add_argument(
        "--target-background-window", action="append", default=[], metavar="TARGET=PIXELS",
        help="Per-target odd local-background window; repeatable",
    )
    parser.add_argument("--threshold-k-mad", type=float, default=SHARED_THRESHOLD_K_MAD)
    parser.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA)
    parser.add_argument("--threshold-sample-stride", type=int, default=THRESHOLD_SAMPLE_STRIDE)
    parser.add_argument(
        "--allow-varying-filters", action="store_true",
        help="Allow the same target to use different size filters across selected folders.",
    )
    parser.add_argument("--include-date")
    parser.add_argument("--exclude-date")
    args = parser.parse_args()
    if args.include_date and args.exclude_date:
        raise SystemExit("Use either --include-date or --exclude-date, not both")
    if args.background_window < 3 or args.background_window % 2 == 0:
        raise SystemExit("--background-window must be an odd integer >= 3")
    if args.threshold_sample_stride < 1 or args.gaussian_sigma < 0:
        raise SystemExit("Sample stride must be >= 1 and Gaussian sigma must be >= 0")
    TARGETS = {target.casefold() for target in args.target}
    if args.parameter_table.is_absolute():
        raise SystemExit("--parameter-table must be relative to each selected TIFF folder")
    base.PARAMETER_TABLE_RELATIVE_PATH = args.parameter_table
    DEFAULT_BACKGROUND_WINDOW_PX = args.background_window
    SHARED_THRESHOLD_K_MAD = args.threshold_k_mad
    THRESHOLD_SAMPLE_STRIDE = args.threshold_sample_stride
    GAUSSIAN_SIGMA = args.gaussian_sigma
    for item in args.target_background_window:
        try:
            target, window_text = item.split("=", 1)
            window = int(window_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid --target-background-window {item!r}; use TARGET=PIXELS") from exc
        if window < 3 or window % 2 == 0:
            raise SystemExit(f"Background window for {target!r} must be odd and >= 3")
        BACKGROUND_WINDOW_PX[target.casefold()] = window
    using_gui = args.batch_root is None
    batch_root = args.batch_root or base.choose_folder_gui()
    if batch_root is None:
        return 0
    xy_pixel_size = args.xy_pixel_size
    z_step = args.z_step
    if using_gui:
        xy_pixel_size = xy_pixel_size or base.choose_positive_float_gui(
            "XY calibration", "XY pixel size in micrometers:"
        )
        z_step = z_step or base.choose_positive_float_gui(
            "Z calibration", "Z step in micrometers:"
        )
        if xy_pixel_size is None or z_step is None:
            return 0
    if xy_pixel_size is None or z_step is None:
        raise SystemExit("Provide --xy-pixel-size and --z-step in command-line mode")
    if z_step <= 0 or xy_pixel_size <= 0:
        raise SystemExit("--z-step and --xy-pixel-size must be positive")
    return run(
        batch_root, xy_pixel_size, z_step, args.include_date,
        args.exclude_date, args.folder_name, not args.allow_varying_filters,
    )


if __name__ == "__main__":
    raise SystemExit(main())
