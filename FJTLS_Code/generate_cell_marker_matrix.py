"""Generate a whole-cell by marker matrix for one mIF sample."""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openslide
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cell_matrix import (
    build_cell_mask_rois,
    build_items_by_patch,
    cell_mask_roi_pixel_counts,
    measure_cell_raw_sum_rois,
    measure_cell_mask_rois,
    patch_bbox,
    write_cell_marker_h5ad,
)
from utils.marker import (
    CLOSE_ITERATIONS,
    DILATE_ITERATIONS,
    DILATE_KERNEL_SIZE,
    KERNEL_SIZE,
    MIN_AREA,
    Z_THRESHOLD,
    build_marker_arrays,
    compute_log1p_stats,
)
from utils.paths import (
    default_cell_path,
    default_coord_path,
    default_img_dir,
    discover_markers,
    load_coords,
    normalize_markers,
)

DEFAULT_SAMPLE_ID = "21-07010"
BBOX_MARGIN = 16
MATRIX_BATCH_SIZE = 16
MATRIX_CONTEXT = None
MATRIX_THREAD_STATE = threading.local()


def init_matrix_worker(context: dict) -> None:
    global MATRIX_CONTEXT
    MATRIX_CONTEXT = context


def get_matrix_slide(marker: str) -> openslide.OpenSlide:
    if MATRIX_CONTEXT is None:
        raise RuntimeError("matrix worker context is not initialized")

    slides = getattr(MATRIX_THREAD_STATE, "slides", None)
    if slides is None:
        slides = {}
        MATRIX_THREAD_STATE.slides = slides
    slide = slides.get(marker)
    if slide is None:
        slide = openslide.OpenSlide(str(MATRIX_CONTEXT["slide_paths"][marker]))
        slides[marker] = slide
    return slide


def process_matrix_patch(patch_i: int):
    if MATRIX_CONTEXT is None:
        raise RuntimeError("matrix worker context is not initialized")

    markers = MATRIX_CONTEXT["markers"]
    marker_stats = MATRIX_CONTEXT["marker_stats"]
    polys = MATRIX_CONTEXT["polys"]
    cells_by_patch = MATRIX_CONTEXT["cells_by_patch"]
    slide_w, slide_h = MATRIX_CONTEXT["slide_dims"]
    bbox_margin = MATRIX_CONTEXT["bbox_margin"]
    close_mask = MATRIX_CONTEXT["close_mask"]
    dilate_mask = MATRIX_CONTEXT["dilate_mask"]
    n_markers = len(markers)

    cell_indices = cells_by_patch.get(int(patch_i))
    if cell_indices is None or len(cell_indices) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, n_markers), dtype=np.float32),
            np.empty((0, n_markers), dtype=np.uint32),
            np.empty((0, n_markers), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
        )

    bbox = patch_bbox(polys, cell_indices, slide_w, slide_h, bbox_margin)
    patch_intensity = np.zeros((len(cell_indices), n_markers), dtype=np.float32)
    patch_count = np.zeros((len(cell_indices), n_markers), dtype=np.uint32)
    patch_raw_sum = np.zeros((len(cell_indices), n_markers), dtype=np.float32)
    if bbox is None:
        return (
            cell_indices,
            patch_intensity,
            patch_count,
            patch_raw_sum,
            np.zeros(len(cell_indices), dtype=np.uint32),
        )

    x0, y0, x1, y1 = bbox
    cell_mask_rois = build_cell_mask_rois(
        polys,
        cell_indices,
        (x0, y0),
        (y1 - y0, x1 - x0),
    )
    patch_cell_pixel_count = cell_mask_roi_pixel_counts(cell_mask_rois)
    for marker_i, marker in enumerate(markers):
        stats = marker_stats[marker]
        slide = get_matrix_slide(marker)
        region = slide.read_region((x0, y0), 0, (x1 - x0, y1 - y0))
        arr_channel = np.asarray(region)[..., 0]
        patch_raw_sum[:, marker_i] = measure_cell_raw_sum_rois(
            cell_mask_rois,
            arr_channel,
        )
        zscore, marker_mask = build_marker_arrays(
            arr_channel,
            float(stats["log_mean"]),
            float(stats["log_std"]),
            close_mask,
            dilate_mask,
        )
        values, counts = measure_cell_mask_rois(
            cell_mask_rois,
            zscore,
            marker_mask,
        )
        patch_intensity[:, marker_i] = values
        patch_count[:, marker_i] = counts

    return (
        cell_indices,
        patch_intensity,
        patch_count,
        patch_raw_sum,
        patch_cell_pixel_count,
    )


def process_matrix_patch_batch(patch_indices: np.ndarray):
    return [process_matrix_patch(int(patch_i)) for patch_i in patch_indices]


def compute_cell_marker_matrix(
    markers: list[str],
    slide_paths: dict[str, Path],
    marker_stats: dict[str, dict[str, float | int]],
    polys: np.ndarray,
    cells_by_patch: dict[int, np.ndarray],
    valid_patch_index: np.ndarray,
    slide_dims: tuple[int, int],
    num_workers: int,
    bbox_margin: int,
    close_mask: bool,
    dilate_mask: bool,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global MATRIX_CONTEXT

    n_cells = len(polys)
    n_markers = len(markers)
    intensity = np.zeros((n_cells, n_markers), dtype=np.float32)
    binary_count = np.zeros((n_cells, n_markers), dtype=np.uint32)
    raw_sum = np.zeros((n_cells, n_markers), dtype=np.float32)
    cell_pixel_count = np.zeros(n_cells, dtype=np.uint32)
    context = {
        "markers": markers,
        "slide_paths": slide_paths,
        "marker_stats": marker_stats,
        "polys": polys,
        "cells_by_patch": cells_by_patch,
        "slide_dims": slide_dims,
        "bbox_margin": bbox_margin,
        "close_mask": close_mask,
        "dilate_mask": dilate_mask,
    }
    batches = [
        valid_patch_index[start : start + MATRIX_BATCH_SIZE]
        for start in range(0, len(valid_patch_index), MATRIX_BATCH_SIZE)
    ]
    executor_kwargs = {
        "max_workers": num_workers,
        "initializer": init_matrix_worker,
        "initargs": (context,),
    }

    desc = f"cell matrix patches (workers={num_workers})"
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = [
            executor.submit(process_matrix_patch_batch, batch) for batch in batches
        ]
        progress = (
            tqdm(total=len(valid_patch_index), desc=desc, file=sys.stdout)
            if show_progress
            else None
        )
        try:
            for future in as_completed(futures):
                batch_results = future.result()
                if progress is not None:
                    progress.update(len(batch_results))
                for (
                    cell_indices,
                    patch_intensity,
                    patch_count,
                    patch_raw_sum,
                    patch_cell_pixel_count,
                ) in batch_results:
                    if len(cell_indices) == 0:
                        continue
                    intensity[cell_indices] = patch_intensity
                    binary_count[cell_indices] = patch_count
                    raw_sum[cell_indices] = patch_raw_sum
                    cell_pixel_count[cell_indices] = patch_cell_pixel_count
        finally:
            if progress is not None:
                progress.close()

    return intensity, binary_count, raw_sum, cell_pixel_count


def compute_marker_stats_task(task):
    marker, slide_path, coords, stats_workers, stats_chunk_size = task
    stats = compute_log1p_stats(
        coords,
        slide_path,
        num_workers=stats_workers,
        chunk_size=stats_chunk_size,
        show_progress=False,
    )
    return marker, stats


def compute_marker_stats(
    markers: list[str],
    slide_paths: dict[str, Path],
    coords: np.ndarray,
    marker_workers: int,
    stats_workers: int,
    stats_chunk_size: int,
    show_progress: bool,
) -> dict[str, dict[str, float | int]]:
    tasks = [
        (marker, slide_paths[marker], coords, stats_workers, stats_chunk_size)
        for marker in markers
    ]

    marker_stats: dict[str, dict[str, float | int]] = {}
    if marker_workers == 1:
        results = map(compute_marker_stats_task, tasks)
        if show_progress:
            results = tqdm(
                results, total=len(tasks), desc="marker stats", file=sys.stdout
            )
        for marker, stats in results:
            marker_stats[marker] = stats
        return marker_stats

    with ProcessPoolExecutor(max_workers=marker_workers) as executor:
        futures = [executor.submit(compute_marker_stats_task, task) for task in tasks]
        results = as_completed(futures)
        if show_progress:
            results = tqdm(
                results, total=len(futures), desc="marker stats", file=sys.stdout
            )
        for future in results:
            marker, stats = future.result()
            marker_stats[marker] = stats

    return marker_stats


def normalize_marker_stats(stats: dict) -> dict[str, float | int]:
    return {
        "log_mean": float(stats["log_mean"]),
        "log_std": float(stats["log_std"]),
        "log_var": float(stats["log_var"]),
        "n_pixels": int(stats["n_pixels"]),
    }


def load_marker_stats_cache(stats_json: Path) -> dict[str, dict[str, float | int]]:
    if not stats_json.exists():
        return {}

    with stats_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_stats = data.get("marker_stats", data)
    cached = {}
    for marker, stats in raw_stats.items():
        try:
            cached[str(marker)] = normalize_marker_stats(stats)
        except (KeyError, TypeError, ValueError):
            continue
    return cached


def save_marker_stats_cache(
    stats_json: Path,
    sample_id: str,
    markers: list[str],
    coord_path: Path,
    img_dir: Path,
    marker_stats: dict[str, dict[str, float | int]],
) -> None:
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "sample_id": sample_id,
        "coord_path": str(coord_path),
        "img_dir": str(img_dir),
        "markers": markers,
        "marker_stats": {
            marker: normalize_marker_stats(marker_stats[marker])
            for marker in sorted(marker_stats)
        },
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with stats_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_or_compute_marker_stats(
    markers: list[str],
    slide_paths: dict[str, Path],
    coords: np.ndarray,
    marker_workers: int,
    stats_workers: int,
    stats_chunk_size: int,
    stats_json: Path,
    sample_id: str,
    coord_path: Path,
    img_dir: Path,
    show_progress: bool,
) -> dict[str, dict[str, float | int]]:
    cached_stats = load_marker_stats_cache(stats_json)
    marker_stats = {
        marker: cached_stats[marker] for marker in markers if marker in cached_stats
    }
    missing_markers = [marker for marker in markers if marker not in marker_stats]

    if marker_stats:
        print(
            f"loaded_marker_stats: {len(marker_stats)}/{len(markers)} from {stats_json}"
        )

    if missing_markers:
        print(f"compute_marker_stats: {','.join(missing_markers)}")
        computed_stats = compute_marker_stats(
            markers=missing_markers,
            slide_paths=slide_paths,
            coords=coords,
            marker_workers=min(marker_workers, len(missing_markers)),
            stats_workers=stats_workers,
            stats_chunk_size=stats_chunk_size,
            show_progress=show_progress,
        )
        marker_stats.update(computed_stats)
        save_marker_stats_cache(
            stats_json,
            sample_id,
            markers,
            coord_path,
            img_dir,
            {**cached_stats, **marker_stats},
        )
        print(f"saved_marker_stats: {stats_json}")
    else:
        print(f"all_marker_stats_loaded: {stats_json}")

    return {marker: marker_stats[marker] for marker in markers}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a whole-cell by marker matrix from registered mIF slides.",
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--markers", nargs="*")
    parser.add_argument("--img-dir", type=Path)
    parser.add_argument("--coord-path", type=Path)
    parser.add_argument("--cell-path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cell_marker_matrix"),
    )
    parser.add_argument("--out-h5ad", type=Path)
    parser.add_argument("--stats-json", type=Path)
    parser.add_argument("--marker-workers", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--stats-workers", type=int, default=16)
    parser.add_argument("--stats-chunk-size", type=int, default=256)
    parser.add_argument("--bbox-margin", type=int, default=BBOX_MARGIN)
    parser.add_argument(
        "--close-mask", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dilate-mask", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.marker_workers < 1:
        raise ValueError("--marker-workers must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.stats_workers < 1:
        raise ValueError("--stats-workers must be >= 1")
    if args.stats_chunk_size < 1:
        raise ValueError("--stats-chunk-size must be >= 1")


def main() -> int:
    args = parse_args()
    validate_args(args)

    started = time.time()
    img_dir = args.img_dir or default_img_dir(args.sample_id)
    coord_path = args.coord_path or default_coord_path(args.sample_id)
    cell_path = args.cell_path or default_cell_path(args.sample_id)
    out_h5ad = args.out_h5ad or (
        args.output_dir / args.sample_id / f"{args.sample_id}_cell_marker_matrix.h5ad"
    )
    stats_json = args.stats_json or (
        out_h5ad.parent / f"{args.sample_id}_marker_stats.json"
    )
    markers = (
        normalize_markers(args.markers) if args.markers else discover_markers(img_dir)
    )
    show_progress = not args.no_progress

    if not markers:
        raise ValueError(f"no marker slides found in {img_dir}")
    if not cell_path.exists():
        raise FileNotFoundError(f"cell file not found: {cell_path}")
    if not coord_path.exists():
        raise FileNotFoundError(f"coord h5 not found: {coord_path}")

    slide_paths = {marker: img_dir / f"{marker}.ome.tiff" for marker in markers}
    missing = [str(path) for path in slide_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("marker slides not found: " + ", ".join(missing))

    coords = load_coords(coord_path)
    cell_data = np.load(cell_path, allow_pickle=True).item()
    polys = np.asarray(cell_data["polys"], dtype=object)
    centroids = np.asarray(cell_data["centroids"], dtype=np.float32)
    patch_index = np.asarray(cell_data["patch_index"], dtype=np.int64)
    area = np.asarray(cell_data["area"], dtype=np.float32)
    cell_ids = np.arange(len(polys), dtype=np.int64)
    if centroids.shape != (len(polys), 2):
        raise ValueError(f"centroids shape does not match cells: {centroids.shape}")

    cells_by_patch = build_items_by_patch(patch_index)
    valid_patch_index = np.asarray(sorted(cells_by_patch), dtype=np.int64)
    if patch_index.max(initial=-1) >= len(coords):
        raise ValueError("cell patch_index contains indices outside coords")

    first_slide = openslide.OpenSlide(str(next(iter(slide_paths.values()))))
    try:
        slide_dims = first_slide.dimensions
    finally:
        first_slide.close()

    marker_workers = min(args.marker_workers, len(markers))

    print(f"sample_id: {args.sample_id}")
    print(f"markers: {','.join(markers)}")
    print(f"num_cells: {len(polys)}")
    print(f"num_patches: {len(valid_patch_index)}")
    print(f"marker_workers: {marker_workers}")
    print(f"matrix_workers: {args.num_workers}")
    print(f"out_h5ad: {out_h5ad}")
    print(f"stats_json: {stats_json}")

    marker_stats = load_or_compute_marker_stats(
        markers=markers,
        slide_paths=slide_paths,
        coords=coords,
        marker_workers=marker_workers,
        stats_workers=args.stats_workers,
        stats_chunk_size=args.stats_chunk_size,
        stats_json=stats_json,
        sample_id=args.sample_id,
        coord_path=coord_path,
        img_dir=img_dir,
        show_progress=show_progress,
    )

    intensity, binary_count, raw_sum, cell_pixel_count = compute_cell_marker_matrix(
        markers=markers,
        slide_paths=slide_paths,
        marker_stats=marker_stats,
        polys=polys,
        cells_by_patch=cells_by_patch,
        valid_patch_index=valid_patch_index,
        slide_dims=slide_dims,
        num_workers=args.num_workers,
        bbox_margin=args.bbox_margin,
        close_mask=args.close_mask,
        dilate_mask=args.dilate_mask,
        show_progress=show_progress,
    )

    metadata = {
        "sample_id": args.sample_id,
        "cell_path": str(cell_path),
        "coord_path": str(coord_path),
        "img_dir": str(img_dir),
        "markers": markers,
        "cell_marker_matrix_schema": "marker_raw_sum_per_area_v2",
        "stats_json": str(stats_json),
        "z_threshold": Z_THRESHOLD,
        "min_area": MIN_AREA,
        "kernel_size": list(KERNEL_SIZE),
        "close_mask": args.close_mask,
        "close_iterations": CLOSE_ITERATIONS,
        "dilate_mask": args.dilate_mask,
        "dilate_kernel_size": list(DILATE_KERNEL_SIZE),
        "dilate_iterations": DILATE_ITERATIONS,
        "bbox_margin": args.bbox_margin,
        "marker_workers": marker_workers,
        "num_workers": args.num_workers,
        "stats_workers": args.stats_workers,
        "stats_chunk_size": args.stats_chunk_size,
        "marker_stats": marker_stats,
        "elapsed_seconds": time.time() - started,
    }
    write_cell_marker_h5ad(
        out_h5ad,
        cell_ids,
        area,
        cell_pixel_count,
        centroids,
        polys,
        markers,
        intensity,
        binary_count,
        raw_sum,
        metadata,
    )

    print(f"saved_h5ad: {out_h5ad}")
    print(f"elapsed_seconds: {time.time() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
