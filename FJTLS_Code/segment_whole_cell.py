"""Convert nuclei polygons into whole-cell polygons for one sample."""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.whole_cell import (
    PATCH_SIZE,
    build_context_nuclei_by_patch,
    build_nuclei_by_patch,
    init_worker,
    iter_patch_results,
    merge_patch_cells,
)

warnings.filterwarnings(
    "ignore",
    message="Only one label was provided to `remove_small_objects`.*",
    category=UserWarning,
)

DEFAULT_CONTEXT_PX = 32
DEFAULT_MAX_EXPAND_PX = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_id",
        type=str,
        default="21-08371",
        help="Sample ID for processing",
    )
    parser.add_argument(
        "--nuclei_dir",
        type=str,
        default="data/nucleiseg",
        help="Directory containing nuclei data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/wholecellseg",
        help="Output directory for whole-cell segmentation results",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of workers for per-patch cell segmentation. Use 1 for serial.",
    )
    parser.add_argument(
        "--backend",
        choices=("process", "thread"),
        default="process",
        help="Parallel backend. Process backend is faster for the Voronoi algorithm.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="Patch chunksize for process backend.",
    )
    parser.add_argument(
        "--context-px",
        type=int,
        default=DEFAULT_CONTEXT_PX,
        help=(
            "Run Voronoi on a patch plus this halo on all sides, then keep only "
            "cells whose nucleus centroid belongs to the original patch."
        ),
    )
    parser.add_argument(
        "--max-expand-px",
        type=int,
        default=DEFAULT_MAX_EXPAND_PX,
        help="Maximum nucleus expansion radius passed to nuclei_to_cell_voronoi.",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")
    if args.context_px < 0:
        raise ValueError("--context-px must be >= 0")
    if args.max_expand_px < 0:
        raise ValueError("--max-expand-px must be >= 0")


def polygons_to_uint32_object_array(polygons: list) -> np.ndarray:
    out = np.empty(len(polygons), dtype=object)
    for idx, poly in enumerate(polygons):
        out[idx] = np.asarray(poly, dtype=np.uint32)
    return out


def main():
    args = parse_args()
    validate_args(args)

    sample_id = args.sample_id
    nuclei_dir = Path(args.nuclei_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nuclei_path = nuclei_dir / f"{sample_id}.npy"
    output_path = output_dir / f"{sample_id}.npy"
    nuclei_data = np.load(nuclei_path, allow_pickle=True).item()

    polygons = np.asarray(nuclei_data["outlines_pred"], dtype=object)
    coords = np.asarray(nuclei_data["coords"], dtype=np.int64)
    centroids = np.asarray(nuclei_data["centroid"], dtype=np.float32)
    patch_index = np.asarray(nuclei_data["patch_index"], dtype=np.int64)

    valid_patch_index = np.unique(patch_index)
    nuclei_by_patch = build_nuclei_by_patch(patch_index)
    context_nuclei_by_patch = build_context_nuclei_by_patch(
        coords,
        nuclei_by_patch,
        args.context_px,
        PATCH_SIZE,
    )
    init_worker(
        polygons,
        coords,
        centroids,
        nuclei_by_patch,
        context_nuclei_by_patch,
        args.context_px,
        args.max_expand_px,
    )

    cell_data = {
        "polys": [],
        "centroids": [],
        "patch_index": [],
        "nuclei_index": [],
        "area": [],
        "fallback": [],
        "contains_centroid": [],
    }
    results, executor = iter_patch_results(
        valid_patch_index,
        polygons,
        coords,
        centroids,
        nuclei_by_patch,
        context_nuclei_by_patch,
        args.num_workers,
        args.backend,
        args.chunksize,
        args.context_px,
        args.max_expand_px,
    )

    try:
        for patch_cells in tqdm(
            results,
            total=len(valid_patch_index),
            desc=f"nuclei to cell ({args.backend}, workers={args.num_workers})",
        ):
            merge_patch_cells(cell_data, patch_cells)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    cell_data["polys"] = polygons_to_uint32_object_array(cell_data["polys"])
    cell_data["centroids"] = np.asarray(cell_data["centroids"], dtype=np.float32)
    cell_data["patch_index"] = np.asarray(cell_data["patch_index"], dtype=np.int64)
    cell_data["nuclei_index"] = np.asarray(cell_data["nuclei_index"], dtype=np.int64)
    cell_data["area"] = np.asarray(cell_data["area"], dtype=np.float32)
    cell_data["fallback"] = np.asarray(cell_data["fallback"], dtype=bool)
    cell_data["contains_centroid"] = np.asarray(
        cell_data["contains_centroid"], dtype=bool
    )
    cell_data["coords"] = coords
    cell_data["metadata"] = {
        "sample_id": sample_id,
        "nuclei_path": str(nuclei_path),
        "patch_size": int(PATCH_SIZE[0]),
        "context_px": int(args.context_px),
        "max_expand_px": int(args.max_expand_px),
        "num_workers": int(args.num_workers),
        "backend": args.backend,
        "chunksize": int(args.chunksize),
        "n_patches": int(len(coords)),
        "n_valid_patches": int(len(valid_patch_index)),
        "n_cells": int(len(cell_data["polys"])),
        "n_fallback": int(np.count_nonzero(cell_data["fallback"])),
        "n_contains_centroid": int(np.count_nonzero(cell_data["contains_centroid"])),
        "segmentation_mode": "context_voronoi_owner_patch_one_cell_per_nucleus",
    }
    np.save(output_path, cell_data, allow_pickle=True)

    print(f"Saved to: {output_path}")
    print(f"Num cells: {len(cell_data['polys'])}")


if __name__ == "__main__":
    main()
