import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import openslide
import cv2
from scipy import ndimage
from tqdm import tqdm


DEFAULT_DAPI_PATH = Path("data/registered/21-07010/registered_slides/DAPI.ome.tiff")
DEFAULT_COORD_PATH = Path("data/patches/21-07010.h5")
DEFAULT_SAVE_DIR = Path("data/nucleiseg")
DEFAULT_PATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment DAPI nuclei with CellposeSAM and save global polygons plus "
            "centroids. Use an environment with Cellpose-SAM installed."
        )
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_DAPI_PATH)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORD_PATH)
    parser.add_argument("--save", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output file name. Defaults to <slide_stem>_cellposesam_nuclei.npy.",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "h5ad"),
        default="npy",
        help="npy stores a Python dict with object-array polygons; h5ad stores polygon_json in obs.",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument(
        "--context-px",
        type=int,
        default=32,
        help=(
            "Read this many extra pixels around every patch before CellposeSAM "
            "inference, then keep only nuclei whose centroid falls inside the "
            "original patch. This prevents nuclei polygons from being clipped "
            "at 256x256 patch borders. Use 0 for the old no-overlap behavior."
        ),
    )
    parser.add_argument(
        "--inference-scale",
        type=float,
        default=1.0,
        help=(
            "Resize each patch before raw CellposeSAM inference, then map "
            "polygons/centroids back to level-0 coordinates. Use <1.0 for speed."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cellpose-batch-size", type=int, default=128)
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Use Cellpose resample=True. Default is faster resample=False.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1",
        help="Comma-separated physical GPU ids used by parent multi-GPU mode.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Number of CellposeSAM worker processes launched on each GPU.",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="Single physical GPU id used by worker/single-GPU mode.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Debug option: segment only the first N patches.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if the target output already exists.",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep multi-GPU worker part files for debugging.",
    )
    parser.add_argument(
        "--worker-index", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--coord-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--coord-end", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def configure_cuda(gpu_id: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def create_cellposesam_model() -> Any:
    import torch
    from cellpose import core, models

    if not core.use_gpu():
        raise RuntimeError(
            "Cellpose cannot see CUDA. Check that this is running in cellpsam3 "
            "and that CUDA_VISIBLE_DEVICES points to GPU 1."
        )
    device = torch.device("cuda:0")
    return models.CellposeModel(gpu=True, pretrained_model="cpsam", device=device)


def output_path(args: argparse.Namespace) -> Path:
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".h5ad" if args.format == "h5ad" else ".npy"
    if args.output_name:
        name = args.output_name
        if not name.endswith(suffix):
            name += suffix
    else:
        slide_stem = Path(args.path).name
        for suffix_part in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
            if slide_stem.endswith(suffix_part):
                slide_stem = slide_stem[: -len(suffix_part)]
                break
        name = f"{slide_stem}_cellposesam_nuclei{suffix}"
    return save_dir / name


def load_coords(coords_path: Path, max_patches: int | None = None) -> np.ndarray:
    with h5py.File(coords_path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"{coords_path} must contain a 'coords' dataset")
        coords = np.asarray(handle["coords"], dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must have shape (N, 2+), got {coords.shape}")
    coords = coords[:, :2]
    if max_patches is not None:
        coords = coords[: max(int(max_patches), 0)]
    return coords


def slice_coords_for_worker(
    coords: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, int]:
    coord_start = max(int(args.coord_start), 0)
    coord_end = (
        len(coords) if args.coord_end is None else min(int(args.coord_end), len(coords))
    )
    if coord_end < coord_start:
        coord_end = coord_start
    return coords[coord_start:coord_end], coord_start


def read_region_gray(
    slide: openslide.OpenSlide, x: int, y: int, patch_size: int
) -> np.ndarray:
    region = slide.read_region((int(x), int(y)), 0, (int(patch_size), int(patch_size)))
    array = np.asarray(region)
    if array.ndim == 3:
        array = array[..., 0]
    return np.ascontiguousarray(array)


def read_context_region_gray(
    slide: openslide.OpenSlide, x: int, y: int, patch_size: int, context_px: int
) -> tuple[np.ndarray, int, int]:
    context_px = max(int(context_px), 0)
    origin_x = int(x) - context_px
    origin_y = int(y) - context_px
    read_size = int(patch_size) + 2 * context_px
    return read_region_gray(slide, origin_x, origin_y, read_size), origin_x, origin_y


def iter_batches(coords: np.ndarray, batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(coords), batch_size):
        end = min(start + batch_size, len(coords))
        yield start, end, coords[start:end]


def outline_to_global_polygon(
    outline: np.ndarray,
    origin_x: float,
    origin_y: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    if outline.ndim != 2 or outline.shape[0] < 4 or outline.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float32)
    polygon = outline.astype(np.float32, copy=True)
    polygon[:, 0] = polygon[:, 0] * float(scale_x) + float(origin_x)
    polygon[:, 1] = polygon[:, 1] * float(scale_y) + float(origin_y)
    if len(polygon) > 0 and not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])
    return polygon


def collect_cells_from_mask(
    mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    patch_index: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    owner_x: int | None = None,
    owner_y: int | None = None,
    owner_size: int | None = None,
) -> list[dict[str, Any]]:
    if mask.size == 0 or int(mask.max()) == 0:
        return []

    from cellpose import utils

    outlines = utils.outlines_list(mask)
    cells: list[dict[str, Any]] = []
    objects = ndimage.find_objects(mask)
    for label_id, label_slice in enumerate(objects, start=1):
        if label_slice is None:
            continue
        row_slice, col_slice = label_slice
        min_row = int(row_slice.start)
        max_row = int(row_slice.stop)
        min_col = int(col_slice.start)
        max_col = int(col_slice.stop)
        local_mask = mask[min_row:max_row, min_col:max_col] == label_id
        ys, xs = np.nonzero(local_mask)
        area = int(ys.size)
        if area == 0:
            continue
        outline_index = label_id - 1
        if outline_index >= len(outlines):
            continue
        polygon = outline_to_global_polygon(
            outlines[outline_index],
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            scale_x=scale_x,
            scale_y=scale_y,
        )
        if len(polygon) < 4:
            continue

        center_y = float(min_row) + float(ys.mean())
        center_x = float(min_col) + float(xs.mean())
        centroid_x = float(origin_x) + float(center_x) * float(scale_x)
        centroid_y = float(origin_y) + float(center_y) * float(scale_y)
        if owner_x is not None and owner_y is not None and owner_size is not None:
            if not (
                float(owner_x) <= centroid_x < float(owner_x + owner_size)
                and float(owner_y) <= centroid_y < float(owner_y + owner_size)
            ):
                continue

        cells.append(
            {
                "patch_index": int(patch_index),
                "patch_x": int(owner_x if owner_x is not None else origin_x),
                "patch_y": int(owner_y if owner_y is not None else origin_y),
                "label_id": int(label_id),
                "centroid": np.array(
                    [
                        centroid_x,
                        centroid_y,
                    ],
                    dtype=np.float32,
                ),
                "area": int(round(float(area) * float(scale_x) * float(scale_y))),
                "bbox": np.array(
                    [
                        int(
                            np.floor(float(origin_x) + float(min_col) * float(scale_x))
                        ),
                        int(
                            np.floor(float(origin_y) + float(min_row) * float(scale_y))
                        ),
                        int(np.ceil(float(origin_x) + float(max_col) * float(scale_x)))
                        - 1,
                        int(np.ceil(float(origin_y) + float(max_row) * float(scale_y)))
                        - 1,
                    ],
                    dtype=np.int64,
                ),
                "polygon": polygon,
            }
        )
    return cells


def predict_masks(
    model: Any, images: list[np.ndarray], args: argparse.Namespace
) -> list[np.ndarray]:
    masks, _flows, _styles = model.eval(
        images,
        batch_size=int(args.cellpose_batch_size),
        resample=bool(args.resample),
        diameter=args.diameter,
        flow_threshold=float(args.flow_threshold),
        cellprob_threshold=float(args.cellprob_threshold),
        min_size=int(args.min_size),
        niter=int(args.niter),
    )
    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        return [masks]
    return list(masks)


def write_npy(
    path: Path,
    cells: list[dict[str, Any]],
    coords: np.ndarray,
    args: argparse.Namespace,
    elapsed_sec: float,
) -> None:
    outlines_pred = np.empty(len(cells), dtype=object)
    for index, cell in enumerate(cells):
        outlines_pred[index] = cell["polygon"]

    centroid = (
        np.vstack([cell["centroid"] for cell in cells]).astype(np.float32)
        if cells
        else np.empty((0, 2), dtype=np.float32)
    )
    bbox = (
        np.vstack([cell["bbox"] for cell in cells]).astype(np.int64)
        if cells
        else np.empty((0, 4), dtype=np.int64)
    )
    patch_index = np.asarray([cell["patch_index"] for cell in cells], dtype=np.int64)
    label_id = np.asarray([cell["label_id"] for cell in cells], dtype=np.int32)
    area = np.asarray([cell["area"] for cell in cells], dtype=np.int32)

    payload = {
        "outlines_pred": outlines_pred,
        "centroid": centroid,
        "spatial": centroid,
        "bbox": bbox,
        "patch_index": patch_index,
        "label_id": label_id,
        "area": area,
        "coords": coords,
        "metadata": {
            "dapi_path": str(args.path),
            "coord_path": str(args.coords),
            "patch_size": int(args.patch_size),
            "context_px": int(args.context_px),
            "inference_scale": float(args.inference_scale),
            "n_patches": int(len(coords)),
            "n_cells": int(len(cells)),
            "gpu_id": str(args.gpu_id),
            "gpu_ids": str(getattr(args, "gpu_ids", "")),
            "workers_per_gpu": int(getattr(args, "workers_per_gpu", 1)),
            "worker_index": getattr(args, "worker_index", None),
            "model": "cellpose_cpsam",
            "niter": int(args.niter),
            "resample": bool(args.resample),
            "batch_size": int(args.batch_size),
            "cellpose_batch_size": int(args.cellpose_batch_size),
            "segmentation_mode": getattr(
                args, "segmentation_mode", "cellposesam_inference"
            ),
            "elapsed_sec": float(elapsed_sec),
        },
    }
    with path.open("wb") as handle:
        np.save(handle, payload, allow_pickle=True)


def write_h5ad(
    path: Path,
    cells: list[dict[str, Any]],
    coords: np.ndarray,
    args: argparse.Namespace,
    elapsed_sec: float,
) -> None:
    try:
        import anndata as ad
        import pandas as pd
        from scipy import sparse
    except ImportError as exc:
        raise ImportError("h5ad output requires anndata, pandas, and scipy") from exc

    centroid = (
        np.vstack([cell["centroid"] for cell in cells]).astype(np.float32)
        if cells
        else np.empty((0, 2), dtype=np.float32)
    )
    obs = pd.DataFrame(
        {
            "patch_index": [cell["patch_index"] for cell in cells],
            "patch_x": [cell["patch_x"] for cell in cells],
            "patch_y": [cell["patch_y"] for cell in cells],
            "label_id": [cell["label_id"] for cell in cells],
            "area": [cell["area"] for cell in cells],
            "bbox_xmin": [int(cell["bbox"][0]) for cell in cells],
            "bbox_ymin": [int(cell["bbox"][1]) for cell in cells],
            "bbox_xmax": [int(cell["bbox"][2]) for cell in cells],
            "bbox_ymax": [int(cell["bbox"][3]) for cell in cells],
            "polygon_json": [
                json.dumps(cell["polygon"].round(2).tolist(), ensure_ascii=False)
                for cell in cells
            ],
        },
        index=[f"cell_{i:09d}" for i in range(len(cells))],
    )
    adata = ad.AnnData(X=sparse.csr_matrix((len(cells), 0)), obs=obs)
    adata.obsm["spatial"] = centroid
    adata.uns["outlines_pred_format"] = "obs['polygon_json'] stores global xy polygons"
    adata.uns["coords"] = coords
    adata.uns["metadata"] = {
        "dapi_path": str(args.path),
        "coord_path": str(args.coords),
        "patch_size": int(args.patch_size),
        "context_px": int(args.context_px),
        "inference_scale": float(args.inference_scale),
        "n_patches": int(len(coords)),
        "n_cells": int(len(cells)),
        "gpu_id": str(args.gpu_id),
        "gpu_ids": str(getattr(args, "gpu_ids", "")),
        "workers_per_gpu": int(getattr(args, "workers_per_gpu", 1)),
        "worker_index": getattr(args, "worker_index", None),
        "model": "cellpose_cpsam",
        "niter": int(args.niter),
        "resample": bool(args.resample),
        "batch_size": int(args.batch_size),
        "cellpose_batch_size": int(args.cellpose_batch_size),
        "segmentation_mode": getattr(
            args, "segmentation_mode", "cellposesam_inference"
        ),
        "elapsed_sec": float(elapsed_sec),
    }
    adata.write_h5ad(path)


def collect_cells_with_cellposesam(
    coords: np.ndarray, args: argparse.Namespace, patch_index_offset: int = 0
) -> list[dict[str, Any]]:
    model = create_cellposesam_model()
    cells: list[dict[str, Any]] = []
    inference_scale = float(args.inference_scale)
    if inference_scale <= 0:
        raise ValueError("--inference-scale must be > 0")
    slide = openslide.OpenSlide(str(args.path))
    context_px = max(int(args.context_px), 0)
    input_size = int(args.patch_size) + 2 * context_px
    try:
        progress = tqdm(
            total=len(coords),
            desc=f"GPU{args.gpu_id} CellposeSAM DAPI",
            unit="patch",
            dynamic_ncols=True,
        )
        for start, _end, batch_coords in iter_batches(coords, args.batch_size):
            regions = [
                read_context_region_gray(
                    slide, int(x), int(y), int(args.patch_size), context_px
                )
                for x, y in batch_coords
            ]
            images = [region[0] for region in regions]
            if inference_scale != 1.0:
                target_size = max(16, int(round(float(input_size) * inference_scale)))
                interpolation = (
                    cv2.INTER_AREA if inference_scale < 1.0 else cv2.INTER_LINEAR
                )
                images = [
                    cv2.resize(
                        image, (target_size, target_size), interpolation=interpolation
                    )
                    for image in images
                ]
            masks = predict_masks(model, images, args)
            for offset, (mask, coord, region) in enumerate(
                zip(masks, batch_coords, regions)
            ):
                patch_index = patch_index_offset + start + offset
                patch_x, patch_y = int(coord[0]), int(coord[1])
                _image, origin_x, origin_y = region
                scale_x = float(input_size) / float(mask.shape[1])
                scale_y = float(input_size) / float(mask.shape[0])
                cells.extend(
                    collect_cells_from_mask(
                        mask,
                        origin_x,
                        origin_y,
                        patch_index,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        owner_x=patch_x,
                        owner_y=patch_y,
                        owner_size=int(args.patch_size),
                    )
                )
            progress.update(len(batch_coords))
            progress.set_postfix(cells=len(cells))
        progress.close()
    finally:
        slide.close()
    return cells


def parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in str(value).split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    return gpu_ids


def expand_gpu_workers(gpu_ids: list[str], workers_per_gpu: int) -> list[str]:
    worker_count = max(int(workers_per_gpu), 1)
    expanded: list[str] = []
    for gpu_id in gpu_ids:
        expanded.extend([gpu_id] * worker_count)
    return expanded


def append_optional_arg(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([name, str(value)])


def build_worker_command(
    args: argparse.Namespace,
    gpu_id: str,
    worker_index: int,
    coord_start: int,
    coord_end: int,
    part_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--path",
        str(args.path),
        "--coords",
        str(args.coords),
        "--save",
        str(part_dir),
        "--output-name",
        f"worker_{worker_index:02d}.npy",
        "--format",
        "npy",
        "--patch-size",
        str(args.patch_size),
        "--context-px",
        str(args.context_px),
        "--inference-scale",
        str(args.inference_scale),
        "--batch-size",
        str(args.batch_size),
        "--cellpose-batch-size",
        str(args.cellpose_batch_size),
        "--niter",
        str(args.niter),
        "--flow-threshold",
        str(args.flow_threshold),
        "--cellprob-threshold",
        str(args.cellprob_threshold),
        "--min-size",
        str(args.min_size),
        "--gpu-id",
        str(gpu_id),
        "--gpu-ids",
        str(gpu_id),
        "--workers-per-gpu",
        "1",
        "--worker-index",
        str(worker_index),
        "--coord-start",
        str(coord_start),
        "--coord-end",
        str(coord_end),
    ]
    append_optional_arg(command, "--diameter", args.diameter)
    append_optional_arg(command, "--max-patches", args.max_patches)
    if args.resample:
        command.append("--resample")
    if args.keep_parts:
        command.append("--keep-parts")
    return command


def load_npy_payload(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def write_merged_npy(
    path: Path,
    part_paths: list[Path],
    coords: np.ndarray,
    args: argparse.Namespace,
    worker_elapsed_sec: float,
) -> None:
    payloads = [load_npy_payload(path) for path in part_paths]
    outlines = [payload["outlines_pred"] for payload in payloads]
    centroids = [
        payload["centroid"] for payload in payloads if len(payload["centroid"])
    ]
    bboxes = [payload["bbox"] for payload in payloads if len(payload["bbox"])]
    patch_indices = [payload["patch_index"] for payload in payloads]
    label_ids = [payload["label_id"] for payload in payloads]
    areas = [payload["area"] for payload in payloads]

    outlines_pred = np.concatenate(outlines) if outlines else np.empty(0, dtype=object)
    centroid = (
        np.vstack(centroids).astype(np.float32)
        if centroids
        else np.empty((0, 2), dtype=np.float32)
    )
    bbox = (
        np.vstack(bboxes).astype(np.int64)
        if bboxes
        else np.empty((0, 4), dtype=np.int64)
    )
    patch_index = (
        np.concatenate(patch_indices) if patch_indices else np.empty(0, dtype=np.int64)
    )
    label_id = np.concatenate(label_ids) if label_ids else np.empty(0, dtype=np.int32)
    area = np.concatenate(areas) if areas else np.empty(0, dtype=np.int32)
    order = (
        np.lexsort((label_id, patch_index))
        if len(patch_index)
        else np.empty(0, dtype=np.int64)
    )

    worker_metadata = [payload.get("metadata", {}) for payload in payloads]
    merged = {
        "outlines_pred": outlines_pred[order],
        "centroid": centroid[order],
        "spatial": centroid[order],
        "bbox": bbox[order],
        "patch_index": patch_index[order],
        "label_id": label_id[order],
        "area": area[order],
        "coords": coords,
        "metadata": {
            "dapi_path": str(args.path),
            "coord_path": str(args.coords),
            "patch_size": int(args.patch_size),
            "context_px": int(args.context_px),
            "inference_scale": float(args.inference_scale),
            "n_patches": int(len(coords)),
            "n_cells": int(len(outlines_pred)),
            "gpu_ids": ",".join(parse_gpu_ids(args.gpu_ids)),
            "workers_per_gpu": int(args.workers_per_gpu),
            "model": "cellpose_cpsam",
            "niter": int(args.niter),
            "resample": bool(args.resample),
            "batch_size": int(args.batch_size),
            "cellpose_batch_size": int(args.cellpose_batch_size),
            "segmentation_mode": "cellposesam_inference_multi_gpu",
            "elapsed_sec": float(worker_elapsed_sec),
            "worker_metadata": worker_metadata,
        },
    }
    with path.open("wb") as handle:
        np.save(handle, merged, allow_pickle=True)


def run_multi_gpu_parent(
    args: argparse.Namespace, target: Path, coords: np.ndarray
) -> None:
    if args.format != "npy":
        raise ValueError("multi-GPU mode currently writes npy output; use --format npy")

    physical_gpu_ids = parse_gpu_ids(args.gpu_ids)
    gpu_ids = expand_gpu_workers(physical_gpu_ids, args.workers_per_gpu)
    worker_count = min(len(gpu_ids), len(coords))
    if worker_count <= 1:
        configure_cuda(gpu_ids[0])
        args.gpu_id = gpu_ids[0]
        args.segmentation_mode = "cellposesam_inference"
        worker_coords, offset = slice_coords_for_worker(coords, args)
        cells = collect_cells_with_cellposesam(worker_coords, args, offset)
        elapsed_sec = time.perf_counter() - args._start_time
        write_npy(
            target.with_suffix(target.suffix + ".tmp"), cells, coords, args, elapsed_sec
        )
        os.replace(target.with_suffix(target.suffix + ".tmp"), target)
        print(
            f"Done: {len(cells)} nuclei from {len(coords)} patches in "
            f"{elapsed_sec:.1f}s ({elapsed_sec / 60.0:.2f} min)."
        )
        return

    part_dir = target.with_suffix(target.suffix + f".parts.{int(time.time())}")
    part_dir.mkdir(parents=True, exist_ok=True)
    boundaries = [
        round(i * len(coords) / worker_count) for i in range(worker_count + 1)
    ]
    processes: list[tuple[int, str, Path, subprocess.Popen[bytes]]] = []
    print(
        f"Launching {worker_count} raw CellposeSAM workers on GPUs "
        f"{','.join(gpu_ids[:worker_count])}"
    )
    for worker_index in range(worker_count):
        coord_start = int(boundaries[worker_index])
        coord_end = int(boundaries[worker_index + 1])
        gpu_id = gpu_ids[worker_index]
        part_path = part_dir / f"worker_{worker_index:02d}.npy"
        command = build_worker_command(
            args=args,
            gpu_id=gpu_id,
            worker_index=worker_index,
            coord_start=coord_start,
            coord_end=coord_end,
            part_dir=part_dir,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"worker {worker_index}: GPU{gpu_id}, coords[{coord_start}:{coord_end}]")
        processes.append(
            (worker_index, gpu_id, part_path, subprocess.Popen(command, env=env))
        )

    failed_workers: list[int] = []
    for worker_index, _gpu_id, _part_path, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed_workers.append(worker_index)
    if failed_workers:
        raise RuntimeError(f"CellposeSAM workers failed: {failed_workers}")

    part_paths = [
        part_path for _worker_index, _gpu_id, part_path, _process in processes
    ]
    for part_path in part_paths:
        if not part_path.exists():
            raise FileNotFoundError(f"Missing worker output: {part_path}")
    elapsed_sec = time.perf_counter() - args._start_time
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    write_merged_npy(tmp_target, part_paths, coords, args, elapsed_sec)
    os.replace(tmp_target, target)
    if not args.keep_parts:
        shutil.rmtree(part_dir, ignore_errors=True)
    merged = load_npy_payload(target)
    print(
        f"Done: {len(merged['outlines_pred'])} nuclei from {len(coords)} patches in "
        f"{elapsed_sec:.1f}s ({elapsed_sec / 60.0:.2f} min)."
    )


def main() -> None:
    args = parse_args()
    target = output_path(args)
    if args.skip_existing and target.exists():
        print(f"Skip existing output: {target}")
        return

    args._start_time = time.perf_counter()
    coords = load_coords(args.coords, args.max_patches)
    coords, patch_index_offset = slice_coords_for_worker(coords, args)
    print(f"Loaded {len(coords)} patches from {args.coords}")
    print(f"Using DAPI slide: {args.path}")
    print(f"Saving {args.format} output to: {target}")
    print(f"Patch size: {args.patch_size}, context px: {max(int(args.context_px), 0)}")

    if args.worker_index is None:
        full_coords = load_coords(args.coords, args.max_patches)
        run_multi_gpu_parent(args, target, full_coords)
        return

    configure_cuda(args.gpu_id)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    args.segmentation_mode = "cellposesam_inference"
    cells = collect_cells_with_cellposesam(coords, args, patch_index_offset)

    elapsed_sec = time.perf_counter() - args._start_time
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    if args.format == "h5ad":
        write_h5ad(tmp_target, cells, coords, args, elapsed_sec)
    else:
        write_npy(tmp_target, cells, coords, args, elapsed_sec)
    os.replace(tmp_target, target)
    print(
        f"Done: {len(cells)} nuclei from {len(coords)} patches in "
        f"{elapsed_sec:.1f}s ({elapsed_sec / 60.0:.2f} min)."
    )


if __name__ == "__main__":
    main()
