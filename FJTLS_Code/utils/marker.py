import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import openslide
from tqdm import tqdm

PATCH_SIZE = (256, 256)
LEVEL = 0
CHANNEL_IDX = 0

Z_THRESHOLD = 3.0
MIN_AREA = 20
KERNEL_SIZE = (10, 10)
CONNECTIVITY = 8
CLOSE_ITERATIONS = 1
DILATE_KERNEL_SIZE = (5, 5)
DILATE_ITERATIONS = 1


def iter_chunks(items: np.ndarray, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def compute_log1p_stats(
    coords: np.ndarray,
    slide_path: Path,
    num_workers: int,
    chunk_size: int,
    show_progress: bool = True,
    desc: str | None = None,
) -> dict[str, float | int]:
    coord_chunks = list(iter_chunks(np.asarray(coords), chunk_size))
    thread_state = threading.local()

    def get_slide() -> openslide.OpenSlide:
        slide = getattr(thread_state, "slide", None)
        if slide is None:
            slide = openslide.OpenSlide(str(slide_path))
            thread_state.slide = slide
        return slide

    def compute_chunk(coord_chunk):
        slide = get_slide()
        chunk_sum = 0.0
        chunk_sumsq = 0.0
        chunk_n = 0

        for coord in coord_chunk:
            x, y = int(coord[0]), int(coord[1])
            region = slide.read_region((x, y), LEVEL, PATCH_SIZE).convert("RGB")
            arr = np.asarray(region)[..., CHANNEL_IDX].astype(np.float64)
            arr_log = np.log1p(arr)
            chunk_sum += arr_log.sum(dtype=np.float64)
            chunk_sumsq += np.square(arr_log).sum(dtype=np.float64)
            chunk_n += arr_log.size

        return chunk_sum, chunk_sumsq, chunk_n

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(compute_chunk, coord_chunks)
        if show_progress:
            results = tqdm(
                results,
                total=len(coord_chunks),
                desc=desc or f"stats {slide_path.stem}",
                file=sys.stdout,
            )

        total_sum = 0.0
        total_sumsq = 0.0
        total_n = 0
        for chunk_sum, chunk_sumsq, chunk_n in results:
            total_sum += chunk_sum
            total_sumsq += chunk_sumsq
            total_n += chunk_n

    if total_n == 0:
        raise ValueError("coords is empty")

    mean = total_sum / total_n
    var = max(total_sumsq / total_n - mean**2, 0.0)
    return {
        "log_mean": float(mean),
        "log_std": float(np.sqrt(var)),
        "log_var": float(var),
        "n_pixels": int(total_n),
    }


def log1p_zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((np.log1p(arr.astype(np.float32)) - mean) / (std + 1e-8)).astype(np.float32)


def clean_marker_mask(mask: np.ndarray, close_mask: bool) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=CONNECTIVITY,
    )

    clean_mask = np.zeros_like(binary, dtype=np.uint8)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean_mask[labels == idx] = 255

    if not close_mask:
        return clean_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
    return cv2.morphologyEx(
        clean_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=CLOSE_ITERATIONS,
    )


def dilate_marker_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATE_KERNEL_SIZE)
    return cv2.dilate(mask, kernel, iterations=DILATE_ITERATIONS)


def build_marker_arrays(
    arr_channel: np.ndarray,
    log_mean: float,
    log_std: float,
    close_mask: bool,
    dilate_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    zscore = log1p_zscore(arr_channel, log_mean, log_std)
    marker_mask = clean_marker_mask(zscore >= Z_THRESHOLD, close_mask)
    if dilate_mask:
        marker_mask = dilate_marker_mask(marker_mask)
    return zscore, marker_mask


def build_masked_outputs(
    arr_channel: np.ndarray,
    write_w: int,
    write_h: int,
    log_mean: float,
    log_std: float,
    build_mask_patch: bool,
    close_mask: bool,
    dilate_mask: bool,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    zscore, marker_mask = build_marker_arrays(
        arr_channel,
        log_mean,
        log_std,
        close_mask,
        dilate_mask,
    )
    binary_mask_patch = (marker_mask[:write_h, :write_w] > 0).astype(np.uint8)
    patch = arr_channel[:write_h, :write_w]
    zscore_patch = zscore[:write_h, :write_w]
    masked_patch = (
        np.where(binary_mask_patch > 0, patch, 0).astype(np.uint8)
        if build_mask_patch
        else None
    )
    masked_zscore_patch = np.where(binary_mask_patch > 0, zscore_patch, 0).astype(
        np.float32
    )
    return masked_patch, masked_zscore_patch, binary_mask_patch
