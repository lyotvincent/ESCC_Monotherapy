import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def build_items_by_patch(patch_index: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(patch_index, kind="stable")
    sorted_patch_index = patch_index[order]
    starts = np.r_[
        0, np.flatnonzero(sorted_patch_index[1:] != sorted_patch_index[:-1]) + 1
    ]
    ends = np.r_[starts[1:], len(sorted_patch_index)]
    return {
        int(sorted_patch_index[start]): order[start:end]
        for start, end in zip(starts, ends)
    }


def patch_bbox(
    polys: np.ndarray,
    cell_indices: np.ndarray,
    slide_w: int,
    slide_h: int,
    margin: int,
):
    mins = []
    maxs = []
    for cell_i in cell_indices:
        poly = np.asarray(polys[int(cell_i)], dtype=np.int32)
        if poly.size == 0:
            continue
        mins.append(poly.min(axis=0))
        maxs.append(poly.max(axis=0))

    if not mins:
        return None

    xy_min = np.min(np.vstack(mins), axis=0) - margin
    xy_max = np.max(np.vstack(maxs), axis=0) + margin + 1
    x0 = max(int(xy_min[0]), 0)
    y0 = max(int(xy_min[1]), 0)
    x1 = min(int(xy_max[0]), slide_w)
    y1 = min(int(xy_max[1]), slide_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def measure_cell(
    poly_global: np.ndarray,
    origin: tuple[int, int],
    zscore: np.ndarray,
    marker_mask: np.ndarray,
):
    x0, y0 = origin
    poly = np.asarray(poly_global, dtype=np.int32) - np.array([x0, y0], dtype=np.int32)
    if poly.size == 0:
        return 0.0, 0

    h, w = marker_mask.shape
    px0 = max(int(poly[:, 0].min()), 0)
    py0 = max(int(poly[:, 1].min()), 0)
    px1 = min(int(poly[:, 0].max()) + 1, w)
    py1 = min(int(poly[:, 1].max()) + 1, h)
    if px1 <= px0 or py1 <= py0:
        return 0.0, 0

    local_poly = poly - np.array([px0, py0], dtype=np.int32)
    cell_mask = np.zeros((py1 - py0, px1 - px0), dtype=np.uint8)
    cv2.fillPoly(cell_mask, [local_poly], 1)

    marker_roi = marker_mask[py0:py1, px0:px1] > 0
    positive = (cell_mask > 0) & marker_roi
    count = int(np.count_nonzero(positive))
    if count == 0:
        return 0.0, 0
    intensity = float(zscore[py0:py1, px0:px1][positive].sum(dtype=np.float64))
    return intensity, count


def build_cell_mask_rois(
    polys: np.ndarray,
    cell_indices: np.ndarray,
    origin: tuple[int, int],
    shape: tuple[int, int],
):
    x0, y0 = origin
    h, w = shape
    offset = np.array([x0, y0], dtype=np.int32)
    rois = []

    for cell_i in cell_indices:
        poly = np.asarray(polys[int(cell_i)], dtype=np.int32) - offset
        if poly.size == 0:
            rois.append(None)
            continue

        px0 = max(int(poly[:, 0].min()), 0)
        py0 = max(int(poly[:, 1].min()), 0)
        px1 = min(int(poly[:, 0].max()) + 1, w)
        py1 = min(int(poly[:, 1].max()) + 1, h)
        if px1 <= px0 or py1 <= py0:
            rois.append(None)
            continue

        local_poly = poly - np.array([px0, py0], dtype=np.int32)
        cell_mask = np.zeros((py1 - py0, px1 - px0), dtype=np.uint8)
        cv2.fillPoly(cell_mask, [local_poly], 1)
        rois.append((py0, py1, px0, px1, cell_mask > 0))

    return rois


def measure_cell_mask_rois(
    cell_mask_rois,
    zscore: np.ndarray,
    marker_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = len(cell_mask_rois)
    intensity = np.zeros(n_cells, dtype=np.float32)
    count = np.zeros(n_cells, dtype=np.uint32)

    for out_i, roi in enumerate(cell_mask_rois):
        if roi is None:
            continue
        py0, py1, px0, px1, cell_mask = roi
        positive = cell_mask & (marker_mask[py0:py1, px0:px1] > 0)
        positive_count = int(np.count_nonzero(positive))
        if positive_count == 0:
            continue
        count[out_i] = positive_count
        intensity[out_i] = float(
            zscore[py0:py1, px0:px1][positive].sum(dtype=np.float64)
        )

    return intensity, count


def cell_mask_roi_pixel_counts(cell_mask_rois) -> np.ndarray:
    counts = np.zeros(len(cell_mask_rois), dtype=np.uint32)
    for out_i, roi in enumerate(cell_mask_rois):
        if roi is None:
            continue
        counts[out_i] = int(np.count_nonzero(roi[4]))
    return counts


def measure_cell_raw_sum_rois(cell_mask_rois, arr_channel: np.ndarray) -> np.ndarray:
    raw_sum = np.zeros(len(cell_mask_rois), dtype=np.float32)
    for out_i, roi in enumerate(cell_mask_rois):
        if roi is None:
            continue
        py0, py1, px0, px1, cell_mask = roi
        raw_sum[out_i] = float(
            arr_channel[py0:py1, px0:px1][cell_mask].sum(dtype=np.float64)
        )
    return raw_sum


def write_cell_marker_h5(
    out_h5: Path,
    cell_ids: np.ndarray,
    area: np.ndarray,
    cell_pixel_count: np.ndarray,
    markers: list[str],
    intensity: np.ndarray,
    binary_count: np.ndarray,
    raw_sum: np.ndarray,
    metadata: dict,
) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    feature_names = marker_feature_names(markers)
    columns = ["cell_id", "area", "cell_pixel_count", *feature_names]
    marker_matrix, _feature_names, _measurements = build_marker_feature_matrix(
        markers,
        intensity,
        binary_count,
        raw_sum,
        cell_pixel_count,
    )

    matrix = np.empty((len(cell_ids), len(columns)), dtype=np.float32)
    matrix[:, 0] = cell_ids.astype(np.float32)
    matrix[:, 1] = area.astype(np.float32)
    matrix[:, 2] = cell_pixel_count.astype(np.float32)
    matrix[:, 3:] = marker_matrix

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_h5, "w") as h5:
        h5.create_dataset("matrix", data=matrix, compression="gzip", compression_opts=4)
        h5.create_dataset(
            "columns",
            data=np.asarray(columns, dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "cell_id", data=cell_ids, compression="gzip", compression_opts=4
        )
        h5.create_dataset(
            "area",
            data=area.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset(
            "cell_pixel_count",
            data=cell_pixel_count.astype(np.uint32),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset(
            "marker_names",
            data=np.asarray(markers, dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "log1p_zscore_intensity",
            data=intensity.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset(
            "binary_count",
            data=binary_count.astype(np.uint32),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset(
            "raw_sum",
            data=raw_sum.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        mean_zscore = divide_zero_to_zero(intensity, binary_count)
        positive_fraction = divide_zero_to_zero(binary_count, cell_pixel_count[:, None])
        raw_sum_per_area = divide_zero_to_zero(raw_sum, cell_pixel_count[:, None])
        h5.create_dataset(
            "log1p_zscore_mean",
            data=mean_zscore.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset(
            "positive_fraction",
            data=positive_fraction.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset(
            "raw_sum_per_area",
            data=raw_sum_per_area.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        h5.attrs["metadata"] = json.dumps(metadata)


def marker_feature_specs(markers: list[str]) -> list[tuple[str, str, str]]:
    return [(marker, marker, "raw_sum_per_area") for marker in markers]


def marker_feature_names(markers: list[str]) -> list[str]:
    return [
        feature_name
        for feature_name, _marker, _measurement in marker_feature_specs(markers)
    ]


def divide_zero_to_zero(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )


def build_marker_feature_matrix(
    markers: list[str],
    intensity: np.ndarray,
    binary_count: np.ndarray,
    raw_sum: np.ndarray,
    cell_pixel_count: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    feature_specs = marker_feature_specs(markers)
    feature_names = [
        feature_name for feature_name, _marker, _measurement in feature_specs
    ]
    measurements = [
        measurement for _feature_name, _marker, measurement in feature_specs
    ]
    matrix = np.empty((intensity.shape[0], len(feature_names)), dtype=np.float32)
    raw_sum_per_area = divide_zero_to_zero(raw_sum, cell_pixel_count[:, None])

    for marker_i in range(len(markers)):
        matrix[:, marker_i] = raw_sum_per_area[:, marker_i]
    return matrix, feature_names, measurements


def marker_obs_columns(
    markers: list[str],
    intensity: np.ndarray,
    binary_count: np.ndarray,
    raw_sum: np.ndarray,
    cell_pixel_count: np.ndarray,
) -> dict[str, np.ndarray]:
    columns = {}
    mean_zscore = divide_zero_to_zero(intensity, binary_count)
    positive_fraction = divide_zero_to_zero(binary_count, cell_pixel_count[:, None])
    for marker_i, marker in enumerate(markers):
        columns[f"{marker}_log1p_zscore_intensity"] = intensity[:, marker_i].astype(
            np.float32
        )
        columns[f"{marker}_binary_count"] = binary_count[:, marker_i].astype(np.uint32)
        columns[f"{marker}_log1p_zscore_mean"] = mean_zscore[:, marker_i].astype(
            np.float32
        )
        columns[f"{marker}_positive_fraction"] = positive_fraction[:, marker_i].astype(
            np.float32
        )
        columns[f"{marker}_raw_sum"] = raw_sum[:, marker_i].astype(np.float32)
    return columns


def legacy_marker_feature_names(markers: list[str]) -> list[str]:
    feature_names = []
    for marker in markers:
        feature_names.append(f"{marker}_log1p_zscore_intensity")
        feature_names.append(f"{marker}_binary_count")
    return feature_names


def build_legacy_marker_feature_matrix(
    markers: list[str],
    intensity: np.ndarray,
    binary_count: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    feature_names = legacy_marker_feature_names(markers)
    matrix = np.empty((intensity.shape[0], len(feature_names)), dtype=np.float32)
    for marker_i in range(len(markers)):
        matrix[:, marker_i * 2] = intensity[:, marker_i]
        matrix[:, marker_i * 2 + 1] = binary_count[:, marker_i].astype(np.float32)
    return matrix, feature_names


def pack_polygons(polys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.zeros(len(polys), dtype=np.int64)
    n_points = np.zeros(len(polys), dtype=np.int32)
    vertices = []
    offset = 0

    for idx, poly in enumerate(polys):
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            arr = np.empty((0, 2), dtype=np.float32)
        starts[idx] = offset
        n_points[idx] = arr.shape[0]
        if arr.shape[0] > 0:
            vertices.append(arr)
            offset += arr.shape[0]

    if vertices:
        vertex_array = np.vstack(vertices).astype(np.float32, copy=False)
    else:
        vertex_array = np.empty((0, 2), dtype=np.float32)
    return vertex_array, starts, n_points


def get_cells_in_region(
    coord,
    size,
    adata,
    margin: int | float = 0,
    select_by: str = "intersect",
    local_coords: bool = False,
) -> dict[str, np.ndarray]:
    margin = float(margin)
    if margin < 0:
        raise ValueError("margin must be >= 0")

    x0, y0 = float(coord[0]) - margin, float(coord[1]) - margin
    w, h = float(size[0]), float(size[1])
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive")

    x1 = x0 + w + 2 * margin
    y1 = y0 + h + 2 * margin
    obs = adata.obs
    cell_ids = obs["cell_id"].to_numpy(dtype=np.int64)
    centroids = np.column_stack(
        (
            obs["centroid_x"].to_numpy(dtype=np.float32),
            obs["centroid_y"].to_numpy(dtype=np.float32),
        )
    )
    vertices = np.asarray(adata.uns["polygon_vertices"], dtype=np.float32)
    starts = obs["polygon_start"].to_numpy(dtype=np.int64)
    n_points = obs["polygon_n_points"].to_numpy(dtype=np.int64)

    if select_by == "centroid":
        selected = np.flatnonzero(
            (centroids[:, 0] >= x0)
            & (centroids[:, 0] < x1)
            & (centroids[:, 1] >= y0)
            & (centroids[:, 1] < y1)
        )
    elif select_by == "intersect":
        selected = []
        for idx, (start, n_point) in enumerate(zip(starts, n_points)):
            if n_point <= 0:
                cx, cy = centroids[idx]
                if x0 <= cx < x1 and y0 <= cy < y1:
                    selected.append(idx)
                continue
            poly = vertices[start : start + n_point]
            if (
                poly[:, 0].max() >= x0
                and poly[:, 0].min() < x1
                and poly[:, 1].max() >= y0
                and poly[:, 1].min() < y1
            ):
                selected.append(idx)
        selected = np.asarray(selected, dtype=np.int64)
    else:
        raise ValueError("select_by must be 'intersect' or 'centroid'")

    polygons = np.empty(len(selected), dtype=object)
    offset = np.array([x0, y0], dtype=np.float32)
    for out_i, cell_i in enumerate(selected):
        start = starts[cell_i]
        n_point = n_points[cell_i]
        poly = vertices[start : start + n_point].astype(np.float32, copy=True)
        if local_coords:
            poly -= offset
        polygons[out_i] = poly

    selected_centroids = centroids[selected].astype(np.float32, copy=True)
    if local_coords:
        selected_centroids -= offset

    return {
        "cell_id": cell_ids[selected],
        "centroid": selected_centroids,
        "polygons": polygons,
    }


def write_cell_marker_h5ad(
    out_h5ad: Path,
    cell_ids: np.ndarray,
    area: np.ndarray,
    cell_pixel_count: np.ndarray,
    centroids: np.ndarray,
    polys: np.ndarray,
    markers: list[str],
    intensity: np.ndarray,
    binary_count: np.ndarray,
    raw_sum: np.ndarray,
    metadata: dict,
) -> None:
    import anndata as ad
    import pandas as pd

    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    matrix, feature_names, measurements = build_marker_feature_matrix(
        markers,
        intensity,
        binary_count,
        raw_sum,
        cell_pixel_count,
    )
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.shape != (len(cell_ids), 2):
        raise ValueError(
            f"centroids must have shape ({len(cell_ids)}, 2), got {centroids.shape}"
        )
    polygon_vertices, polygon_start, polygon_n_points = pack_polygons(polys)
    if len(polygon_start) != len(cell_ids):
        raise ValueError(
            f"polys length must match cell_ids length: {len(polygon_start)} != {len(cell_ids)}"
        )

    obs = pd.DataFrame(
        {
            "cell_id": cell_ids.astype(np.int64),
            "area": area.astype(np.float32),
            "cell_pixel_count": cell_pixel_count.astype(np.uint32),
            "centroid_x": centroids[:, 0].astype(np.float32),
            "centroid_y": centroids[:, 1].astype(np.float32),
            "polygon_start": polygon_start,
            "polygon_n_points": polygon_n_points,
            **marker_obs_columns(
                markers,
                intensity,
                binary_count,
                raw_sum,
                cell_pixel_count,
            ),
        },
        index=[str(int(cell_id)) for cell_id in cell_ids],
    )
    var = pd.DataFrame(
        {
            "feature_name": feature_names,
            "marker": [
                marker
                for _feature_name, marker, _measurement in marker_feature_specs(markers)
            ],
            "measurement": measurements,
        },
        index=feature_names,
    )

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm["spatial"] = centroids.astype(np.float32)
    adata.uns["polygon_vertices"] = polygon_vertices
    adata.uns["polygon_format"] = (
        "For cell i, use obs['polygon_start'] and obs['polygon_n_points'] "
        "to slice uns['polygon_vertices']; vertices are global x,y coordinates."
    )
    adata.uns["metadata"] = metadata
    adata.uns["markers"] = list(markers)
    adata.write_h5ad(out_h5ad, compression="gzip")


def write_polygon_h5(out_h5: Path, cell_ids: np.ndarray, polys: np.ndarray) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    polygon_dtype = h5py.vlen_dtype(np.dtype("float32"))
    polygons = np.empty(len(cell_ids), dtype=object)
    n_points = np.zeros(len(cell_ids), dtype=np.int32)

    for idx, poly in enumerate(polys):
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            arr = np.empty((0, 2), dtype=np.float32)
        n_points[idx] = arr.shape[0]
        polygons[idx] = arr.reshape(-1)

    with h5py.File(out_h5, "w") as h5:
        h5.create_dataset(
            "columns",
            data=np.asarray(["cell_id", "polygon"], dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "cell_id",
            data=cell_ids.astype(np.int64),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset("polygon", data=polygons, dtype=polygon_dtype)
        h5.create_dataset(
            "polygon_n_points", data=n_points, compression="gzip", compression_opts=4
        )
        h5.attrs["polygon_format"] = (
            "flattened float32 x,y pairs; reshape each row to (-1, 2)"
        )
