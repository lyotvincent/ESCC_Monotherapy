from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np

from .cell_draw import mask2poly, nuclei_to_cell_voronoi, polygon2mask

PATCH_SIZE = (256, 256)

POLYGONS = None
COORDS = None
CENTROIDS = None
NUCLEI_BY_PATCH = None
CONTEXT_NUCLEI_BY_PATCH = None
CONTEXT_PX = 32
MAX_EXPAND_PX = 10


def polygon_contains_point(poly: np.ndarray, point: np.ndarray) -> bool:
    poly = np.asarray(poly, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return False
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def fallback_cell_candidate(
    nuclei_i: int,
    origin: np.ndarray,
    mask_shape: tuple[int, int],
    max_expand_px: int,
) -> dict:
    local_poly = np.asarray(POLYGONS[nuclei_i], dtype=np.float32) - origin[None, :]
    local_poly = np.clip(local_poly, [0, 0], [mask_shape[1] - 1, mask_shape[0] - 1])
    mask = polygon2mask([local_poly.astype(np.uint32)], mask_shape, start_value=1)
    mask = (mask > 0).astype(np.uint8)

    if max_expand_px > 0 and np.any(mask):
        kernel_size = 2 * max_expand_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    centroid = np.asarray(CENTROIDS[nuclei_i], dtype=np.float32)
    local_centroid = centroid - origin.astype(np.float32)
    cx = int(round(float(local_centroid[0])))
    cy = int(round(float(local_centroid[1])))
    if 0 <= cx < mask_shape[1] and 0 <= cy < mask_shape[0]:
        foreground = np.argwhere(mask > 0)
        if len(foreground) > 0:
            distances = np.square(foreground[:, 1] - cx) + np.square(
                foreground[:, 0] - cy
            )
            nearest_y, nearest_x = foreground[int(np.argmin(distances))]
            cv2.line(
                mask,
                (cx, cy),
                (int(nearest_x), int(nearest_y)),
                1,
                thickness=max(max_expand_px // 2, 1),
            )
        cv2.circle(mask, (cx, cy), max(max_expand_px, 1), 1, -1)

    if not np.any(mask):
        if 0 <= cx < mask_shape[1] and 0 <= cy < mask_shape[0]:
            cv2.circle(mask, (cx, cy), max(max_expand_px, 1), 1, -1)

    polys = mask2poly(mask, is_label=False, min_area=0, epsilon=0.5, close=True)
    if not polys:
        cx = int(np.clip(round(float(centroid[0])), 0, np.iinfo(np.uint32).max - 2))
        cy = int(np.clip(round(float(centroid[1])), 0, np.iinfo(np.uint32).max - 2))
        poly_global = np.asarray(
            [
                [cx - 1, cy - 1],
                [cx + 1, cy - 1],
                [cx + 1, cy + 1],
                [cx - 1, cy + 1],
                [cx - 1, cy - 1],
            ],
            dtype=np.int64,
        )
        poly_global[:, 0] = np.clip(poly_global[:, 0], 0, np.iinfo(np.uint32).max)
        poly_global[:, 1] = np.clip(poly_global[:, 1], 0, np.iinfo(np.uint32).max)
        return {
            "poly": poly_global.astype(np.uint32),
            "area": float(cv2.contourArea(poly_global.astype(np.float32))),
            "contains_centroid": True,
            "fallback": True,
        }

    for item in polys:
        item_poly = np.asarray(item["poly"], dtype=np.float32).copy()
        item_poly[:, 0] += origin[0]
        item_poly[:, 1] += origin[1]
        item["poly"] = item_poly
        item["contains_centroid"] = polygon_contains_point(item_poly, centroid)

    polys.sort(
        key=lambda x: (
            int(x["contains_centroid"]),
            float(x["area"]),
        ),
        reverse=True,
    )
    selected = polys[0]
    return {
        "poly": np.asarray(selected["poly"], dtype=np.uint32),
        "area": float(selected["area"]),
        "contains_centroid": bool(selected["contains_centroid"]),
        "fallback": True,
    }


def init_worker(
    polygons,
    coords,
    centroids,
    nuclei_by_patch,
    context_nuclei_by_patch,
    context_px,
    max_expand_px,
) -> None:
    global POLYGONS, COORDS, CENTROIDS, NUCLEI_BY_PATCH
    global CONTEXT_NUCLEI_BY_PATCH, CONTEXT_PX, MAX_EXPAND_PX
    POLYGONS = polygons
    COORDS = coords
    CENTROIDS = centroids
    NUCLEI_BY_PATCH = nuclei_by_patch
    CONTEXT_NUCLEI_BY_PATCH = context_nuclei_by_patch
    CONTEXT_PX = int(context_px)
    MAX_EXPAND_PX = int(max_expand_px)


def build_nuclei_by_patch(patch_index):
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


def build_context_nuclei_by_patch(coords, nuclei_by_patch, context_px, patch_size):
    context_px = max(int(context_px), 0)
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    if context_px == 0:
        return {
            int(k): np.asarray(v, dtype=np.int64) for k, v in nuclei_by_patch.items()
        }

    coord_to_patch = {tuple(map(int, coord)): int(i) for i, coord in enumerate(coords)}
    x_steps = range(
        -((context_px + patch_w - 1) // patch_w),
        ((context_px + patch_w - 1) // patch_w) + 1,
    )
    y_steps = range(
        -((context_px + patch_h - 1) // patch_h),
        ((context_px + patch_h - 1) // patch_h) + 1,
    )
    context_by_patch = {}
    for idx, coord in enumerate(coords):
        x, y = int(coord[0]), int(coord[1])
        expanded_x0 = x - context_px
        expanded_y0 = y - context_px
        expanded_x1 = x + patch_w + context_px
        expanded_y1 = y + patch_h + context_px
        context_indices = []
        for dy in y_steps:
            for dx in x_steps:
                neighbor = coord_to_patch.get((x + dx * patch_w, y + dy * patch_h))
                if neighbor is None or neighbor not in nuclei_by_patch:
                    continue
                nx, ny = coords[neighbor]
                if (
                    int(nx) < expanded_x1
                    and int(nx) + patch_w > expanded_x0
                    and int(ny) < expanded_y1
                    and int(ny) + patch_h > expanded_y0
                ):
                    context_indices.extend(nuclei_by_patch[neighbor])
        context_by_patch[int(idx)] = np.asarray(context_indices, dtype=np.int64)
    return context_by_patch


def process_patch(idx):
    patch_cells = {
        "polys": [],
        "centroids": [],
        "patch_index": [],
        "nuclei_index": [],
        "area": [],
        "fallback": [],
        "contains_centroid": [],
    }

    idx = int(idx)
    coord = COORDS[idx].astype(np.int64)
    owner_nuclei_index = NUCLEI_BY_PATCH[idx]
    context_nuclei_index = CONTEXT_NUCLEI_BY_PATCH.get(idx, owner_nuclei_index)
    context_px = max(int(CONTEXT_PX), 0)
    max_expand_px = max(int(MAX_EXPAND_PX), 0)
    origin = coord - np.array([context_px, context_px], dtype=np.int64)
    mask_shape = (
        int(PATCH_SIZE[0]) + 2 * context_px,
        int(PATCH_SIZE[1]) + 2 * context_px,
    )
    owner_nuclei_set = set(map(int, owner_nuclei_index))

    seen_context = set()
    context_owner = []
    context_neighbor = []
    for nuclei_i in context_nuclei_index:
        nuclei_i = int(nuclei_i)
        if nuclei_i in seen_context:
            continue
        seen_context.add(nuclei_i)
        if nuclei_i in owner_nuclei_set:
            context_owner.append(nuclei_i)
        else:
            context_neighbor.append(nuclei_i)

    local_polygons = []
    label_to_nuclei_index = []
    for nuclei_i in context_neighbor + context_owner:
        poly = np.asarray(POLYGONS[nuclei_i], dtype=np.float32) - origin[None, :]
        poly = np.clip(poly, [0, 0], [mask_shape[1] - 1, mask_shape[0] - 1])
        local_polygons.append(poly.astype(np.uint32))
        label_to_nuclei_index.append(int(nuclei_i))

    mask = polygon2mask(local_polygons, mask_shape, start_value=1)
    cell_label, _nuclei_label, _cyto_label = nuclei_to_cell_voronoi(
        mask,
        max_expand_px=max_expand_px,
        min_nucleus_area=1,
    )
    polys = mask2poly(cell_label, is_label=True, min_area=0, epsilon=0.5, close=True)

    candidates_by_nuclei = {}
    for item in polys:
        lab = int(item["label"])
        if lab <= 0:
            continue

        local_i = lab - 1
        if local_i >= len(label_to_nuclei_index):
            continue

        global_nuclei_i = int(label_to_nuclei_index[local_i])
        if global_nuclei_i not in owner_nuclei_set:
            continue

        poly_global = np.asarray(item["poly"], dtype=np.float32).copy()
        poly_global[:, 0] += origin[0]
        poly_global[:, 1] += origin[1]
        centroid = np.asarray(CENTROIDS[global_nuclei_i], dtype=np.float32)
        contains_centroid = polygon_contains_point(poly_global, centroid)
        candidate = {
            "poly": poly_global.astype(np.uint32),
            "area": float(item["area"]),
            "contains_centroid": bool(contains_centroid),
            "fallback": False,
        }
        candidates_by_nuclei.setdefault(global_nuclei_i, []).append(candidate)

    for global_nuclei_i in owner_nuclei_index:
        global_nuclei_i = int(global_nuclei_i)
        candidates = candidates_by_nuclei.get(global_nuclei_i)
        if not candidates:
            candidates = [
                fallback_cell_candidate(
                    global_nuclei_i,
                    origin,
                    mask_shape,
                    max_expand_px,
                )
            ]
        else:
            candidates = list(candidates)
            if not any(candidate["contains_centroid"] for candidate in candidates):
                candidates.append(
                    fallback_cell_candidate(
                        global_nuclei_i,
                        origin,
                        mask_shape,
                        max_expand_px,
                    )
                )
        candidates.sort(
            key=lambda x: (
                int(x["contains_centroid"]),
                -int(x["fallback"]),
                float(x["area"]),
            ),
            reverse=True,
        )
        selected = candidates[0]

        patch_cells["polys"].append(selected["poly"])
        patch_cells["centroids"].append(CENTROIDS[global_nuclei_i])
        patch_cells["patch_index"].append(idx)
        patch_cells["nuclei_index"].append(global_nuclei_i)
        patch_cells["area"].append(float(selected["area"]))
        patch_cells["fallback"].append(bool(selected["fallback"]))
        patch_cells["contains_centroid"].append(bool(selected["contains_centroid"]))

    return patch_cells


def merge_patch_cells(cell_data, patch_cells) -> None:
    for key in cell_data:
        cell_data[key].extend(patch_cells[key])


def iter_patch_results(
    valid_patch_index,
    polygons,
    coords,
    centroids,
    nuclei_by_patch,
    context_nuclei_by_patch,
    num_workers,
    backend,
    chunksize,
    context_px,
    max_expand_px,
):
    if num_workers == 1:
        return map(process_patch, valid_patch_index), None

    if backend == "process":
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(
                polygons,
                coords,
                centroids,
                nuclei_by_patch,
                context_nuclei_by_patch,
                context_px,
                max_expand_px,
            ),
        )
        results = executor.map(process_patch, valid_patch_index, chunksize=chunksize)
    else:
        executor = ThreadPoolExecutor(max_workers=num_workers)
        results = executor.map(process_patch, valid_patch_index)

    return results, executor
