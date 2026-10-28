import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import binary_dilation, disk, remove_small_objects


def polygon2mask(
    polygons,
    image_shape,
    values=None,
    start_value=1,
    background=0,
    dtype=np.uint32,
    clip=True,
):
    h, w = image_shape[:2]
    mask = np.full((h, w), background, dtype=np.int32)
    n_poly = len(polygons)

    if values is None:
        values = range(start_value, start_value + n_poly)
    elif len(values) != n_poly:
        raise ValueError("values length must match polygons length")

    for poly, value in zip(polygons, values):
        if poly is None:
            continue

        points = np.asarray(poly)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue

        points = np.round(points).astype(np.int32)
        if clip:
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)

        cv2.fillPoly(mask, [points], color=int(value))

    return mask.astype(dtype)


def mask2poly(
    mask,
    is_label=False,
    min_area=20,
    epsilon=1.0,
    offset=(0, 0),
    close=False,
):
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    x_off, y_off = offset
    polygons = []

    def single_mask_to_polys(bin_mask, label_id=None):
        bin_mask = (bin_mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            bin_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            if epsilon > 0:
                contour = cv2.approxPolyDP(contour, epsilon, True)

            poly = contour.reshape(-1, 2).astype(np.float32)
            if len(poly) < 3:
                continue

            poly[:, 0] += x_off
            poly[:, 1] += y_off

            if close and not np.array_equal(poly[0], poly[-1]):
                poly = np.vstack([poly, poly[0]])

            results.append({"label": label_id, "area": float(area), "poly": poly})

        return results

    if is_label:
        labels = np.unique(mask)
        labels = labels[labels != 0]
        for label_id in labels:
            polygons.extend(
                single_mask_to_polys(mask == label_id, label_id=int(label_id))
            )
    else:
        polygons.extend(single_mask_to_polys(mask, label_id=1))

    return polygons


def nuclei_to_cell_voronoi(
    nuclei_mask,
    max_expand_px=12,
    tissue_mask=None,
    min_nucleus_area=20,
):
    nuclei_mask = np.asarray(nuclei_mask)

    if nuclei_mask.max() <= 1:
        nuclei_label = label(nuclei_mask > 0).astype(np.int32)
    else:
        nuclei_label = nuclei_mask.astype(np.int32)

    nuclei_label = remove_small_objects(nuclei_label, min_size=min_nucleus_area).astype(
        np.int32
    )
    nuclei_bin = nuclei_label > 0

    allowed_mask = binary_dilation(nuclei_bin, footprint=disk(max_expand_px))
    if tissue_mask is not None:
        allowed_mask = allowed_mask & (tissue_mask > 0)

    _dist, indices = ndi.distance_transform_edt(~nuclei_bin, return_indices=True)
    nearest_label = nuclei_label[tuple(indices)]
    cell_label = np.where(allowed_mask, nearest_label, 0).astype(np.int32)

    cyto_label = cell_label.copy()
    cyto_label[nuclei_bin] = 0

    return cell_label, nuclei_label, cyto_label
