from pathlib import Path

import numpy as np
import tifffile

TILE_SIZE = 256
COMPRESSION = "lzw"


def merge_max_patch_tiles(
    tile_map: dict[tuple[int, int], np.ndarray],
    x: int,
    y: int,
    patch: np.ndarray,
) -> None:
    if not patch.any():
        return

    patch_h, patch_w = patch.shape
    x_end = x + patch_w
    y_end = y + patch_h

    for tile_y in range(y // TILE_SIZE, (y_end - 1) // TILE_SIZE + 1):
        tile_top = tile_y * TILE_SIZE
        dst_y0 = max(y, tile_top) - tile_top
        dst_y1 = min(y_end, tile_top + TILE_SIZE) - tile_top
        src_y0 = tile_top + dst_y0 - y
        src_y1 = tile_top + dst_y1 - y

        for tile_x in range(x // TILE_SIZE, (x_end - 1) // TILE_SIZE + 1):
            tile_left = tile_x * TILE_SIZE
            dst_x0 = max(x, tile_left) - tile_left
            dst_x1 = min(x_end, tile_left + TILE_SIZE) - tile_left
            src_x0 = tile_left + dst_x0 - x
            src_x1 = tile_left + dst_x1 - x

            src = patch[src_y0:src_y1, src_x0:src_x1]
            if not src.any():
                continue

            key = (tile_y, tile_x)
            tile = tile_map.get(key)
            if tile is None:
                tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                tile_map[key] = tile

            dst = tile[dst_y0:dst_y1, dst_x0:dst_x1]
            np.maximum(dst, src, out=dst)


def iter_tiff_tiles(
    tile_map: dict[tuple[int, int], np.ndarray],
    width: int,
    height: int,
    dtype: np.dtype | type = np.uint8,
):
    tile_cols = (width + TILE_SIZE - 1) // TILE_SIZE
    tile_rows = (height + TILE_SIZE - 1) // TILE_SIZE
    zero_tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=dtype)
    for tile_y in range(tile_rows):
        for tile_x in range(tile_cols):
            yield tile_map.get((tile_y, tile_x), zero_tile)


def pyramid_shapes(width: int, height: int) -> list[tuple[int, int]]:
    shapes = []
    while max(width, height) > TILE_SIZE:
        width = (width + 1) // 2
        height = (height + 1) // 2
        shapes.append((width, height))
    return shapes


def downsample_tile_map(
    tile_map: dict[tuple[int, int], np.ndarray],
    width: int,
    height: int,
) -> tuple[dict[tuple[int, int], np.ndarray], int, int]:
    next_width = (width + 1) // 2
    next_height = (height + 1) // 2
    next_cols = (next_width + TILE_SIZE - 1) // TILE_SIZE
    next_rows = (next_height + TILE_SIZE - 1) // TILE_SIZE
    next_map: dict[tuple[int, int], np.ndarray] = {}

    for next_y in range(next_rows):
        for next_x in range(next_cols):
            block = np.zeros((TILE_SIZE * 2, TILE_SIZE * 2), dtype=np.uint8)
            has_data = False

            for dy in range(2):
                for dx in range(2):
                    child = tile_map.get((next_y * 2 + dy, next_x * 2 + dx))
                    if child is None:
                        continue
                    has_data = True
                    y0 = dy * TILE_SIZE
                    x0 = dx * TILE_SIZE
                    block[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = child

            if not has_data:
                continue

            downsampled = np.maximum.reduce(
                (
                    block[0::2, 0::2],
                    block[1::2, 0::2],
                    block[0::2, 1::2],
                    block[1::2, 1::2],
                )
            )
            if downsampled.any():
                next_map[(next_y, next_x)] = downsampled

    return next_map, next_width, next_height


def write_pyramid_tif(
    tile_map: dict[tuple[int, int], np.ndarray],
    width: int,
    height: int,
    out_tif: Path,
) -> None:
    levels = pyramid_shapes(width, height)
    with tifffile.TiffWriter(out_tif, bigtiff=True, ome=True) as tif:
        tif.write(
            data=iter_tiff_tiles(tile_map, width, height, dtype=np.uint8),
            shape=(height, width),
            dtype=np.uint8,
            photometric="minisblack",
            tile=(TILE_SIZE, TILE_SIZE),
            compression=COMPRESSION,
            subifds=len(levels),
            metadata={"axes": "YX"},
        )

        current_map = tile_map
        current_width = width
        current_height = height
        for level_width, level_height in levels:
            current_map, current_width, current_height = downsample_tile_map(
                current_map,
                current_width,
                current_height,
            )
            tif.write(
                data=iter_tiff_tiles(
                    current_map, level_width, level_height, dtype=np.uint8
                ),
                shape=(level_height, level_width),
                dtype=np.uint8,
                photometric="minisblack",
                tile=(TILE_SIZE, TILE_SIZE),
                compression=COMPRESSION,
                subfiletype=1,
                metadata=None,
            )
