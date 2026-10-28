from pathlib import Path

import h5py
import numpy as np


def default_img_dir(sample_id: str) -> Path:
    return Path(f"data/registered/{sample_id}/registered_slides")


def default_coord_path(sample_id: str) -> Path:
    return Path(f"data/patches/{sample_id}.h5")


def default_cell_path(sample_id: str) -> Path:
    new_path = Path(f"data/wholecellseg/{sample_id}.npy")
    if new_path.exists():
        return new_path
    return Path(f"data/wholecellseg/{sample_id}_cell.npy")


def load_coords(coord_path: Path) -> np.ndarray:
    with h5py.File(coord_path, "r") as coord_h5:
        return np.asarray(coord_h5["coords"], dtype=np.int64)


def discover_markers(img_dir: Path, exclude_dapi: bool = True) -> list[str]:
    markers = (path.name[: -len(".ome.tiff")] for path in img_dir.glob("*.ome.tiff"))
    if exclude_dapi:
        markers = (marker for marker in markers if marker.upper() != "DAPI")
    return sorted(markers)


def normalize_markers(markers: list[str], exclude_dapi: bool = True) -> list[str]:
    if exclude_dapi:
        return [marker for marker in markers if marker.upper() != "DAPI"]
    return list(markers)
