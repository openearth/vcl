"""Compressed serialization for VCL preprocessed datasets.

Saves/loads the nested dict of preprocessed data produced by ``vcl.preprocess``.
Numpy arrays are stored in a Zarr store (with Blosc LZ4 compression), which
gives typically 3-10x smaller files compared to raw pickle.  All non-array
values (shapely geometries, GeoDataFrames, plain Python objects) are pickled
into a small sidecar file alongside the Zarr store.

Format on disk::

    <output_path>/
        arrays.zarr/   - Zarr group with all numpy arrays (Blosc-compressed)
        meta.pkl       - Pickle file containing:
                           - 'structure': nested skeleton of the original dict
                             with arrays replaced by a sentinel referencing the zarr key
                           - 'zarr_key_map': mapping from zarr key -> original path info

The ``output_path`` is a directory (conventionally named ``preprocessed-data``).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np
import zarr
from numcodecs import Blosc

logger = logging.getLogger(__name__)

_CODEC = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
_ARRAY_SENTINEL = "__zarr_array__"


def _make_zarr_key(index: int) -> str:
    """Return a safe zarr dataset name for the i-th array."""
    return f"arr_{index:06d}"


# ---------------------------------------------------------------------------
# Serialise: walk the tree and build a serialisable skeleton
# ---------------------------------------------------------------------------


def _to_skeleton(obj: Any, arrays: list) -> Any:
    """Recursively convert *obj* to a JSON-serialisable skeleton.

    Numpy arrays are replaced by a sentinel dict and appended to *arrays*.
    Everything else that is not a basic Python container is left as-is
    (to be pickled as part of the skeleton via pickle, not json).
    """
    if isinstance(obj, np.ndarray):
        idx = len(arrays)
        arrays.append(obj)
        return {_ARRAY_SENTINEL: _make_zarr_key(idx)}

    if isinstance(obj, dict):
        return {
            "__type__": "dict",
            "__items__": {k: _to_skeleton(v, arrays) for k, v in obj.items()},
        }

    if isinstance(obj, list):
        return {"__type__": "list", "__items__": [_to_skeleton(v, arrays) for v in obj]}

    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        return {
            "__type__": "tuple",
            "__items__": [_to_skeleton(v, arrays) for v in obj],
        }

    # Scalars, strings, shapely geometries, GeoDataFrames, etc.
    return obj


def _from_skeleton(skeleton: Any, zarr_store: zarr.Group) -> Any:
    """Inverse of *_to_skeleton*: reconstruct the original Python object."""
    if isinstance(skeleton, dict):
        if _ARRAY_SENTINEL in skeleton:
            return zarr_store[skeleton[_ARRAY_SENTINEL]][:]

        t = skeleton.get("__type__")
        items = skeleton.get("__items__")

        if t == "dict":
            return {k: _from_skeleton(v, zarr_store) for k, v in items.items()}
        if t == "list":
            return [_from_skeleton(v, zarr_store) for v in items]
        if t == "tuple":
            return tuple(_from_skeleton(v, zarr_store) for v in items)

        # Fallback – shouldn't happen, but pass through as-is
        return skeleton

    if isinstance(skeleton, list):
        return [_from_skeleton(v, zarr_store) for v in skeleton]

    return skeleton


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save(datasets: dict, output_path: Union[str, Path]) -> None:
    """Save *datasets* to *output_path* using Zarr + pickle.

    Args:
        datasets: Nested dict returned by ``vcl.preprocess.preprocess``.
        output_path: Destination directory path (created if necessary).
            Conventionally ``<data_dir>/preprocessed-data``.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Walk the tree and extract arrays
    arrays: list = []
    skeleton = _to_skeleton(datasets, arrays)

    # 2. Write arrays to zarr with compression
    zarr_path = output_path / "arrays.zarr"
    store = zarr.open(str(zarr_path), mode="w")

    total_before = 0
    total_after = 0
    for i, arr in enumerate(arrays):
        key = _make_zarr_key(i)
        z = store.require_dataset(
            key,
            shape=arr.shape,
            dtype=arr.dtype,
            compressor=_CODEC,
            chunks=True,
            overwrite=True,
        )
        z[:] = arr
        total_before += arr.nbytes
        total_after += z.nbytes_stored

    ratio = total_before / total_after if total_after else float("inf")
    logger.info(
        "Zarr arrays: %.1f MB → %.1f MB (%.1fx compression)",
        total_before / 1e6,
        total_after / 1e6,
        ratio,
    )

    # 3. Pickle the skeleton (contains all non-array objects)
    meta_path = output_path / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(skeleton, f, protocol=4)

    logger.info("Metadata pickle: %.1f MB", meta_path.stat().st_size / 1e6)
    logger.info("Saved to: %s", output_path)


def load(input_path: Union[str, Path]) -> dict:
    """Load a dataset previously saved with :func:`save`.

    Args:
        input_path: Directory path that was passed to :func:`save`.

    Returns:
        The reconstructed nested dict of preprocessed data.
    """
    input_path = Path(input_path)

    zarr_path = input_path / "arrays.zarr"
    meta_path = input_path / "meta.pkl"

    store = zarr.open(str(zarr_path), mode="r")

    with open(meta_path, "rb") as f:
        skeleton = pickle.load(f)

    return _from_skeleton(skeleton, store)
