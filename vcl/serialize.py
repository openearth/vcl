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
from typing import Any, Union, Iterable, Tuple

import numpy as np
import zarr
from zarr.codecs import Blosc

import warnings

warnings.filterwarnings(
    "ignore", message="Numcodecs codecs are not in the Zarr version 3 specification"
)


logger = logging.getLogger(__name__)

_CODEC = zarr.codecs.Zstd(level=3)
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


def _choose_chunk_shape(
    shape: Iterable[int], dtype: np.dtype, target_chunk_bytes: int = 2 * 1024 * 1024
) -> Tuple[int, ...]:
    """
    Heuristic: pick a regular chunk shape with product near target bytes,
    keeping full length on the last axis and shrinking earlier axes.
    This works well for arrays accessed by trailing-axis slices.
    """
    shape = tuple(int(s) for s in shape)
    itemsize = np.dtype(dtype).itemsize
    if itemsize == 0:
        itemsize = 1

    # Start with full shape and shrink from the front until we fit near the target.
    chunks = list(shape)

    def chunk_nbytes(chunks_):
        n = 1
        for c in chunks_:
            n *= max(1, c)
        return n * itemsize

    # Cap each dimension at its size (obvious) and at least 1.
    for i, s in enumerate(chunks):
        chunks[i] = max(1, min(s, s))

    # If already small, just return shape (won't over-chunk tiny arrays)
    if chunk_nbytes(chunks) <= target_chunk_bytes:
        return tuple(chunks)

    # Iteratively reduce from leading dims (except keep last axis larger for cache-friendly reads)
    i = 0
    while chunk_nbytes(chunks) > target_chunk_bytes and any(c > 1 for c in chunks[:-1]):
        if i < len(chunks) - 1:
            if chunks[i] > 1:
                chunks[i] = np.ceil(chunks[i] / 2)
        else:
            # If we reached the last axis and still too big, halve it as well
            if chunks[i] > 1:
                chunks[i] = np.ceil(chunks[i] / 2)
        i = (i + 1) % len(chunks)

    # Ensure no zero
    chunks = [max(1, int(c)) for c in chunks]
    return tuple(chunks)


def save(datasets: dict, output_path: Union[str, Path]) -> None:
    """Save *datasets* to *output_path* using Zarr v3 + pickle."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1) Walk the tree and extract arrays
    arrays: list[np.ndarray] = []
    skeleton = _to_skeleton(datasets, arrays)

    # 2) Write arrays to Zarr v3 with compression
    zarr_path = output_path / "arrays.zarr"
    store = zarr.open(str(zarr_path), mode="w", zarr_version=3)

    total_before = 0
    total_after = 0
    for i, arr in enumerate(arrays):
        key = _make_zarr_key(i)

        chunk_shape = _choose_chunk_shape(arr.shape, arr.dtype)

        z = store.require_dataset(
            key,
            shape=arr.shape,
            dtype=arr.dtype,
            compressors=[_CODEC],  # Change 'codecs' to 'compressors'
            chunks=chunk_shape,  # Note: 'chunk_shape' is often just 'chunks' in many Zarr 3.x builds
            fill_value=None,
            overwrite=True,
        )

        z[...] = arr
        total_before += arr.nbytes
        try:
            total_after += z.nbytes_stored()  # available in recent v3 alphas
        except AttributeError:
            pass

    ratio = total_before / total_after if total_after else float("inf")
    logger.info(
        "Zarr arrays: %.1f MB → %.1f MB (%.1fx compression)",
        total_before / 1e6,
        (total_after / 1e6) if total_after else 0.0,
        ratio,
    )

    # 3) Pickle the skeleton (contains all non-array objects)
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
