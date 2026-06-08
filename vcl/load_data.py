"""Data loading helpers for the Virtual Climate Lab.

Currently exposes a single public function, ``load_preprocessed``, which
deserialises a dataset previously written by ``vcl.serialize.save``.
The legacy ``load()`` function (which pointed to a hard-coded developer
path) has been removed – use the preprocessing pipeline instead.
"""

from pathlib import Path
from typing import Union

import vcl.serialize


def load_preprocessed(data_path: Union[str, Path]):
    """Load preprocessed VCL data from a compressed zarr store.

    Args:
        data_path: Directory path previously written by ``vcl.serialize.save``.
    """
    return vcl.serialize.load(data_path)
