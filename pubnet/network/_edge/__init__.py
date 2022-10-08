"""Provides classes for storing graph edges as different representations. """

import re

import numpy as np

from ._base import Edge
from .compressed_edge import CompressedEdge
from .numpy_edge import NumpyEdge

__all__ = ["from_file", "from_data", "Edge", "id_dtype"]

_edge_class = {"numpy": NumpyEdge, "igraph": CompressedEdge}
id_dtype = np.int64


def from_file(file, representation):
    with open(file, "r") as f:
        header_line = f.readline()

    ids = re.findall(r":((?:START)|(?:END))_ID\((\w+)\)", header_line)
    for id, node in ids:
        if id == "START":
            start_id = node
        elif id == "END":
            end_id = node

    data = np.genfromtxt(
        file,
        # All edge values should be integer IDs.
        dtype=id_dtype,
        skip_header=1,
    )
    if ids[0][0] == "END":
        data = data[:, [1, 0]]

    return from_data(data, start_id, end_id, representation)


def from_data(data, start_id, end_id, representation):
    return _edge_class[representation](data, start_id, end_id)
