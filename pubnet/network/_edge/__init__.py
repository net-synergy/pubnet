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

    return from_data(data, representation, start_id=start_id, end_id=end_id)


def from_data(
    data, representation, start_id=None, end_id=None, dtype=id_dtype
):
    if start_id is None or end_id is None:
        try:
            columns = data.columns
        except AttributeError:
            columns = None

    if (start_id is None) and (columns is not None):
        start_id = list(
            filter(
                lambda x: x is not None,
                [re.search(r":START_ID\((\w+)\)", name) for name in columns],
            )
        )[0]
        start_id = start_id.groups()[0]

    if (end_id is None) and (columns is not None):
        end_id = list(
            filter(
                lambda x: x is not None,
                [re.search(r":END_ID\((\w+)\)", name) for name in columns],
            )
        )[0]
        end_id = end_id.groups()[0]

        if start_id is None or end_id is None:
            raise TypeError(
                "Missing required keyword argument: 'start_id' or 'end_id'"
            )

    return _edge_class[representation](data, start_id, end_id, dtype)
