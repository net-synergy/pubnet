"""Provides classes for storing graph edges as different representations. """

import gzip
import re

import igraph as ig
import numpy as np

from ._base import Edge
from .igraph_edge import IgraphEdge
from .numpy_edge import NumpyEdge

__all__ = ["from_file", "from_data", "Edge", "id_dtype"]

_edge_class = {"numpy": NumpyEdge, "igraph": IgraphEdge}
id_dtype = np.int64


def from_file(file_name, representation):
    """
    Read edge in from file

    Reads the data in from a file. The file should be in the form
    f"{edge[0]}_{edge[1]}_edges.tsv, where the order the node types
    are given in the edge argument is not important.

    As with the Node class it expects ID columns to be in Neo4j format
    f":START_ID({namespace})" and f":END_ID({namespace})". Start and
    end will be important only if the graph is directed. The
    `namespace` value provides the name of the node and will link to
    that node's ID column.
    """

    ext = file_name.split(".")[-1]

    if ext in ("tsv", "npy", "ig"):
        ext_open = open
    elif ext == "gz":
        ext_open = gzip.open
    else:
        raise ValueError(f"Extension {ext} not supported")

    if ext in ("npy", "ig"):
        header_file = re.sub(
            r"(?:edges)\.(?:\w+)", "edge_header.tsv", file_name
        )
    else:
        header_file = file_name

    with ext_open(header_file, "rt") as f:
        header_line = f.readline()

    ids = re.findall(r":((?:START)|(?:END))_ID\((\w+)\)", header_line)
    for id, node in ids:
        if id == "START":
            start_id = node
        elif id == "END":
            end_id = node

    if ext == "npy":
        data = np.load(file_name, allow_pickle=True)
    elif ext == "ig":
        data = ig.Graph.Read_Pickle(file_name)
    else:
        data = np.genfromtxt(
            file_name,
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
    """
    Make an edge from data.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
    representation : {"numpy", "igraph"}
    start_id, end_id : str, optional
       The name of the to and from node types. If `data` is a ndarray, must be
       provided. For DataFrames, the IDs can be detected based on the column
       names.

    Returns
    -------
    Edge
    """
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
