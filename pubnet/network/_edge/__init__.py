"""Provides classes for storing graph edges as different representations."""

import gzip
import os
from typing import Any, Optional

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from pubnet.network._utils import (
    edge_file_parts,
    edge_gen_file_name,
    edge_header_parts,
)

from ._base import Edge
from .igraph_edge import IgraphEdge
from .numpy_edge import NumpyEdge

__all__ = ["from_file", "from_data", "from_edge", "Edge", "id_dtype"]

_edge_class = {"numpy": NumpyEdge, "igraph": IgraphEdge}
id_dtype = np.int64


def from_file(file_name: str, representation: str) -> Edge:
    """Read edge in from file.

    Reads the data in from a file. The file should be in the form
    f"{edge[0]}_{edge[1]}_edges.tsv, where the order the node types
    are given in the edge argument is not important.

    As with the Node class it expects ID columns to be in Neo4j format
    f":START_ID({namespace})" and f":END_ID({namespace})". Start and
    end will be important only if the graph is directed. The
    `namespace` value provides the name of the node and will link to
    that node's ID column.
    """
    name, ext = edge_file_parts(file_name)

    if ext in ("npy", "pickle"):
        header_file = edge_gen_file_name(
            name, ext, os.path.split(file_name)[0]
        )[1]
    else:
        header_file = file_name

    if ext in ("tsv", "npy", "pickle"):
        with open(header_file, "rt") as f:
            header_line = f.readline()
    elif ext == "tsv.gz":
        with gzip.open(header_file, "rt") as f:
            header_line = f.readline()
    else:
        raise ValueError(f"Extension {ext} not supported")

    start_id, end_id, feature_ids, flip = edge_header_parts(header_line)

    if ext == "npy":
        data = np.load(file_name, allow_pickle=True)
    elif ext == "pickle":
        data = ig.Graph.Read_Pickle(file_name)
    else:
        data = np.genfromtxt(
            file_name,
            # All edge values should be integer IDs.
            skip_header=1,
        )

    # FIXME: Does not handle if start and end ids are not the first two columns
    if flip:
        data = data[:, [1, 0]]

    if isinstance(data, np.ndarray) and data.shape[1] > 2:
        features = {feat: data[:, col] for col, feat in enumerate(feature_ids)}
        data = data[:, :2]
    elif isinstance(data, ig.Graph) and representation == "numpy":
        features = {feat: data.es[feat] for feat in feature_ids}
    else:  # If data is an igraph.Graph, features are already in the graph
        features = {}

    if isinstance(data, np.ndarray):
        data = data.astype(id_dtype)

    return from_data(data, name, features, start_id, end_id, representation)


def from_data(
    data,
    name: str,
    features: dict[str, NDArray[Any]] = {},
    start_id: Optional[str] = None,
    end_id: Optional[str] = None,
    representation: str = "numpy",
    dtype: type = id_dtype,
) -> Edge:
    """Make an edge from data.

    Parameters
    ----------
    data : numpy.ndarray, igraph.Graph, pandas.DataFrame
        Data to convert to an Edge.
    name : str
        Name of the Edge.
    features : dict
        A dictionary of features to add to the edge set. Defaults to no
        features.
    representation : {"numpy", "igraph"}
        Whether to use the "numpy" or "igraph" Edge backend.
    start_id, end_id : str, optional
       The name of the to and from node types. If `data` is a ndarray, must be
       provided. For DataFrames, the IDs can be detected based on the column
       names.
    dtype : type
       Edge list data type passed to numpy array. Only applicable if using the
       "numpy" backend.

    Returns
    -------
    Edge

    """
    if start_id is None or end_id is None:
        try:
            columns = data.columns
            start_id_i, end_id_i, _, _ = edge_header_parts("\t".join(columns))
        except AttributeError:
            raise ValueError(
                'Either "start_id" or "end_id" was not provided and cannot be'
                " inferred by column names."
            )

    if start_id is None:
        start_id = start_id_i

    if end_id is None:
        end_id = end_id_i

    return _edge_class[representation](
        data, name, start_id, end_id, dtype, features=features
    )


def from_edge(edge: Edge, representation: str) -> Edge:
    """Construct a new edge from a preexisting edge.

    If the template edge is already of the new representation type, return the
    edge unmodified.
    """
    if edge.representation == representation:
        return edge

    feature_dict = {f: edge.feature_vector(f) for f in edge.features()}
    return _edge_class[representation](
        edge.get_edgelist(),
        edge.name,
        edge.start_id,
        edge.end_id,
        edge.dtype,
        features=feature_dict,
    )
