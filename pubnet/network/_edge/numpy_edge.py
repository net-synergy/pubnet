"""Implementation of the Edge class storing edges as numpy arrays."""

import numpy as np
from scipy import sparse as sp
from scipy.stats import rankdata

from ._base import Edge as _AbstractEdge


class Edge(_AbstractEdge):
    """An impelmentation of the Edge class that stores edges as numpy arrays.

    Uses arrays to list the non-zero edges in a sparse matrix form.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self._data = np.genfromtxt(
            self._file_path,
            # All edge values should be integer IDs.
            dtype=np.int64,
            skip_header=1,
        )

    def __str__(self):
        return (
            f"col 0: {self.start_id}\ncol 1: {self.end_id}\n{str(self._data)}"
        )

    def __repr__(self):
        return (
            f"col 0: {self.start_id}\ncol 1: {self.end_id}\n{repr(self._data)}"
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == self.start_id:
                key = 0
            elif key == self.end_id:
                key = 1
            else:
                raise KeyError(
                    f'Key "{key}" not one of "{self.start_id}" or "{self.end_id}".'
                )
            return self._data[:, key]

        return self._data[key]

    def isin(self, column, test_elements):
        """Check which elements of column are members of test_elements.

        Arguments
        ---------
        column : the column to test, can be anything accepted by `__getitem__`.
        test_elements : array, the elemnts to test against.

        Returns
        -------
        isin : array, a boolean array of the same size as
        self[column], such that all elements of self[column][isin] are
        in the set test_elements.
        """

        return np.isin(self[column], test_elements)

    @property
    def shape(self):
        return self._data.shape[0]

    @property
    def overlap(self):
        """Pairwise number of neighbors nodes have in common."""
        if not hasattr(self, "_overlap"):
            setattr(self, "_overlap", _overlap(self._data))

        return self._overlap

    def similarity(self, target_publications, method="shortest_path"):
        """Calculate similarity between publications based on edge's overlap.

        Arguments
        ---------
        target_publication : array, an array of publications to return
            similarity between which must be a subset of all edges in
            `self.overlap`.
        method : {'shortest_path'}, the method to use for
            calculating similarity.

        Returns
        -------
        similarity : a 3 column 2d array, listing the similarity (3rd
            column) between all pairs of publications (1st--2nd
            column) in target_publications. Only non-zero similarities
            are listed.
        """

        all_methods = {"shortest_path": _shortest_path}

        return all_methods[method](target_publications, self.overlap)


def _overlap(edges, weights=None):
    """Calculate the neighbor overlap between nodes.

    For all pairs of nodes in column 0, calculate the number of nodes
    both are connected to.

    Arguments
    ---------
    edges : array, two column array of edges.
    weights : array or None, if not None, weights is a array of the
        same length of edges. Otherwise, edges is assumed to be
        unweighted.

    Returns
    -------
    overlap : a three column array with the node ids in the first two
        columns and the overlap between them in the third, where
        overlap is a count of the number of neighbors the two nodes
        have in common.
    """

    data_type = edges.dtype
    if not weights:
        weights = np.ones((edges.shape[0]), dtype=data_type)

    adj = sp.coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])), dtype=data_type
    ).tocsr()

    res = adj @ adj.T

    res = sp.triu(
        res - sp.diags(res.diagonal(), dtype=data_type, format="csr"),
        format="csr",
    ).tocoo()
    return np.stack((res.row, res.col, res.data), axis=1)


def _shortest_path(target_nodes, overlap):
    """Calculate shortest path using Dijkstra's Algorithm.

    Does not support negative edge weights (which should not be
    meaningful in the context of overlap).

    Notice that target_nodes can be a subset of all nodes in the
    graph in which case only paths between the selected target_nodes
    will be found.
    """

    def renumber(edges, target_nodes):
        """Renumber nodes to have values between 0 and all_nodes.shape[0].
        The target_nodes are brought to the front such that the first
        target_nodes.shape[0] nodes are the target_nodes."""

        edge_nodes = edges[:, 0:2].T.flatten()
        target_locs = np.isin(edge_nodes, target_nodes)
        target_nodes = np.unique(edge_nodes[target_locs])
        edge_nodes[np.logical_not(target_locs)] = (
            edge_nodes[np.logical_not(target_locs)] + 999999999
        )

        edge_ranks = rankdata(edge_nodes, "dense") - 1
        edge_ranks = edge_ranks.reshape((2, -1)).T
        new_edges = edges.copy()
        new_edges[:, 0:2] = edge_ranks

        return new_edges, target_nodes

    all_nodes = np.unique(
        np.concatenate((overlap[:, 0:2].flatten(), target_nodes))
    )

    overlap, target_nodes = renumber(overlap, target_nodes)

    weights = 1 / overlap[:, 2].astype(float)
    overlap = sp.coo_matrix((weights, (overlap[:, 0], overlap[:, 1])))
    overlap_row = overlap.tocsr()
    overlap_col = overlap.tocsc()
    del overlap

    # dist(dest, src)
    # Due to renumbering nodes, the top target_nodes.shape[0] rows of
    # dist are the src to src distances.
    target_dist = (
        np.zeros((all_nodes.shape[0], target_nodes.shape[0]), dtype=float)
        + np.Inf
    )
    # May be able to reuse already found paths in previous iterations
    # but do that later.

    max_row = max(overlap_col.indices)
    max_col = max(overlap_row.indices)
    for src in range(target_nodes.shape[0]):
        dist = np.zeros((all_nodes.shape[0],), dtype=float) + np.inf
        unmarked = list(range(all_nodes.shape[0]))
        dist[src] = 0
        while len(unmarked) > 0:
            d_j = unmarked.pop(np.argmin(dist[unmarked]))
            if d_j <= max_row:
                d_potential = dist[d_j] + overlap_row[d_j, :].data
                dist[overlap_row[d_j, :].indices] = np.minimum(
                    dist[overlap_row[d_j, :].indices], d_potential
                )

            if d_j <= max_col:
                d_potential = dist[d_j] + overlap_col[:, d_j].data
                dist[overlap_col[:, d_j].indices] = np.minimum(
                    dist[overlap_col[:, d_j].indices], d_potential
                )

        # So self loops get removed with any overlap that don't exist.
        dist[src] = np.Inf
        target_dist[src, :] = dist[0 : target_nodes.shape[0]]

    out = np.zeros(
        (
            int((target_dist < np.Inf).sum() / 2),
            3,
        )
    )
    count = 0
    for i in range(target_nodes.shape[0]):
        for j in range(i + 1, target_nodes.shape[0]):
            if target_dist[i, j] < np.Inf:
                out[count, 0] = target_nodes[i]
                out[count, 1] = target_nodes[j]
                out[count, 2] = target_dist[i, j]
                count += 1

    return out