""" Similarity metrics written for numpy based edges.
"""

import numpy as np
from scipy import sparse as sp
from scipy.stats import rankdata


def overlap(edges, weights=None):
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


def shortest_path(target_nodes, edges):
    """Calculate shortest path using Dijkstra's Algorithm.

    Does not support negative edge weights (which should not be
    meaningful in the context of overlap).

    Notice that target_nodes can be a subset of all nodes in the
    graph in which case only paths between the selected target_nodes
    will be found.
    """

    all_nodes = np.unique(
        np.concatenate((edges[:, 0:2].flatten(), target_nodes))
    )
    edges, target_nodes = _renumber(edges, target_nodes)

    weights = 1 / edges[:, 2].astype(float)
    edges = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])))
    edges_row = edges.tocsr()
    edges_col = edges.tocsc()
    del edges

    # dist(dest, src)
    # Due to renumbering nodes, the top target_nodes.shape[0] rows of
    # dist are the src to src distances.
    target_dist = (
        np.zeros((all_nodes.shape[0], target_nodes.shape[0]), dtype=float)
        + np.Inf
    )
    # May be able to reuse already found paths in previous iterations
    # but do that later.

    max_row = max(edges_col.indices)
    max_col = max(edges_row.indices)
    for src in range(target_nodes.shape[0]):
        dist = np.zeros((all_nodes.shape[0],), dtype=float) + np.inf
        unmarked = list(range(all_nodes.shape[0]))
        dist[src] = 0
        while len(unmarked) > 0:
            d_j = unmarked.pop(np.argmin(dist[unmarked]))
            if d_j <= max_row:
                d_potential = dist[d_j] + edges_row[d_j, :].data
                dist[edges_row[d_j, :].indices] = np.minimum(
                    dist[edges_row[d_j, :].indices], d_potential
                )

            if d_j <= max_col:
                d_potential = dist[d_j] + edges_col[:, d_j].data
                dist[edges_col[:, d_j].indices] = np.minimum(
                    dist[edges_col[:, d_j].indices], d_potential
                )

        # So self loops get removed with any edges that don't exist.
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


def _renumber(edges, target_nodes):
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
