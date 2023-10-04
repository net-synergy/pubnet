"""Implementation of the Edge class storing edges as numpy arrays."""

import igraph as ig
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse as sp
from scipy.stats import rankdata

from pubnet.network._utils import edge_key

from ._base import Edge


class NumpyEdge(Edge):
    """An implementation of the Edge class that stores edges as numpy arrays.

    Uses arrays to list the non-zero edges in a sparse matrix form.
    """

    def __init__(self, *args, **keys):
        self._features = None
        super().__init__(*args, **keys)

        self.representation = "numpy"

    def __getitem__(self, key):
        row, col = self._parse_key(key)

        if (row is None) and (col is not None):
            return self._data[:, col]

        if col is None:
            if isinstance(row, int):
                return self._data[row, :]

            feats = {f: self.feature_vector(f)[row] for f in self.features()}
            return NumpyEdge(
                self._data[row, :],
                self.name,
                self.start_id,
                self.end_id,
                self.dtype,
                features=feats,
            )

        return self._data[row, col]

    def set_data(self, new_data):
        if isinstance(new_data, np.ndarray):
            self._data = new_data
        else:
            self._data = np.asarray(new_data, self.dtype)

    def __len__(self) -> int:
        return self._data.shape[0]

    def __contains__(self, item: int) -> bool:
        return self._data.__contains__(item)

    def isin(
        self, column: str | int, test_elements: ArrayLike
    ) -> NDArray[np.bool_]:
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

    def isequal(self, other):
        if self.start_id != other.start_id:
            return False

        if self.end_id != other.end_id:
            return False

        return (self._data == other._data).all()

    def distribution(self, column):
        dist = np.bincount(self[column])
        # Because id's start at 1 but the 0th value in the distribution is
        # reserved for id == 0.
        return dist[1:]

    def _to_binary(self, file_name, header_name, header):
        np.save(file_name, self._data)
        with open(header_name, "wt") as header_file:
            header_file.write(header)

    def _to_tsv(self, file_name, header):
        # NOTE: IDs should be ints so select integer fmt string but this will
        # need modification if we add weighted edges as the weight column(s)
        # are likely going to be floats.
        np.savetxt(
            file_name,
            self._data,
            fmt="%d",
            delimiter="\t",
            header=header,
            comments="",
        )

    def get_edgelist(self):
        return self._data.copy()

    def as_igraph(self):
        g = ig.Graph(self._data)
        for feat in self.features():
            g.es[feat] = self.feature_vector(feat)

        return g

    def features(self):
        """Return a list of the edge's features names."""
        if self._features is None:
            return []

        return list(self._features.keys())

    def feature_vector(self, name):
        self._assert_has_feature(name)
        return self._features[name]

    def add_feature(self, feature, name):
        """Add a new feature to the edge."""
        if name in self.features():
            raise KeyError(f"{name} is already a feature.")

        if self._features is None:
            self._features = {name: feature}
        else:
            self._features[name] = feature

    def overlap(self, id, weights=None):
        """Calculate the neighbor overlap between nodes.

        For all pairs of nodes in the id column, calculate the number of nodes
        both are connected to.

        Parameters
        ----------
        id : str
            The id column to use. In an "Author--Publication" edge set, If id
            is "Author", overlap will be the number of publications each author
            has in common with every other author.
        weights : str, optional
            If left None, each edge will be counted equally. Otherwise weight
            edges based on the edge's feature with the provided name. If the
            edge doesn't have the passed feature, an error will be raised.

        Returns
        -------
        overlap : Edge
            A new edge set with the same representation as self. The edges will
            have edges between all nodes with non-zero overlap and it will
            contain a feature "overlap".
        """

        edges = self._data
        data_type = edges.dtype
        if weights is None:
            _weights = np.ones((edges.shape[0]), dtype=data_type)
        else:
            self._assert_has_feature(weights)
            _weights = self._features[weights]

        primary, secondary = self._column_to_indices(id)
        adj = sp.coo_matrix(
            (_weights, (edges[:, primary], edges[:, secondary])),
            dtype=data_type,
        ).tocsr()

        res = adj @ adj.T

        res = sp.triu(
            res - sp.diags(res.diagonal(), dtype=data_type, format="csr"),
            format="csr",
        ).tocoo()

        new_edge = NumpyEdge(
            np.stack((res.row, res.col), axis=1),
            edge_key(id, "Overlap"),
            start_id=id,
            end_id=id,
            dtype=self.dtype,
        )

        new_edge.add_feature(res.data, "overlap")
        return new_edge

    def _shortest_path(self, target_nodes):
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
            np.concatenate((self.overlap[:, 0:2].flatten(), target_nodes))
        )

        overlap, target_nodes = renumber(self.overlap, target_nodes)

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
