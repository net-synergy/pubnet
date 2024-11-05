"""Implementation of the Edge class storing edges as numpy arrays."""

from typing import Any, Optional

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
        elif isinstance(new_data, ig.Graph):
            self._data = new_data.get_edgelist()
        else:
            self._data = np.asarray(new_data, self.dtype)

        if self._data.dtype != self.dtype:
            self._data = self._data.astype(self.dtype)

    def __len__(self) -> int:
        return self._data.shape[0]

    def __contains__(self, item: int) -> bool:
        return self._data.__contains__(item)

    def isin(
        self, column: str | int, test_elements: ArrayLike
    ) -> NDArray[np.bool_]:
        """Check which elements of column are members of test_elements.

        Parameters
        ----------
        column : str, int
            The column to test, can be anything accepted by `__getitem__`.
        test_elements : np.ndarray
            The elemnts to test against.

        Returns
        -------
        isin : np.ndarray
            a boolean array of the same size as self[column], such that all
            elements of self[column][isin] are in the set test_elements.

        """
        return np.isin(self[column], test_elements)

    def isequal(self, other):
        if self.start_id != other.start_id:
            return False

        if self.end_id != other.end_id:
            return False

        return (self._data == other._data).all()

    def distribution(self, column):
        return np.unique(self[column], return_counts=True)

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

    def drop_feature(self, name):
        if name in self.features():
            self._features.pop(name)

    def to_sparse_matrix(
        self,
        row: Optional[str] = None,
        column: Optional[str] = None,
        weights: Optional[str | NDArray[Any]] = None,
        shape: Optional[tuple[int, int]] = None,
    ) -> sp.spmatrix:
        """Create a csr sparse matrix from the edge data.

        Exactly one of row or column must be specified. If left blank, weights
        will be one for all non-zero elements.
        """
        edges = self._data
        data_type = edges.dtype
        if weights is None:
            _weights = np.ones((edges.shape[0]), dtype=data_type)
        elif isinstance(weights, np.ndarray):
            if weights.shape[0] != edges.shape[0]:
                raise ValueError(
                    "Weights must have the same number of rows as edges."
                )
            _weights = weights
        else:
            self._assert_has_feature(weights)
            _weights = self._features[weights]

        if row and column and row != self.other_node(column):
            raise KeyError(
                "Over-specified matrix. Provide one of row or column."
            )

        if row:
            primary, secondary = self._column_to_indices(row)
        elif column:
            secondary, primary = self._column_to_indices(column)
        else:
            raise KeyError("One of row or column must be specified")

        return sp.coo_matrix(
            (_weights, (edges[:, primary], edges[:, secondary])),
            dtype=data_type,
            shape=shape,
        ).tocsr()

    def _compose_with(self, other, counts: str, mode: str):
        shared_keys = {self.start_id, self.end_id}.intersection(
            {other.start_id, other.end_id}
        )
        if not shared_keys:
            raise AssertionError("No key edge between the two edge sets")

        if len(shared_keys) > 1:
            raise AssertionError(
                "The node types of both edges already match, can't compose."
            )

        if counts not in (
            "drop",
            "absolute",
            "normalize_self",
            "normalize_other",
        ):
            raise ValueError(counts)

        if other.isdirected:
            if mode not in ("in", "out"):
                raise KeyError(f'mode {mode} not one of "in" or "out".')
            key = "to" if mode == "in" else "from"
        else:
            key = shared_keys.pop()

        n_key = max(self[:, key].max(), other[:, key].max()) + 1
        res = self.to_sparse_matrix(
            column=key,
            shape=(self[:, self.other_node(key)].max() + 1, n_key),
        ) @ other.to_sparse_matrix(
            row=key,
            shape=(n_key, other[:, other.other_node(key)].max() + 1),
        )

        if counts == "normalize_self":
            weights = res.sum(axis=1)
            weights[weights == 0] = 1
            res = res.multiply(1 / weights)
        elif counts == "normalize_other":
            weights = res.sum(axis=0)
            weights[weights == 0] = 1
            res = res.multiply(1 / weights)

        name = edge_key(self.other_node(key), other.other_node(key))
        if other.isdirected:
            name += mode.title()

        feature_name = "counts" if counts != "drop" else None

        return self.from_sparse_matrix(
            res,
            name,
            start_id=self.other_node(key),
            end_id=other.other_node(key),
            feature_name=feature_name,
        )

    def overlap(
        self, node_type: str, weights: Optional[str | NDArray[Any]] = None
    ) -> Edge:
        """Calculate the neighbor overlap between nodes.

        For all pairs of nodes in the node_type column, calculate the number of
        nodes both are connected to.

        Parameters
        ----------
        node_type : str
            The node_type column to use. In an "Author--Publication" edge set,
            If node_type is "Author", overlap will be the number of
            publications each author has in common with every other author.
        weights : str, np.ndarray, optional
            If left None, each edge will be counted equally. Otherwise weight
            edges based on the edge's feature with the provided name or the
            array of weights. If the edge doesn't have the passed feature, an
            error will be raised.

        Returns
        -------
        overlap : Edge
            A new edge set with the same representation as self. The edges will
            have edges between all nodes with non-zero overlap and it will
            contain a feature "overlap".

        """
        data_type = self._data.dtype

        if len(self) == 0:
            res = sp.coo_matrix(np.array([]))
        else:
            adj = self.to_sparse_matrix(row=node_type, weights=weights)
            res = adj @ adj.T
            res = sp.triu(
                res - sp.diags(res.diagonal(), dtype=data_type, format="csr"),
                format="csr",
            ).tocoo()

        return self.from_sparse_matrix(
            res,
            edge_key(node_type, f"{self.other_node(node_type)}Overlap"),
            start_id=node_type,
            end_id=node_type,
            feature_name="overlap",
        )

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
            target_nodes.shape[0] nodes are the target_nodes.
            """
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

    def _duplicates_to_weights(self, weight_name: str) -> None:
        """Convert the number of occurrences of an edge to weights."""
        new_data, weights = np.unique(self._data, axis=0, return_counts=True)
        self.set_data(new_data)
        self.add_feature(weights, weight_name)

    def _reset_index(self, node: str, old_indices: np.ndarray) -> None:
        """Replace old IDs with condensed IDs for the given node.

        When filtering / modifying a network, some nodes can get dropped
        leading to gaps in the node indices. This replaces the old sparse IDs
        with a dense set IDs (i.e. without any gaps). This should not be called
        without also re-indexing the nodes.

        Parameters
        ----------
        node : str
          Which node to reindex
        old_indices : np.ndarray
          The full set of old indices. This differs from np.unique(self[node])
          if there are nodes in the node dataframe without any edges in the
          given edge set.

        """
        if len(self[node]) == 0:
            return

        if old_indices.shape[0] == 0:
            old_indices = np.unique(self[node])

        uniq = np.unique(old_indices)
        index_map = np.zeros((uniq.max() + 1,))
        for i, index in enumerate(uniq):
            index_map[index] = i

        self[node][:] = index_map[self[node]]
