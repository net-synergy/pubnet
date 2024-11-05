"""Implementation of the Edge class storing edges in a compressed form."""

import igraph as ig
import numpy as np
from numpy.typing import ArrayLike, NDArray

from pubnet.network._utils import edge_key

from ._base import Edge


class IgraphEdge(Edge):
    def __init__(self, *args, **keys):
        super().__init__(*args, **keys)
        self.representation = "igraph"

    def set_data(self, new_data) -> None:
        # Treating the graph as directed prevents igraph from flipping the
        # columns so source is always the data in column 1 and target
        # column 2.

        if isinstance(new_data, ig.Graph):
            self._data = new_data
        else:
            self._data = ig.Graph(new_data, directed=True)

    def __getitem__(self, key):
        row, col = self._parse_key(key)

        if isinstance(col, slice):
            start = col.start
            stop = col.stop

            if start is None:
                start = 0
            if stop is None:
                stop = 2

            col = None if start == 0 and stop == 2 else start

        if self._is_mask(row):
            row = np.arange(len(self))[row]

        if (row is None) and isinstance(col, int):
            if col == 0:
                res = (eid.source for eid in self._data.es.select())
            else:
                res = (eid.target for eid in self._data.es.select())

            return np.fromiter(res, dtype=self.dtype)

        if isinstance(row, int) and (col is not None):
            if col == 0:
                return self._data.es[row].source

            return self._data.es[row].target

        if col is not None:
            if col == 0:
                res = (eid.source for eid in self._data.es[row].select())
            else:
                res = (eid.target for eid in self._data.es[row].select())

            return np.fromiter(res, dtype=self.dtype)

        if isinstance(row, int):
            return (self._data.es[row].source, self._data.es[row].target)

        feats = {f: self.feature_vector(f)[row] for f in self.features()}
        return IgraphEdge(
            ((eid.source, eid.target) for eid in self._data.es[row].select()),
            self.name,
            self.start_id,
            self.end_id,
            self.dtype,
            features=feats,
        )

    def _is_mask(self, arr):
        if not isinstance(arr, np.ndarray):
            return False

        if not isinstance(arr[0], np.bool_):
            return False

        if arr.shape[0] != len(self):
            raise KeyError(
                "Boolean mask must have same size as edge set for indexing"
            )

        return True

    def __len__(self) -> int:
        return self._data.ecount()

    def __contains__(self, item: int) -> bool:
        try:
            node = list(self._data.vs.select(item))[0]
        except ValueError:
            return False

        return len(node.all_edges()) > 0

    def isin(
        self, column: str | int, test_elements: ArrayLike
    ) -> NDArray[np.bool_]:
        """Find which elements from column are in the set of test_elements."""
        return np.isin(
            np.fromiter(self[:, column], dtype=self.dtype), test_elements
        )

    def isequal(self, other: Edge):
        """Determine if two edge sets are equivalent."""
        return self._data.get_edgelist() == other._data.get_edgelist()

    def _to_binary(self, file_name, header_name, header):
        self._data.write_pickle(fname=file_name)
        with open(header_name, "wt") as header_file:
            header_file.write(header)

    def _to_tsv(self, file_name, header):
        np.savetxt(
            file_name,
            self.as_array(),
            delimiter="\t",
            header=header,
            comments="",
        )

    def get_edgelist(self):
        return np.asarray(self._data.get_edgelist())

    def as_igraph(self):
        return self._data.copy()

    def features(self):
        """Return a list of the edge's features names."""
        return list(self._data.es.attribute_names())

    def feature_vector(self, name):
        self._assert_has_feature(name)
        return self._data.es[name]

    def add_feature(self, feature, name):
        """Add a new feature to the edge."""
        if name in self.features():
            raise KeyError(f"{name} is already a feature.")

        self._data.es[name] = feature

    def overlap(self, node_type, weights=None):
        es = []
        ovr = []
        nodes = list(set(self[:, node_type]))
        mode = "out" if node_type == self.start_id else "in"

        for i, ni in enumerate(nodes):
            first_node_neighbors = set(self._data.neighbors(ni, mode=mode))
            for nj in nodes[i + 1 :]:
                second_node_neighbors = set(self._data.neighbors(nj))

                n_common = len(
                    first_node_neighbors.intersection(second_node_neighbors)
                )
                if n_common != 0:
                    es.append((ni, nj))
                    ovr.append(n_common)

        new_edge = ig.Graph(es, directed=False)
        new_edge.es["overlap"] = ovr
        return IgraphEdge(
            new_edge,
            edge_key(node_type, f"{self.other_node(node_type)}Overlap"),
            start_id=node_type,
            end_id=node_type,
            dtype=self.dtype,
        )
