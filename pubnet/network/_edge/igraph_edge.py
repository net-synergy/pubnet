"""Implementation of the Edge class storing edges in a compressed form."""


import os

import igraph as ig
import numpy as np
from numpy.typing import ArrayLike, NDArray

from pubnet.data import default_data_dir

from ._base import Edge


class IgraphEdge(Edge):
    def __init__(self, *args):
        super().__init__(*args)
        self.representation = "igraph"

    def set(self, new_data) -> None:
        # Treating the graph as directed prevents igraph from flipping the
        # columns so source is always the data in column 1 and target
        # column 2.

        if isinstance(new_data, ig.Graph):
            self._data = new_data
        else:
            self._data = ig.Graph(new_data, directed=True)

    def __getitem__(self, key):
        row, col = self._parse_key(key)

        if (row is None) and (col is not None):
            if col == 0:
                return (eid.source for eid in self._data.es.select())
            else:
                return (eid.target for eid in self._data.es.select())

        if isinstance(row, int) and (col is not None):
            if col == 0:
                return self._data.es[row].source
            else:
                return self._data.es[row].target

        if col is not None:
            if col == 0:
                return (eid.source for eid in self._data.es[row].select())
            else:
                return (eid.target for eid in self._data.es[row].select())

        if isinstance(row, int):
            return self._data.subgraph_edges(row)

        return self._data.subgraph_edges(self._data.es[row].select())

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

    def to_file(
        self, edge_name, graph_name, data_dir=default_data_dir(), format="tsv"
    ):
        """Save the edge to disk.

        Arguments
        ---------
        edge_name : str, the name of the edge.
        graph_name : str, directory under `data_dir` to store the graph's
            files.
        data_dir : str, where to store graphs (default `default_data_dir`)
        format : str {"tsv", "gzip", "binary"}, how to store the edge (default
            "tsv"). Binary uses numpy's npy format.

        Returns
        -------
        None

        See also
        --------
        `pubnet.data.default_data_dir`
        `pubnet.network.PubNet.to_dir`
        `pubnet.network.from_dir`
        """

        ext = {"binary": "ig", "gzip": "tsv.gz", "tsv": "tsv"}
        data_dir = os.path.join(data_dir, graph_name)

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if isinstance(edge_name, tuple):
            n1, n2 = edge_name[:2]
        else:
            n1, n2 = edge_name.split("-")

        file_name = os.path.join(data_dir, f"{n1}_{n2}_edges.{ext[format]}")
        header_name = os.path.join(data_dir, f"{n1}_{n2}_edge_header.tsv")
        header = f":START_ID({self.start_id})\t:END_ID({self.end_id})"

        if format == "binary":
            self._to_binary(file_name, header_name, header)
        else:
            # `np.savetxt` handles "gz" extensions so nothing extra to do.
            self._to_tsv(file_name, header)

    def _to_binary(self, file_name, header_name, header):
        self._data.write_pickle(fname=file_name)
        with open(header_name, "wt") as header_file:
            header_file.write(header)

    def _to_tsv(self, file_name, header):
        # NOTE: IDs should be ints so select integer fmt string but this will
        # need modification if we add weigthed edges as the weight column(s)
        # are likely going to be floats.
        np.savetxt(
            file_name,
            np.column_stack(
                (self._data.es[self.start_id], self._data.es[self.end_id])
            ),
            fmt="%d",
            delimiter="\t",
            header=header,
            comments="",
        )

    @property
    def overlap(self):
        overlaps = []
        nodes = self._data.vs.select(NodeType_eq=self.start_id)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                first_node_neighbors = self._data.neighbors(nodes[i])
                second_node_neighbors = self._data.neighbors(nodes[j])

                overlap = len(
                    set(first_node_neighbors).intersection(
                        second_node_neighbors
                    )
                )
                if overlap != 0:
                    overlaps.append([nodes[i]["id"], nodes[j]["id"], overlap])

        return np.asarray(overlaps)
