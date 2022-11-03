"""Implementation of the Edge class storing edges in a compressed form."""


import igraph as ig
import numpy as np

from ._base import Edge


class CompressedEdge(Edge):
    def __init__(self, *args):
        super().__init__(*args)
        edge_data = self._data
        NodeID = 0
        NodeDIC = {self.start_id: {}, self.end_id: {}}
        edge_list = []
        for i in range(len(edge_data)):
            data = edge_data[i]
            if int(data[0]) not in NodeDIC[self.start_id].keys():
                NodeDIC[self.start_id][int(data[0])] = NodeID
                NodeID += 1
            if int(data[1]) not in NodeDIC[self.end_id].keys():
                NodeDIC[self.end_id][int(data[1])] = NodeID
                NodeID += 1

            edge_list.append(
                [
                    NodeDIC[self.start_id][int(data[0])],
                    NodeDIC[self.end_id][int(data[1])],
                ]
            )

        self._data = ig.Graph(n=0, edges=edge_list)
        nodes = self._data.vs
        IDdic = {self.start_id: {}, self.end_id: {}}
        IDdic[self.start_id] = dict(
            (v, k) for k, v in NodeDIC[self.start_id].items()
        )
        IDdic[self.end_id] = dict(
            (v, k) for k, v in NodeDIC[self.end_id].items()
        )
        for n in nodes:
            if n.index in IDdic[self.start_id].keys():
                n["NodeType"] = self.start_id
                n["id"] = IDdic[self.start_id][n.index]
            else:
                n["NodeType"] = self.end_id
                n["id"] = IDdic[self.end_id][n.index]

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
                edges = self._data.es
                nodes = []
                for edge in edges:
                    if (
                        self._data.vs[edge.tuple[0]]["NodeType"]
                        == self.start_id
                    ):
                        nodes.append(self._data.vs[edge.tuple[0]]["id"])
                    else:
                        nodes.append(self._data.vs[edge.tuple[1]]["id"])

                return np.asarray(nodes)
            elif key == self.end_id:
                edges = self._data.es
                nodes = []
                for edge in edges:
                    if self._data.vs[edge.tuple[0]]["NodeType"] == self.end_id:
                        nodes.append(self._data.vs[edge.tuple[0]]["id"])
                    else:
                        nodes.append(self._data.vs[edge.tuple[1]]["id"])
                return np.asarray(nodes)
            else:
                raise KeyError(
                    f'Key "{key}" not one of "{self.start_id}" or'
                    f' "{self.end_id}".'
                )

        edges = self._data.es
        new_edges = []
        for index in range(len(edges)):
            if key[index]:
                new_edges.append(edges[index])

        return self._data.subgraph_edges(edges=new_edges)

    def _read_from_file(path):
        raise NotImplementedError

    def isin(self, column, test_elements):
        """Find which elements from column are in the set of test_elements."""
        isin = []
        t_elm = set(test_elements)
        nodes = self[column]
        for node in nodes:
            if node in t_elm:
                isin.append(True)
            else:
                isin.append(False)

        return np.asarray(isin)

    def isequal(self, other):
        """Determine if two edges are equivalent."""
        nodes = self._data.vs
        other_nodes = other._data.vs
        for index in range(len(nodes)):
            if nodes[index].attributes() != other_nodes[index].attributes():
                return False
        edges = self._data.es
        other_edges = other._data.es
        for index in range(len(edges)):
            if (
                not edges[index].tuple[0] == other_edges[index].tuple[0]
                and edges[index].tuple[1] == other_edges[index].tuple[1]
                or edges[index].tuple[0] == other_edges[index].tuple[1]
                and edges[index].tuple[1] == other_edges[index].tuple[0]
            ):
                return False
        return True

    @property
    def shape(self):
        """Find number of edges."""
        return [self._data.ecount()]

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

    def similarity(self, target_publications):
        return self._data.shortest_paths(
            target_publications, target_publications
        )


def _edge_path(n1, n2, data_dir):
    """Find the edge file in data_dir for the provided node types.

    Known possible issues:
        If we need directed edges, the order of nodes in the file name
        may be important. Add in a weighted keyword argument, if true
        look for files only with the nodes in the order they were
        provided otherwise look for both. Another option is to not
        only check the file name but check the header for the START_ID
        and END_ID node types.
    """

    def edge_file_path(n1, n2):
        return os.path.join(data_dir, f"{n1}_{n2}_edges.tsv")

    if os.path.exists(edge_file_path(n1, n2)):
        file_path = edge_file_path(n1, n2)
    elif os.path.exists(edge_file_path(n2, n1)):
        file_path = edge_file_path(n2, n1)
    else:
        raise FileNotFoundError(
            f"No edge file for edges {n1}, {n2} found in"
            f" {data_dir}.\n\nExpceted either file {edge_file_path(n1, n2)} or"
            f" {edge_file_path(n2, n1)}"
        )

    return file_path
