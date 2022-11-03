"""Implementation of the Edge class storing edges in a compressed form."""


import igraph as ig
import numpy as np

from ._base import Edge


class CompressedEdge(Edge):
    def __init__(self, *args):
        super().__init__(*args)
        edge_data = self._data
        node_id = 0
        node_dic = {self.start_id: {}, self.end_id: {}}
        edge_list = []
        for i in range(len(edge_data)):
            data = edge_data[i]
            if int(data[0]) not in node_dic[self.start_id].keys():
                node_dic[self.start_id][int(data[0])] = node_id
                node_id += 1
            if int(data[1]) not in node_dic[self.end_id].keys():
                node_dic[self.end_id][int(data[1])] = node_id
                node_id += 1

            edge_list.append(
                [
                    node_dic[self.start_id][int(data[0])],
                    node_dic[self.end_id][int(data[1])],
                ]
            )

        self._data = ig.Graph(n=0, edges=edge_list)
        nodes = self._data.vs
        id_dic = {self.start_id: {}, self.end_id: {}}
        id_dic[self.start_id] = dict(
            (v, k) for k, v in node_dic[self.start_id].items()
        )
        id_dic[self.end_id] = dict(
            (v, k) for k, v in node_dic[self.end_id].items()
        )
        for n in nodes:
            if n.index in id_dic[self.start_id].keys():
                n["NodeType"] = self.start_id
                n["id"] = id_dic[self.start_id][n.index]
            else:
                n["NodeType"] = self.end_id
                n["id"] = id_dic[self.end_id][n.index]

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
        return [self._data.ecount(), 2]

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
