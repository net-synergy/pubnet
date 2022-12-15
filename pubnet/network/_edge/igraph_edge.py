"""Implementation of the Edge class storing edges in a compressed form."""


import os

import igraph as ig
import numpy as np
from pubnet.data import default_data_dir

from ._base import Edge


class IgraphEdge(Edge):
    def __init__(self, *args):
        super().__init__(*args)
        self.representation = "igraph"
        if not isinstance(self._data, ig.Graph):
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

            edges = self._data.es
            for e in edges:
                if self._data.vs[e.tuple[0]]["NodeType"] == self.start_id:
                    e[self.start_id] = self._data.vs[e.tuple[0]]["id"]
                    e[self.end_id] = self._data.vs[e.tuple[1]]["id"]
                else:
                    e[self.start_id] = self._data.vs[e.tuple[1]]["id"]
                    e[self.end_id] = self._data.vs[e.tuple[0]]["id"]
            self._weighted_data = None

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
            if key == self.start_id or key == self.end_id:
                return np.asarray(self._data.es[key])
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

    def __add_weights(self, nodes):
        current_nodes = self._weighted_data.vs
        edge_list = []
        weights = []
        overlap_pubids = {}
        node_index = current_nodes[-1].index + 1

        for i in range(len(nodes)):
            if nodes[i]["id"] not in overlap_pubids.keys():
                overlap_pubids[nodes[i]["id"]] = node_index
                node_index += 1
            for j in range(i + 1, len(nodes)):
                if nodes[j]["id"] not in overlap_pubids.keys():
                    overlap_pubids[nodes[j]["id"]] = node_index
                    node_index += 1
                first_node_neighbors = self._data.neighbors(nodes[i])
                second_node_neighbors = self._data.neighbors(nodes[j])

                overlap = len(
                    set(first_node_neighbors).intersection(
                        second_node_neighbors
                    )
                )
                if overlap != 0:
                    edge_list.append(
                        [
                            overlap_pubids[nodes[i]["id"]],
                            overlap_pubids[nodes[j]["id"]],
                        ]
                    )
                    weights.append(1 / overlap)
            for k in range(len(current_nodes)):
                first_node_neighbors = self._data.neighbors(nodes[i])
                second_node_neighbors = self._data.neighbors(
                    self._data.find(
                        id_eq=nodes[k]["pubid"], NodeType_eq=self.start_id
                    )
                )

                overlap = len(
                    set(first_node_neighbors).intersection(
                        second_node_neighbors
                    )
                )

                if overlap != 0:
                    edge_list.append(
                        [
                            overlap_pubids[nodes[i]["id"]],
                            nodes[k].index,
                        ]
                    )
                    weights.append(1 / overlap)

        edge_splice = len(self._weighted_data.es)
        vertex_splice = len(self._weighted_data.vs)
        self._weighted_data.add_vertices(overlap_pubids.values())
        self._weighted_data.add_edges(edge_list)

        for i in range(len(new_edges)):
            new_edges[i]["weight"] = weights[i]

        new_nodes = self._weighted_data.vs[vertex_splice + 1 :]
        new_ids = list(overlap_pubids.keys())
        for i in range(len(new_nodes)):
            new_nodes[i]["pubid"] = new_ids[i]

    def __weights(self, nodes):
        edge_list = []
        weights = []
        overlap_pubids = {}
        node_index = 0
        for i in range(len(nodes)):
            if nodes[i]["id"] not in overlap_pubids.keys():
                overlap_pubids[nodes[i]["id"]] = node_index
                node_index += 1
            for j in range(i + 1, len(nodes)):
                if nodes[j]["id"] not in overlap_pubids.keys():
                    overlap_pubids[nodes[j]["id"]] = node_index
                    node_index += 1
                first_node_neighbors = self._data.neighbors(nodes[i])
                second_node_neighbors = self._data.neighbors(nodes[j])

                overlap = len(
                    set(first_node_neighbors).intersection(
                        second_node_neighbors
                    )
                )

                if overlap != 0:
                    edge_list.append(
                        [
                            overlap_pubids[nodes[i]["id"]],
                            overlap_pubids[nodes[j]["id"]],
                        ]
                    )
                    weights.append(1 / overlap)

        self._weighted_data = ig.Graph(edge_list)
        edges = self._weighted_data.es
        for i in range(len(edges)):
            edges[i]["weight"] = weights[i]
        nodes = self._weighted_data.vs
        ids = list(overlap_pubids.keys())
        print(ids)
        for i in range(len(nodes)):
            nodes[i]["pubid"] = ids[i]

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

    def _shortest_path(self, target_publications):
        pubids = set(target_publications)

        if self._weighted_data is None:
            nodes = self._data.vs.select(
                id_in=pubids, NodeType_eq=self.start_id
            )
            self.__weights(nodes)
            distances = self._weighted_data.distances(weights="weight")
            nodeids = nodes["id"]
        else:
            weighted_nodes = self._weighted_data.vs["pubid"]
            if pubids != set(weighted_nodes):
                nodes = self._data.vs.select(
                    id_in=pubids,
                    NodeType_eq=self.start_id,
                    id_notin=weighted_nodes["pubid"],
                )
                self.__add_weights(nodes)
                weighted_nodes = self._weighted_data.vs.select(pubid_in=pubids)
                distances = self._weighted_data.distances(
                    weighted_nodes, weighted_nodes, weights="weight"
                )
                nodeids = weighted_nodes["pubid"]
            else:
                distances = self._weighted_data.distances(weights="weight")
                nodeids = weighted_nodes

        shortest_paths = []
        for source_node_index, node_distances in enumerate(distances):
            for target_node_index, distance in enumerate(node_distances):
                if source_node_index < target_node_index:
                    shortest_paths.append(
                        [
                            nodeids[source_node_index],
                            nodeids[target_node_index],
                            distance,
                        ]
                    )
        return np.asarray(shortest_paths)
