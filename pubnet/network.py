"""Object for storing publication data as a network. """


import copy
import os
import re
from warnings import warn

import numpy as np
import pandas as pd


class PubNet:
    """Store publication network as a graph.

    Constructing:
        nodes: a list of nodes to read in (should contain Publication).
        edges: a list of pairs of nodes to read in.
        data_dir: location of the files.

    Expects node and edge files to conform to a standard naming
    convention and all be under the same directory (specified by
    `data_dir`).
    Node files should be of the form: f"{node_name}_nodes.tsv".
    Edge files should be of the form: f"{node_1_name}_{node_2_name}_edges.tsv".
    The order nodes are supplied for edges does not matter, it will
    look for files in both orders.

    Example:
        PubNet(
            ("Author", "Descriptor", "Publication"),
            (("Author" "Publication") ("Descriptor" "Publication")),
            data_dir = "./data"
        )
    """

    def __init__(self, nodes, edges, data_dir="."):
        if "Publication" not in nodes:
            warn(
                "Constructing PubNet object without Publication \
nodes. This will limit the functionality of the data type."
            )

        self._node_data = {n: Node(n, data_dir) for n in nodes}
        self._edge_data = {
            _edge_key(e[0], e[1]): Edge(e, data_dir) for e in edges
        }
        self.nodes = nodes
        self.edges = edges

    def __getitem__(self, args):
        if isinstance(args, str):
            return self._node_data[args]

        if isinstance(args, np.ndarray):
            return self.slice(args)

        if len(args) == 2:
            return self._edge_data[_edge_key(args[0], args[1])]

        raise KeyError(*args)

    def publications_where(self, node_type, cond, steps=1):
        pass

    def publications_containing(self, node_type, node_feature, value, steps=1):
        """Get a list of publications connected to nodes with a given value.

        Input arguments:
            node_type (string): Name of the type of nodes to perform
              the search on.
            node_feature (string): Which feature to compare.
            value (any): the value of the feature to find.
        Optional arguments:
            steps (positive int): number of steps away from the
              original value.
              Defaults to 1, only publications with direct edges to
              the desired node(s). If `steps > 1`, includes publications
              with indirect edges up to `steps` steps away. For `steps
              == 2`, all direct publications will be returned as well
              as all publications with a node in common to that
              publication.

              For example:
                pubnet.publications_containing(
                    "Author",
                    "LastName",
                    "Smith",
                    steps=2
                )
              Will return publications with authors that have last
              name "Smith" and publications by authors who have
              coauthored a paper with an author with last name
              "Smith".
        Returns
            Publication IDs (np.array)
        """

        assert (
            isinstance(steps, int) and steps >= 1
        ), f"Steps most be a positive integer, got {steps} instead."

        if isinstance(value, str):
            node_idx = self[node_type][node_feature] == value
        else:
            node_idx = self[node_type][node_feature].isin(value)

        node_ids = self[node_type][self[node_type].id][node_idx]
        pub_idx = np.isin(self["Publication", node_type][node_type], node_ids)
        pub_ids = self["Publication", node_type]["Publication"][pub_idx]
        return np.asarray(pub_ids, dtype=np.int64)

    def slice(self, pub_ids, mutate=False):
        """Filter all the PubNet object's edges to those connecting to pub_ids.

        If mutate is False return a new `PubNet` object otherwise
        return self after mutating the edges."""

        if not mutate:
            new_pubnet = copy.deepcopy(self)
            new_pubnet.slice(pub_ids, mutate=True)
            return new_pubnet

        for e in self.edges:
            self[e]._set(self[e][np.isin(self[e]["Publication"], pub_ids)])

        return self

    def merge(self, other, mutate=True):
        # Should handle different publication IDs somehow. Probably
        # have known IDs (PMID, DOI, etc) and use a lookup table for
        # joining. Potentially create a universal ID that is prefered
        # in this data type.
        # Probably for the best if different PubNet objects don't
        # share nodes / edges (other than Publication).
        # Intend on using this to ease generation of a PubNet that
        # combines data from multiple sources (pubmed, crossref).

        pass


def _edge_key(n1, n2):
    return "_".join(sorted((n1, n2)))


class Node:
    """Class for storing node data for PubNet class.

    Provides a wrapper around a panda dataframe adding in information
    about the ID column, which is identified by the special syntax
    f"{name}:ID({namespace})" in order to be compatible with Neo4j data.
    Here the value `namespace` refers to the node so it's not
    important since we already know the the node.
    """

    _id_re = re.compile("(.*):ID\\(.*?\\)")

    def __init__(self, node, data_dir):
        self.data = pd.read_csv(
            os.path.join(data_dir, f"{node}_nodes.tsv"),
            delimiter="\t",
        )
        id_column = list(
            filter(
                lambda x: x is not None,
                [self._id_re.search(name) for name in self.data.columns],
            )
        )[0]
        old_id = id_column.group().replace("(", "\\(").replace(")", "\\)")
        self.id = id_column.groups()[0]
        self.data.columns = self.data.columns.str.replace(old_id, self.id)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, key):
        return self.data[key]


class Edge:
    """Provides a class for storing edges for PubNet.

    For know this is a wrapper around a numpy array with two
    columns. In the future it may support weighted edges (three
    columns) and directed columns. It may also switch to/add a compressed
    graph format for performance.

    As with the Node class it expects ID columns to be in Neo4j format
    f":START_ID({namespace})" and f":END_ID({namespace})". Start and
    end will be important only if the graph is directed. The
    `namespace` value provides the name of the node and will link to
    that node's ID column.
    """

    _start_id_re = re.compile(":START_ID\\((.*?)\\)")
    _end_id_re = re.compile(":END_ID\\((.*?)\\)")

    def __init__(self, edge, data_dir):
        assert len(edge) == 2, "Edge is defined by exactly two nodes."

        def edge_file_path(n1, n2):
            return os.path.join(data_dir, f"{n1}_{n2}_edges.tsv")

        if os.path.exists(edge_file_path(edge[0], edge[1])):
            file_path = edge_file_path(edge[0], edge[1])
        elif os.path.exists(edge_file_path(edge[1], edge[0])):
            file_path = edge_file_path(edge[1], edge[0])
        else:
            raise FileNotFoundError(
                f"No edge file for edges {edge[0]}, {edge[1]} found in {data_dir}.\
\n\nExpceted either file {edge_file_path(edge[0], edge[1])} or \
{edge_file_path(edge[1], edge[0])}"
            )

        self.data = np.genfromtxt(
            file_path,
            # All edge values should be integer IDs.
            dtype=np.int64,
            skip_header=1,
        )

        with open(file_path, "r") as f:
            header_line = f.readline()

        self.start_id = self._start_id_re.search(header_line).groups()[0]
        self.end_id = self._end_id_re.search(header_line).groups()[0]

    def _set(self, new_data):
        self.data = new_data

    def __str__(self):
        return (
            f"col 0: {self.start_id}\ncol 1: {self.end_id}\n{str(self.data)}"
        )

    def __repr__(self):
        return (
            f"col 0: {self.start_id}\ncol 1: {self.end_id}\n{repr(self.data)}"
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
            return self.data[:, key]

        return self.data[key]
