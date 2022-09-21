"""Object for storing publication data as a network. """


import copy
import os
import re
from warnings import warn

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_list_like

from ._similarity import numpy_metrics as np_similarity


class PubNet:
    """Store publication network as a graph.

    Arguments
    ----------
    nodes : a list of nodes to read in (should contain Publication).
    edges : a list of pairs of nodes to read in.
    data_dir: location of the files (default `.`).

    Expects node and edge files to conform to a standard naming
    convention and all be under the same directory (specified by
    `data_dir`).

    Node files should be of the form: f"{node_name}_nodes.tsv".
    Edge files should be of the form: f"{node_1_name}_{node_2_name}_edges.tsv".
    The order nodes are supplied for edges does not matter, it will
    look for files in both orders.

    Example
    -------
    net = PubNet(
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

    def publications_where(self, node_type, func):
        """Get a list of publications that match a condition.

        Arguments
        ---------
        node_type : str, name of the type of nodes to perform
            the search on.
        func : function, a function that accepts a
        pandas.dataframe and returns a list of indices.

        Returns
        -------
        publication_ids : array, list of publication IDs

        Example
        -------
        publication_ids = net.publications_where(
            "Author",
            lambda x: x["LastName" == "Smith"]
        )

        See also
        --------
        publications_containing
        """

        nodes = self[node_type]
        node_idx = func(nodes)

        node_ids = nodes[nodes.id][node_idx]
        publication_idx = np.isin(
            self["Publication", node_type][node_type], node_ids
        )
        publication_ids = self["Publication", node_type]["Publication"][
            publication_idx
        ]

        return np.asarray(publication_ids, dtype=np.int64)

    def publications_containing(self, node_type, node_feature, value, steps=1):
        """Get a list of publications connected to nodes with a given value.

        Parameters
        ----------
        node_type : str, name of the type of nodes to perform
            the search on.
        node_feature : str, which feature to compare.
        value : any, the value of the feature to find.
        steps : positive int, number of steps away from the original
            value. Defaults to 1, only publications with direct edges
            to the desired node(s). If `steps > 1`, includes
            publications with indirect edges up to `steps` steps
            away. For `steps == 2`, all direct publications will be
            returned as well as all publications with a node in common
            to that publication.

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
        -------
        publication_ids : array, list of publication IDs

        See also
        --------
        publications_where
        """

        assert (
            isinstance(steps, int) and steps >= 1
        ), f"Steps most be a positive integer, got {steps} instead."

        if is_list_like(value):
            func = lambda x: x[node_feature].isin(value)
        else:
            func = lambda x: x[node_feature] == value

        publication_ids = self.publications_where(node_type, func)
        while steps > 1:
            node_ids = self["Publication", node_type][node_type][
                np.isin(
                    self["Publication", node_type]["Publication"],
                    publication_ids,
                )
            ]
            func = lambda x: x[x.id].isin(node_ids)
            publication_ids = self.publications_where(node_type, func)
            steps -= 1

        return publication_ids

    def slice(self, pub_ids, mutate=False):
        """Filter all the PubNet object's edges to those connecting to pub_ids.

        Primarily called through indexing with `__getitem__`.

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
    f"{name}:ID({namespace})" in order to be compatible with Neo4j
    data.  Here the value `namespace` refers to the node so it's not
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

    def get_random(self, n=1, seed=None):
        """Sample n nodes.

        Arguments
        ---------
        n : positive int, number of nodes to sample (default 1).
        seed : positive int, random seed for reproducibility (default None).
            If None seed is select at random.

        Returns
        -------
        nodes : dataframe, subset of nodes.
        """

        rng = np.random.default_rng(seed=seed)
        return self.data.loc[rng.integers(0, self.data.shape[0], size=(n,))]


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

    @property
    def overlap(self):
        if not hasattr(self, "_overlap"):
            setattr(self, "_overlap", np_similarity.overlap(self.data))

        return self._overlap

    def shortest_path(self, target_publications, overlap):
        np_similarity.shortest_path(target_publications, overlap)

    def similarity(self, func, target_publications):
        """Calculate similarity between publications based on edge's overlap.

        Arguments
        ---------
        func : function, must take two arguments
            `func(target_publications, overlap)`. Where
            target_publications is described below and overlap is a 3
            column 2d array listing overlap (3rd column) between two
            publications (1st--2nd column).
        target_publication : array, an array of publications to return
            similarity between which must be a subset of all edges in
            `self.overlap`.

        Returns
        -------
        similarity : a 3 column 2d array, listing the similarity (3rd
        column) between all pairs of publications (1st--2nd column) in
        target_publications. Only non-zero similarities are listed.
        """

        return func(target_publications, self.overlap)
