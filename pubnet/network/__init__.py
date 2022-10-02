"""Object for storing publication data as a network."""

import copy
from functools import reduce
from warnings import warn

import numpy as np
from pandas.core.dtypes.common import is_list_like

from pubnet.network import _edge, _node


class PubNet:
    """Store publication network as a graph.

    Arguments
    ----------
    nodes : a list of nodes to read in (should contain Publication).
    edges : a list of pairs of nodes to read in.
    data_dir : location of the files (default `.`).
    compressed : bool, whether to compress edges (default False).

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

    def __init__(self, nodes, edges, data_dir=".", compressed=False):
        if nodes is None:
            nodes = ()
        if edges is None:
            edges = ()

        if "Publication" not in nodes:
            warn(
                "Constructing PubNet object without Publication nodes. "
                "This will limit the functionality of the data type."
            )

        if compressed:
            Edge = _edge.CompressedEdge
        else:
            Edge = _edge.NumpyEdge

        self._edge_data = {
            _edge_key(e[0], e[1]): Edge(e, data_dir) for e in edges
        }
        self._node_data = {n: _node.Node(n, data_dir) for n in nodes}

        nodes_in_edges = reduce(lambda a, b: a + b, edges, ())
        missing_nodes = tuple(set(nodes_in_edges).difference(set(nodes)))
        nodes = nodes + missing_nodes

        missing_nodes = {n: _node.Node(None) for n in missing_nodes}
        self._node_data.update(missing_nodes)

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

    def __repr__(self):
        res = "PubNet"
        res += "\nNodes (number of nodes)"
        for n in self.nodes:
            res += f"\n\t{n}\t({self._node_data[n].shape[0]})"
        res += "\n\n Edges (number of edges)"
        for e in self._edge_data.values():
            res += f"\n\t{e.start_id}-{e.end_id}\t({e.shape})"

        return res

    def add_node(self):
        raise NotImplementedError

    def add_edge(self):
        raise NotImplementedError

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
        publication_idx = self["Publication", node_type].isin(
            node_type, node_ids
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
                self["Publication", node_type].isin(
                    "Publication", publication_ids
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
            self[e].set(self[e][self[e].isin("Publication", pub_ids)])

        for n in self.nodes:
            if len(n) == 0:
                continue
            try:
                edge = self[n, "Publication"]
                node_ids = edge[n][edge.isin("Publication", pub_ids)]
                self[n].set(self[n][self[n][self[n].id].isin(node_ids)])

            except KeyError:
                continue

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
    """Generate a dictionary key for the given pair of nodes."""
    return "_".join(sorted((n1, n2)))


def from_dir(data_dir, *args):
    """Collect all node and edge files in data_dir and use them to
    make a PubNet object."""

    raise NotImplementedError


def from_nodes(nodes, *args):
    """Make PubNet object from given nodes.

    Collects all existing edge files between nodes in argument nodes.
    """

    raise NotImplementedError


def from_edges(edges, *args):
    """Make PubNet object from given edges.

    Collects all existing node files for nodes in list of edges.
    """

    raise NotImplementedError
