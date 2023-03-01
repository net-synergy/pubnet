"""Object for storing publication data as a network.

Components
----------
A graph is made up of a list of node and a list of edges.
"""

import copy
import os
import re
from functools import reduce
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from pandas.core.dtypes.common import is_list_like
from pubnet import data
from pubnet.data import default_data_dir
from pubnet.network import _edge, _node
from pubnet.network._edge._base import Edge
from pubnet.network._node import Node

__all__ = ["from_dir", "from_data", "edge_key", "PubNet", "Edge", "Node"]

EDGE_KEY_DELIM = "-"


class PubNet:
    """
    Store publication network as a set of graphs.

    Parameters
    ----------
    root : str, default "Publication"
        The root of the network. This is used by functions that filter the
        network. (Note: names are case-sensitive)
    nodes : list-like, optional
        The nodes to include in the network.
    edges : list-like, optional
        The edges to include in the network.

    Attributes
    ----------
    nodes : list
        Names of nodes in the network, both from nodes argument and edges. If
        an edge has a node type not provided, a placeholder node of shape (0,0)
        will be added to the node list.
    edges : list
        nodes.
    id_dtype: Datatype
        Datatype used to store id values (edge data).

    Notes
    -----
    Use `from_dir` to construct a PubNet object instead of initializing
    directly.

    See also
    --------
    `from_dir`
    `from_data`
    """

    def __init__(self, nodes=None, edges=None, root="Publication"):
        self.root = root

        if nodes is None:
            nodes = {}
        if edges is None:
            edges = {}

        self._node_data = {}
        self._edge_data = {}
        self.nodes = []
        self.edges = []

        for name, data in nodes.items():
            self.add_node(name, data)

        for name, data in edges.items():
            self.add_edge(name, data)

        edge_nodes = reduce(lambda a, n: a + edge_parts(n), self.edges, [])
        missing_nodes = filter(lambda n: n not in self.nodes, edge_nodes)

        for name in missing_nodes:
            self.add_node(name, None)

        if self.root not in nodes:
            warn(
                f"Constructing PubNet object without {self.root} nodes. "
                "This will limit the functionality of the data type."
            )

        self.id_dtype = _edge.id_dtype

    def add_node(self, name, data):
        """
        Add a new node to the network.

        Parameters
        ----------
        name : str
            Name of the node.
        data : str
            Node, or pandas.DataFrame, the data this can be in the form of a
            file path, a DataFrame or an already constructed Node.

        See also
        --------
        `PubNet.add_edge`
        `PubNet.drop`
        """

        if name in self.nodes:
            raise ValueError(f"The node type {name} is already in network.")

        self.nodes.append(name)
        if isinstance(data, str):
            data = _node.from_file(data)
        elif data is None or not isinstance(data, _node.Node):
            data = _node.from_data(data)

        self._node_data[name] = data

    def add_edge(self, name, data, representation="numpy", **keys):
        """
        Add a new edge set to the network.

        Parameters
        ----------
        name : str, tuple
            Name of the node pair (see `edge_key` for generating
            the name). If tuple, should be a pair of node types.
        data : str, Edge, np.ndarray
            The data in the form of a file path, an array or an already
            constructed edge.
        representation : {"numpy", "igraph"}, default "numpy"
            The backend representation used for storing the edge.
        start_id : str, optional
            The name of the "from" node.
        end_id : str, optional
            The name of the "to" node.

        `start_id` and `end_id` are only needed if `data` is an np.ndarray.

        See also
        --------
        `PubNet.add_node`
        `PubNet.drop`
        """

        if isinstance(name, tuple):
            name = edge_key(*name)

        if name in self.edges:
            raise ValueError(f"The edge {name} is already in the network.")

        self.edges.append(name)
        if isinstance(data, str):
            data = _edge.from_file(data, representation)
        elif not isinstance(data, _edge.Edge):
            data = _edge.from_data(data, **keys, representation=representation)

        self._edge_data[name] = data

    def __getitem__(self, args):
        if isinstance(args, str):
            if args in self.nodes:
                return self._node_data[args]
            elif args in self.edges:
                return self._edge_data[args]
            else:
                raise KeyError(args)

        is_string_array = isinstance(args, np.ndarray) and isinstance(
            args[0], str
        )
        if (is_string_array or isinstance(args, tuple)) and (len(args) == 2):
            return self._edge_data[edge_key(*args)]

        if isinstance(args, np.ndarray):
            return self._slice(args)

        if isinstance(args, (self.id_dtype, int)):
            return self._slice(np.asarray([args]))

        raise KeyError(*args)

    def _slice(self, root_ids, mutate=False):
        """
        Filter all the PubNet object's edges to those connecting to root_ids.

        If mutate is False return a new `PubNet` object otherwise
        return self after mutating the edges."""

        if not mutate:
            new_pubnet = copy.deepcopy(self)
            new_pubnet._slice(root_ids, mutate=True)
            return new_pubnet

        for key in self.edges:
            self[key].set(self[key][self[key].isin(self.root, root_ids)])

        for key in self.nodes:
            if len(self[key]) == 0:
                continue

            if key == self.root:
                node_ids = root_ids
            else:
                try:
                    edge = self[key, self.root]
                except KeyError:
                    continue
                node_ids = edge[key]

            node_locs = self[key][self[key].id].isin(node_ids)
            self[key].set(self[key][node_locs])

        return self

    def __repr__(self):
        res = "PubNet"
        res += "\nNodes (number of nodes)"
        for n in self.nodes:
            res += f"\n\t{n}\t({self._node_data[n].shape[0]})"
        res += "\n\nEdges (number of edges)"
        for e in self.edges:
            res += f"\n\t{e}\t({self._edge_data[e].shape[0]})"

        return res

    def ids_where(self, node_type, func):
        """
        Get a list of the root node's IDs that match a condition.

        Parameters
        ----------
        node_type : str
            Name of the type of nodes to perform the search on.
        func : function
            A function that accepts a pandas.dataframe and returns a list of
            indices.

        Returns
        -------
        root_ids : ndarray
            List of root IDs.

        Examples
        --------
        >>> net = from_dir(graph_name="author_net", root="Publication")
        >>> publication_ids = net.ids_where(
        ...     "Author",
        ...     lambda x: x["LastName" == "Smith"]
        ... )

        See also
        --------
        `PubNet.ids_containing`
        """

        nodes = self[node_type]
        node_idx = func(nodes)

        node_ids = nodes[nodes.id][node_idx]
        root_idx = self[self.root, node_type].isin(node_type, node_ids)

        root_ids = self[self.root, node_type][self.root][root_idx]

        return np.asarray(root_ids, dtype=np.int64)

    def ids_containing(self, node_type, node_feature, value, steps=1):
        """
        Get a list of root IDs connected to nodes with a given value.

        Root IDs is based on the root of the PubNet.

        Parameters
        ----------
        node_type : str
            Name of the type of nodes to perform the search on.
        node_feature : str
            Which feature to compare.
        value : any
            The value of the feature to find.
        steps : positive int, default 1
            Number of steps away from the original value. Defaults to 1, only
            publications with direct edges to the desired node(s). If steps >
            1, includes publications with indirect edges up to `steps` steps
            away. For `steps == 2`, all direct publications will be returned as
            well as all publications with a node in common to that publication.

            For example:
            `>>> pubnet.ids_containing("Author", "LastName", "Smith", steps=2)`

            Will return publications with authors that have last name "Smith"
            and publications by authors who have coauthored a paper with an
            author with last name "Smith".

        Returns
        -------
        root_ids : ndarray
            List of publication IDs.

        See also
        --------
        `PubNet.ids_where`
        """

        assert (
            isinstance(steps, int) and steps >= 1
        ), f"Steps most be a positive integer, got {steps} instead."

        if is_list_like(value):
            func = lambda x: x[node_feature].isin(value)
        else:
            func = lambda x: x[node_feature] == value

        root_ids = self.ids_where(node_type, func)
        while steps > 1:
            node_ids = self[self.root, node_type][node_type][
                self[self.root, node_type].isin(self.root, root_ids)
            ]
            func = lambda x: x[x.id].isin(node_ids)
            root_ids = self.ids_where(node_type, func)
            steps -= 1

        return root_ids

    def where(self, node_type, func):
        """
        Filter network to root nodes satisfying a predicate function.

        All graphs are reduced to a subset of edges related to those associated
        with the root nodes that satisfy the predicate function.

        Returns
        -------
        subnet : PubNet
            A new PubNet object that is subset of the original.

        See also
        --------
        `PubNet.ids_where`
        `PubNet.containing`
        """

        root_ids = self.ids_where(node_type, func)
        return self[root_ids]

    def containing(self, node_type, node_feature, value, steps=1):
        """
        Filter network to root nodes with a given node feature.

        See also
        --------
        `PubNet.ids_containing`
        `PubNet.where`
        """

        root_ids = self.ids_containing(node_type, node_feature, value, steps)
        return self[root_ids]

    def plot_distribution(
        self, node_type, node_feature, threshold=1, fname=None
    ):
        """
        Plot the distribution of the values of a node's feature.

        Parameters
        ----------
        node_type : str
            Name of the node type to use.
        node_feature : str
            Name of one of `node_type`'s features.
        threshold : int, optional
            Minimum number of occurrences for a value to be included. In case
            there are a lot of possible values, threshold reduces the which
            values will be plotted to only the common values.
        fname : str, optional
            The name of the figure.
        """

        distribution = self[self.root, node_type].distribution(node_type)
        names = self[node_type][node_feature].to_numpy()

        retain = distribution >= threshold
        distribution = distribution[retain]
        names = names[retain]

        indices = np.argsort(distribution)
        indices = indices[-1::-1]

        fig, ax = plt.subplots()
        ax.bar(
            np.take_along_axis(
                names,
                indices,
                axis=0,
            ),
            np.take_along_axis(distribution, indices, axis=0),
        )
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        ax.set_xlabel(node_feature)
        ax.set_ylabel(f"{self.root} occurance")

        if fname:
            plt.savefig(fname)
        else:
            plt.show()

    def drop(self, nodes=None, edges=None):
        """Drop given nodes and edges from the network.

        Parameters
        ----------
        nodes : str or tuple of str, optional
            Drop the provided nodes.
        edges : tuple of tuples of str, optional
            Drop the provided edges.

        See also
        --------
        `PubNet.add_node`
        `PubNet.add_edge`
        """

        assert len(self._missing_nodes(nodes)) == 0, (
            f"Node(s) {self._missing_nodes(nodes)} is not in network",
            "\n\nNetwork's nodes are {self.nodes}.",
        )

        assert len(self._missing_edges(edges)) == 0, (
            f"Edge(s) {self._missing_edges(edges)} is not in network",
            "\n\nNetwork's edges are {self.edges}.",
        )

        if nodes is None:
            nodes = []
        elif isinstance(nodes, str):
            nodes = [nodes]

        for node in nodes:
            self._node_data.pop(node)

        self.nodes = list(filter(lambda n: n not in nodes, self.nodes))

        if edges is None:
            edges = []
        else:
            edges = self._as_keys(edges)

        for edge in edges:
            self._edge_data.pop(edge)

        self.edges = list(filter(lambda e: e not in edges, self.edges))

    def update(self, other):
        """
        Add the data from other to the current network.

        Behaves similar to Dict.update(), if other contains nodes or edges in
        this network, the values in other will replace this network's.

        This command mutates the current network and returns nothing.
        """

        self._node_data.update(other._node_data)
        self._edge_data.update(other._edge_data)
        self.nodes = list(set(self.nodes + other.nodes))
        self.edges += list(set(self.edges + other.edges))

    def merge(self, other, mutate=True):
        # Should handle different publication IDs somehow. Probably
        # have known IDs (PMID, DOI, etc) and use a lookup table for
        # joining. Potentially create a universal ID that is prefered
        # in this data type.
        # Probably for the best if different PubNet objects don't
        # share nodes / edges (other than Publication).
        # Intend on using this to ease generation of a PubNet that
        # combines data from multiple sources (pubmed, crossref).

        raise NotImplementedError

    def isequal(self, other):
        """Compare if two PubNet objects are equivalent."""

        if set(self.nodes).symmetric_difference(set(other.nodes)):
            return False

        if set(self.edges).symmetric_difference(set(other.edges)):
            return False

        for n in self.nodes:
            if not self[n].isequal(other[n]):
                return False

        for e in self.edges:
            if not self[e].isequal(other[e]):
                return False

        return True

    def _as_keys(self, edges):
        """Convert a list of edges to their keys."""

        try:
            if isinstance(edges[0], str):
                edges = [edges]
        except IndexError:
            return None

        return [edge_key(*e) for e in edges]

    def _missing_edges(self, edges):
        """
        Find all edges not in self.

        Parameters
        ----------
        edges : list-like, optional

        Returns
        -------
        missing_edges : list
            Edges not in self.
        """

        if edges is None:
            return []

        if isinstance(edges[0], str):
            edges = [edges]

        return list(
            filter(lambda key: key not in self.edges, self._as_keys(edges))
        )

    def _missing_nodes(self, nodes):
        """
        Find all node names in a list not in self.nodes.

        Parameters
        ----------
        nodes : str or list-like of str, optional
            List of names to test.

        Returns
        -------
        missing_nodes : list
            Nodes not in self.
        """

        if nodes is None:
            return []

        if isinstance(nodes, str):
            nodes = [nodes]

        return list(filter(lambda key: key not in self.nodes, nodes))

    def to_dir(
        self,
        graph_name,
        nodes="all",
        edges="all",
        data_dir=default_data_dir(),
        format="tsv",
        overwrite=False,
    ):
        """
        Save a graph to disk.

        Parameters
        ----------
        graph_name : str
            What to name the graph (the directory under `data_dir` to store
            files.).
        nodes : tuple or "all", default "all"
            A list of nodes to save. If "all", see notes.
        edges : tuple or "all", default "all"
            A list of edges to save. If "all", see notes.
        data_dir : str, default `default_data_dir`
            Location to save the graph.
        format : {"tsv", "gzip", "binary"}, default "tsv"
            How to store the files.
        overwrite : bool, default False
            If true delete the current graph on disk. This may be useful for
            replacing a plain text representation with a binary represention if
            storage is a concern. WARNING: This can lose data if the self does
            not contain all the nodes/edges that are in the saved graph. Tries
            to perform the deletion as late as possible to prevent errors from
            erasing data without replacing it, but it may be safer to save the
            data to a new location then delete the graph (with
            `pubnet.data.delete`) after confirming the save worked correctly.

        Notes
        -----
        If nodes and edges are both "all" store the entire graph. If nodes is
        "all" and edges is a tuple, save all nodes in the list of
        edges. Similarly, if edges is "all" and nodes is a tuple, save all
        edges where both the start and end nodes are in the node list.

        See also
        --------
        `pubnet.data.default_data_dir`
        `from_dir`
        """

        def all_edges_containing(nodes):
            edges = set()
            for e in self.edges:
                n1, n2 = edge_parts(e)
                if (n1 in nodes) or (n2 in nodes):
                    edges.add(e)

            return tuple(edges)

        def all_nodes_in(edges):
            nodes = set()
            for e in edges:
                for n in edge_parts(e):
                    if n in self.nodes:
                        nodes.add(n)

            return tuple(nodes)

        if (nodes == "all") and (edges == "all"):
            nodes = self.nodes
            edges = self.edges
        elif (nodes == "all") and (edges is None):
            nodes = self.nodes
        elif (edges == "all") and (nodes is None):
            edges = self.edges
        elif nodes == "all":
            nodes = all_nodes_in(edges)
        elif edges == "all":
            edges = all_edges_containing(nodes)

        if nodes is None:
            nodes = []
        if edges is None:
            edges = []

        nodes = [n for n in nodes if self[n].shape[0] > 0]
        edges = [e for e in edges if self[e].shape[0] > 0]

        if overwrite:
            data.delete(graph_name, data_dir)

        for n in nodes:
            self[n].to_file(n, graph_name, data_dir=data_dir, format=format)

        for e in edges:
            self[e].to_file(e, graph_name, data_dir=data_dir, format=format)


def from_dir(
    graph_name=None,
    nodes="all",
    edges="all",
    root="Publication",
    data_dir=default_data_dir(),
    representation="numpy",
):
    """
    Collect all node and edge files in data_dir and use them to make a
    PubNet object.

    See `PubNet` for more information about parameters.

    Parameters
    ----------
    graph_name : str, optional
       Name of the graph. If not provided, assume files are directly under
       `data_dir`.
    nodes : touple or "all", (default "all")
       A list of nodes to read in.
    edges : touple or "all", (default "all")
       A list of pairs of nodes to read in.
    root : str, default "Publication
       The root node.
    data_dir : str,  default `default_data_dir`
       Location of the files.
    representation : {"numpy", "igraph"}, default "numpy"
       Which edge backend representation to use.

    Returns
    -------
    A PubNet object.

    Notes
    -----
    Node files are exepected to be in the form f"{node_name}_nodes.tsv" and
    edge files should be of the form f"{node_1_name}_{node_2_name}_edges.tsv".
    The order nodes are supplied for edges does not matter, it will look for
    files in both orders.

    If nodes or edges is "all" it will look for all files in the directory that
    match the above file patterns. When one is "all" but the other is a list,
    it will only look for files containing the provided nodes. For example, if
    nodes = ("Author", "Publication", "Chemical") and edges = "all", it will
    only look for edges between those nodes and would ignore files such as
    "Publication_Desrciptor_edges.tsv".

    Graph name is the name of the directory the graph specific files are found
    in. It is added to the end of the `data_dir`, so it is equivalent to
    passing `os.path.join(data_dir, graph_name)` for `data_dir`, the reason to
    seperate them is to easily store multiple seperate graphs in the
    `defalut_data_dir` by only passing a `graph_name` and leaving `data_dir` as
    default.

    Examples
    --------
    >>> net = pubnet.from_dir(
    ...     "author_net"
    ...     ("Author", "Publication"),
    ...     (("Author", "Publication"), ("Publication", "Chemical")),
    ... )

    See also
    --------
    `pubnet.network.PubNet`
    `pubnet.data.default_data_dir`
    `from_data`
    """

    def node_files_containing(nodes):
        all_node_files = _node_files(data_dir)
        if nodes == "all":
            nodes = all_node_files.keys()

        return {n: _node_file_path(n, all_node_files) for n in nodes}

    def edge_files_containing(nodes):
        all_edge_files = _edge_files(data_dir)
        if nodes == "all":
            edges = all_edge_files.keys()
        else:
            edges = (
                edge_key(n1, n2)
                for i, n1 in enumerate(nodes)
                for n2 in nodes[i:]
                if edge_key(n1, n2) in all_edge_files.keys()
            )

        return {
            e: _edge_file_path(*edge_parts(e), all_edge_files) for e in edges
        }

    if nodes is None:
        nodes = ()

    if edges is None:
        edges = ()

    assert isinstance(
        nodes, (str, tuple)
    ), "Nodes must be a string or a tuple."

    assert isinstance(edges, (str, tuple)), 'Edges must be a tuple or "all".'

    if graph_name is not None:
        data_dir = os.path.join(data_dir, graph_name)

    node_files = {}
    edge_files = {}
    if (nodes == "all") and (edges == "all"):
        node_files = node_files_containing("all")
        edge_files = edge_files_containing("all")
    elif nodes == "all":
        edge_nodes = set(reduce(lambda a, b: a + b, edges, ()))
        node_files = node_files_containing(edge_nodes)
        for node_pair in edges:
            edge_files[edge_key(*node_pair)] = _edge_file_path(
                *node_pair, data_dir
            )
    elif edges == "all":
        for node in nodes:
            node_files[node] = _node_file_path(node, data_dir)
        edge_files = edge_files_containing(nodes)
    else:
        for node in nodes:
            node_files[node] = _node_file_path(node, data_dir)

        for node_pair in edges:
            edge_files[edge_key(*node_pair)] = _edge_file_path(
                *node_pair, data_dir
            )

    nodes = {}
    edges = {}
    for name, file in node_files.items():
        nodes[name] = _node.from_file(file)

    for name, file in edge_files.items():
        edges[name] = _edge.from_file(file, representation)

    return PubNet(root=root, nodes=nodes, edges=edges)


def from_data(
    nodes=None, edges=None, root="Publication", representation="numpy"
):
    """
    Make PubNet object from given nodes and edges.

    Parameters
    ----------
    nodes : Dict, optional
        A dictionary of node data of the form {name: DataFrame}.
    edges : Dict, optional
        A dictionary of edge data of the form {name: Array}.
    root : str, default "Publication"
        Root node.
    representation : {"numpy", "igraph"}, default "numpy"
       The edge representation.

    Returns
    -------
    A PubNet object

    See Also
    --------
    `from_dir`
    """

    for name, data in nodes:
        nodes[name] = _node.from_data(data)

    for name, data in edges:
        start_id, end_id = edge_parts(name)
        edges[name] = _edge.from_data(data, start_id, end_id, representation)

    return PubNet(root=root, nodes=nodes, edges=edges)


def edge_key(node_1, node_2):
    """
    Generate a dictionary key for the given pair of nodes.

    Known future issue:
        If we need directed edges, the order of nodes in the file name
        may be important. Add in a weighted keyword argument, if true
        look for files only with the nodes in the order they were
        provided otherwise look for both. Another option is to not
        only check the file name but check the header for the START_ID
        and END_ID node types.
    """

    return EDGE_KEY_DELIM.join(sorted((node_1, node_2)))


def edge_parts(key):
    """Break an edge key into its nodes"""
    return key.split(EDGE_KEY_DELIM)


def _node_files(data_dir):
    """Return all node files in the data_dir"""

    files = os.listdir(data_dir)
    path_regex = r"(?P<node>\w+)_nodes.(?P<ext>[\w\.]+)"
    out = {}
    for f in files:
        m = re.match(path_regex, f)
        try:
            out[m.groups()[0]][m.groups()[1]] = os.path.join(
                data_dir, m.group()
            )
        except KeyError:
            out[m.groups()[0]] = {
                m.groups()[1]: os.path.join(data_dir, m.group())
            }
        except AttributeError:  # File didn't match regex
            continue
    return out


def _node_file_path(name, data_dir):
    """Return the file path for a node."""

    if isinstance(data_dir, dict):
        node_files = data_dir
    elif isinstance(data_dir, str) and os.path.isdir(data_dir):
        node_files = _node_files(data_dir)
    else:
        raise TypeError(
            "Second argument must be a dictionary of files or directory."
        )

    try:
        available_files = node_files[name]
    except KeyError:
        raise FileNotFoundError(f"No file found for node {name}.")

    ext_preference = ["feather", "tsv", "tsv.gz"]
    for ext in ext_preference:
        try:
            return available_files[ext]
        except KeyError:
            continue

    raise FileNotFoundError(
        f"No file found for node {name} with a supported file extension."
    )


def _edge_files(data_dir):
    """Return all edge files in the data_dir"""

    files = os.listdir(data_dir)
    path_regex = r"(?P<n1>\w+)_(?P<n2>\w+)_edges.(?P<ext>[\w\.]+)"
    out = {}
    for f in files:
        m = re.match(path_regex, f)
        try:
            out[edge_key(*m.groups()[:2])][m.groups()[2]] = os.path.join(
                data_dir, m.group()
            )
        except KeyError:
            out[edge_key(*m.groups()[:2])] = {
                m.groups()[2]: os.path.join(data_dir, m.group())
            }
        except AttributeError:  # File didn't match regex
            continue
    return out


def _edge_file_path(node_1, node_2, data_dir):
    """Find the edge file in data_dir for the provided node types."""

    if isinstance(data_dir, dict):
        edge_files = data_dir
    elif isinstance(data_dir, str) and os.path.isdir(data_dir):
        edge_files = _edge_files(data_dir)
    else:
        raise TypeError(
            "Second argument must be a dictionary of files or directory."
        )

    name = edge_key(node_1, node_2)
    try:
        available_files = edge_files[name]
    except KeyError:
        raise FileNotFoundError(f"No file found for nodes {node_1}, {node_2}.")

    ext_preference = ["npy", "tsv", "tsv.gz", "ig"]
    for ext in ext_preference:
        try:
            return available_files[ext]
        except KeyError:
            continue

    raise FileNotFoundError(
        f"No file found for nodes {node_1}, {node_2} with a supported file"
        " extension."
    )
