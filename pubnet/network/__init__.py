"""Object for storing publication data as a network."""

import copy
import os
import re
from functools import reduce
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from pandas.core.dtypes.common import is_list_like
from pubnet.data import default_data_dir
from pubnet.network import _edge, _node

__all__ = ["from_dir", "from_data", "edge_key"]


class PubNet:
    """Store publication network as a graph.

    Attributes
    ----------
    root : str, the root of the network. This is used by function that filter
        the network.
    nodes : list, names of nodes in the network, both from nodes argument and
        edges. If an edge has a node type not provided, a placeholder node of
        shape (0,0) will be added to the node list.
    edges : list, of edges as pairs of nodes.
    id_dtype: class, datatype used to store id values (edge data).

    See also
    --------
    `pubnet.from_dir` and `pubnet.from_data` for constructing.
    """

    def __init__(self, root, nodes=None, edges=None):
        # Maybe should be an optional and if it exists should be used as the
        # default in filtering functions but an alternate could also be
        # supplied.
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

        edge_nodes = reduce(lambda a, n: a + n.split("-"), self.edges, [])
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
        """Add a new node to the network.

        Arguments
        ---------
        name : str, name of the node.
        data : str, Node, or pandas.DataFrame, the data this can be in the form
            of a file path, a DataFrame or an already constructed Node.
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
        """Add a new edge set to the network.

        Arguments
        ---------
        name : str, name of the node set (see `pubnet.network.edge_key` for
            generating the name).
        data : str, Edge, or np.ndarray, the data in the form of a file path, an
            array or an already constructed Node.
        representation : {"numpy", "igraph"}, the method used for storing the
            edge (defaults to "numpy").
        start_id : str, the name of the "from" node.
        end_id : str, the name of the "to" node.

        `start_id` and `end_id` are only needed if `data` is an np.ndarray.
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
        """Filter all the PubNet object's edges to those connecting to
        root_ids.

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
        """Get a list of the root node's IDs that match a condition.

        Arguments
        ---------
        node_type : str, name of the type of nodes to perform
            the search on.
        func : function, a function that accepts a pandas.dataframe and returns
            a list of indices.

        Returns
        -------
        root_ids : array, list of root IDs

        Example
        -------
        publication_ids = net.root_where(
            "Author",
            lambda x: x["LastName" == "Smith"]
        )

        See also
        --------
        ids_containing
        """

        nodes = self[node_type]
        node_idx = func(nodes)

        node_ids = nodes[nodes.id][node_idx]
        root_idx = self[self.root, node_type].isin(node_type, node_ids)

        root_ids = self[self.root, node_type][self.root][root_idx]

        return np.asarray(root_ids, dtype=np.int64)

    def ids_containing(self, node_type, node_feature, value, steps=1):
        """Get a list of publications connected to nodes with a given value.

        Parameters
        ----------
        node_type : str, name of the type of nodes to perform the search on.
        node_feature : str, which feature to compare.
        value : any, the value of the feature to find.
        steps : positive int, number of steps away from the original
            value. Defaults to 1, only publications with direct edges to the
            desired node(s). If `steps > 1`, includes publications with
            indirect edges up to `steps` steps away. For `steps == 2`, all
            direct publications will be returned as well as all publications
            with a node in common to that publication.

            For example:
            pubnet.root_containing(
                "Author",
                "LastName",
                "Smith",
                steps=2
            )

            Will return publications with authors that have last name "Smith"
            and publications by authors who have coauthored a paper with an
            author with last name "Smith".

        Returns
        -------
        root_ids : array, list of publication IDs

        See also
        --------
        `ids_where`
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
        """Filter network to root nodes satisfying a predicate function.

        See also
        --------
        `ids_where`, `containing`.
        """

        root_ids = self.ids_where(node_type, func)
        return self[root_ids]

    def containing(self, node_type, node_feature, value, steps=1):
        """Filter network to root nodes with a given node feature.

        See also
        --------
        `ids_containing`, `where`.
        """

        root_ids = self.ids_containing(node_type, node_feature, value, steps)
        return self[root_ids]

    def plot_distribution(self, node_type, node_feature, threshold=1):
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
        plt.show()

    def drop(self, nodes=None, edges=None):
        """Drop given nodes and edges from the network.

        Arguments
        ---------
        nodes : str tuple, or None, if not none, drop the provided nodes.
        edges : tuple of tuples, None, if not none, drop the provided edges.
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
        """Add the data from other to the current network.

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
        """Find all edges not in self.

        Arguments
        ---------
        edges : list-like, None

        Returns
        -------
        missing_edges : list, edges not in self.
        """

        if edges is None:
            return []

        if isinstance(edges[0], str):
            edges = [edges]

        return list(
            filter(lambda key: key not in self.edges, self._as_keys(edges))
        )

    def _missing_nodes(self, nodes):
        """Find all node names in a list not in self.nodes.

        Arguments
        ---------
        nodes : str, list-like, or None, list of names to test.

        Returns
        -------
        missing_nodes : list, nodes not in self.
        """

        if nodes is None:
            return []

        if isinstance(nodes, str):
            nodes = [nodes]

        return list(filter(lambda key: key not in self.nodes, nodes))


def from_dir(
    root,
    nodes="all",
    edges="all",
    graph_name=None,
    data_dir=default_data_dir(),
    representation="numpy",
):
    """Collect all node and edge files in data_dir and use them to make a
    PubNet object.

    Arguments
    ----------
    root : str, the root node.
    nodes : touple or "all", a list of nodes to read in (default "all").
    edges : touple or "all", a list of pairs of nodes to read in
    (default "all").
    data_dir : str, location of the files (default `default_data_dir`).
    graph_name : str, name of the graph.
    representation : {"numpy", "igraph"}, which edge representation to use.

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

    Example
    -------
    net = pubnet.from_dir(
        ("Author", "Publication"),
        (("Author", "Publication"), ("Publication", "Chemical")),
    )

    See also
    --------
    `pubnet.PubNet`, `pubnet.data.default_data_dir`
    """

    def find_node_files_containing(nodes):
        path_regex = r"^(\w+)_nodes.tsv"
        potential_files = os.listdir(data_dir)
        out = {}
        for file in potential_files:
            m = re.match(path_regex, file)
            if m is not None:
                out[m.groups()[0]] = m.group()

        if nodes != "all":
            out = {key: out[key] for key in nodes}

        return out

    def find_edge_files_containing(nodes):
        path_regex = r"^(\w+)_(\w+)_edges.tsv"
        potential_files = os.listdir(data_dir)
        out = {}
        for file in potential_files:
            m = re.match(path_regex, file)
            if m is not None:
                out[edge_key(*m.groups())] = m.group()

        edge_pair_in_nodes = (
            lambda key: reduce(
                lambda a, k: a + (k in nodes), key.split("-"), 0
            )
            == 2
        )
        if nodes != "all":
            out = {key: out[key] for key in out if edge_pair_in_nodes(key)}

        return out

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
        node_files = find_node_files_containing("all")
        edge_files = find_edge_files_containing("all")
    elif nodes == "all":
        edge_nodes = set(reduce(lambda a, b: a + b, edges, ()))
        node_files = find_node_files_containing(edge_nodes)
        for node_pair in edges:
            edge_files[edge_key(*node_pair)] = _edge_file_path(
                *node_pair, data_dir
            )
    elif edges == "all":
        for node in nodes:
            node_files[node] = _node_file_path(node, data_dir)
        edge_files = find_edge_files_containing(nodes)
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

    return PubNet(root, nodes, edges)


def from_data(root, nodes=None, edges=None, representation="numpy"):
    """Make PubNet object from given nodes and edges.

    Arguments
    ---------
    root : str, root node.
    nodes : Dict, a dictionary of node data of the form {name: DataFrame}.
    edges : Dict, a dictionary of edge data of the form {name: Array}.
    representation : {"numpy", "igraph"} : the edge representation.
    """

    for name, data in nodes:
        nodes[name] = _node.from_data(data)

    for name, data in edges:
        start_id, end_id = name.split("-")
        edges[name] = _edge.from_data(data, start_id, end_id, representation)

    return PubNet(root, nodes, edges)


def edge_key(node_1, node_2):
    """Generate a dictionary key for the given pair of nodes."""

    return "-".join(sorted((node_1, node_2)))


def _node_file_path(name, data_dir):
    """Return the file path for a node."""
    return os.path.join(data_dir, f"{name}_nodes.tsv")


def _edge_file_path(node_1, node_2, data_dir):
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

    if os.path.exists(edge_file_path(node_1, node_2)):
        file_path = edge_file_path(node_1, node_2)
    elif os.path.exists(edge_file_path(node_2, node_1)):
        file_path = edge_file_path(node_2, node_1)
    else:
        raise FileNotFoundError(
            f"No edge file for edges {node_1}, {node_2} found in"
            f" {data_dir}\n\nExpceted either file"
            f" '{edge_file_path(node_1, node_2)}'"
            f" or'{edge_file_path(node_2, node_1)}'"
        )

    return file_path
