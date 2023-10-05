"""Object for storing publication data as a network.

Components
----------
A graph is made up of a list of node and a list of edges.
"""

import copy
import os
from locale import LC_ALL, setlocale
from typing import Optional
from warnings import warn

import numpy as np
from pandas.core.dtypes.common import is_list_like

from pubnet.network import _edge
from pubnet.network._edge._base import Edge
from pubnet.network._node import Node
from pubnet.network._utils import (
    edge_files_containing,
    edge_find_file,
    edge_key,
    edge_list_files,
    edge_parts,
    node_files_containing,
)
from pubnet.storage import delete_graph, graph_path, list_graphs

__all__ = ["edge_key", "PubNet", "Edge", "Node"]


class PubNet:
    """Store publication network as a set of graphs.

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
    Use `load_graph` to construct a PubNet object instead of initializing
    directly.

    See Also
    --------
    `load_graph`
    `from_data`
    """

    def __init__(self, nodes=None, edges=None, root="Publication", name=None):
        self.root = root
        self.name = name

        if nodes is None:
            nodes = []
        if edges is None:
            edges = []

        self._node_data = {}
        self._edge_data = {}
        self.nodes = []
        self.edges = []

        for node in nodes:
            self.add_node(node)

        for edge in edges:
            self.add_edge(edge)

        edge_nodes = {n for e in self.edges for n in edge_parts(e)}

        for name in edge_nodes - set(self.nodes):
            self.add_node(None, name)

        if self.root not in self.nodes:
            warn(
                f"Constructing PubNet object without {self.root} nodes. "
                "This will limit the functionality of the data type."
            )

        self.id_dtype = _edge.id_dtype

    def select_root(self, new_root) -> None:
        """Switch the graph's root node."""
        if new_root in self.nodes:
            self.root = new_root

        available_nodes = "\n\t".join(self.nodes)
        raise KeyError(
            f"{new_root} not in graphs set of nodes.\nMust be one of"
            f"\n\t{available_nodes}"
        )

    def add_node(self, data, name=None):
        """Add a new node to the network.

        Parameters
        ----------
        data : str, Node, or pandas.DataFrame
            The data this can be in the form of a file path, a DataFrame or
            an already constructed Node.
        name : str, optional
            Name of the node. If None, use the data's name if available,
            otherwise raises an error.

        See Also
        --------
        `PubNet.add_edge`
        `PubNet.drop`
        """
        if isinstance(data, str):
            data = Node.from_file(data)
        elif data is None or not isinstance(data, Node):
            data = Node.from_data(data)

        if name is None:
            try:
                name = data.name
            except AttributeError:
                raise ValueError(
                    "Data does not provide a name. Name must be supplied."
                )

        if name in self.nodes:
            raise ValueError(f"The node type {name} is already in network.")

        self.nodes.append(name)
        self._node_data[name] = data

    def add_edge(
        self,
        data,
        name=None,
        representation="numpy",
        **keys,
    ) -> None:
        """Add a new edge set to the network.

        Parameters
        ----------
        data : str, Edge, np.ndarray
            The data in the form of a file path, an array or an already
            constructed edge.
        name : str, optional
            Name of the node pair. If none, uses the data's name.
        representation : {"numpy", "igraph"}, default "numpy"
            The backend representation used for storing the edge.
        start_id : str, optional
            The name of the "from" node.
        end_id : str, optional
            The name of the "to" node.
        **keys : Any
            Keyword arguments to be forwarded to `_edge.from_data` if the data
            isn't already an Edge.

        `start_id` and `end_id` are only needed if `data` is an np.ndarray.

        See Also
        --------
        `PubNet.add_node` for analogous node method.
        `PubNet.drop` to remove edges and nodes.
        """
        if isinstance(data, str):
            data = _edge.from_file(data, representation)
        elif not isinstance(data, _edge.Edge):
            data = _edge.from_data(data, **keys, representation=representation)

        if name is None:
            try:
                name = data.name
            except AttributeError:
                raise ValueError(
                    "Name not supplied by data. Need to supply a name."
                )
        elif isinstance(name, tuple):
            name = edge_key(*name)

        if name in self.edges:
            raise ValueError(f"The edge {name} is already in the network.")

        self.edges.append(name)
        self._edge_data[name] = data

    def get_node(self, name) -> Node:
        """Retrieve the Node in the PubNet object with the given name."""
        return self._node_data[name]

    def get_edge(self, name, node_2=None) -> Edge:
        """Retrieve the Edge in the PubNet object with the given name."""
        if isinstance(name, tuple):
            if len(name) > 2 or node_2 is not None:
                raise KeyError("Too many keys. Accepts at most two keys.")

            name, node_2 = name

        if node_2 is not None:
            name = edge_key(name, node_2)

        return self._edge_data[name]

    def __getitem__(self, args):
        if isinstance(args, str):
            if args in self.nodes:
                return self.get_node(args)

            if args in self.edges:
                return self.get_edge(args)

            raise KeyError(args)

        is_string_array = isinstance(args, np.ndarray) and isinstance(
            args[0], str
        )
        if (is_string_array or isinstance(args, tuple)) and (len(args) == 2):
            return self.get_edge(*args)

        if isinstance(args, np.ndarray | range):
            return self._slice(args)

        if isinstance(args, (self.id_dtype, int)):
            return self._slice(np.asarray([args]))

        raise KeyError(*args)

    def _slice(self, root_ids, mutate=False):
        """Filter the PubNet object's edges to those connected to root_ids.

        If mutate is False return a new `PubNet` object otherwise
        return self after mutating the edges.
        """
        if not mutate:
            new_pubnet = copy.deepcopy(self)
            new_pubnet._slice(root_ids, mutate=True)
            return new_pubnet

        for key in self.edges:
            self.get_edge(key).set_data(
                self.get_edge(key)[
                    self.get_edge(key).isin(self.root, root_ids)
                ]
            )

        for key in self.nodes:
            if len(self[key]) == 0:
                continue

            if key == self.root:
                node_ids = root_ids
            else:
                try:
                    edge = self.get_edge(key, self.root)
                except KeyError:
                    continue

                if len(edge) == 0:
                    continue

                node_ids = edge[key]

            node_locs = np.isin(self.get_node(key).index, node_ids)
            self.get_node(key).set_data(self.get_node(key)[node_locs])

        return self

    def __repr__(self):
        setlocale(LC_ALL, "")

        res = f"{self.name} Publication Network"
        res += "\n\nNode types:"
        for n in self.nodes:
            res += f"\n\t{n}\t({len(self._node_data[n]):n})"
        res += "\n\nEdge sets:"
        for e in self.edges:
            res += f"\n\t{e}\t({len(self._edge_data[e]):n})"

        return res

    def ids_where(self, node_type, func):
        """Get a list of the root node's IDs that match a condition.

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
        >>> net = PubNet.load_graph(name="author_net", root="Publication")
        >>> publication_ids = net.ids_where(
        ...     "Author",
        ...     lambda x: x["LastName" == "Smith"]
        ... )

        See Also
        --------
        `PubNet.ids_containing`
        """
        nodes = self.get_node(node_type)
        node_idx = func(nodes)

        node_ids = nodes.index[node_idx]
        root_idx = self[self.root, node_type].isin(node_type, node_ids)

        root_ids = self[self.root, node_type][self.root][root_idx]

        return np.asarray(root_ids, dtype=np.int64)

    def ids_containing(self, node_type, node_feature, value, steps=1):
        """Get a list of root IDs connected to nodes with a given value.

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

        See Also
        --------
        `PubNet.ids_where`
        """
        assert (
            isinstance(steps, int) and steps >= 1
        ), f"Steps most be a positive integer, got {steps} instead."

        if is_list_like(value):
            func = lambda x: np.isin(x.feature_vector(node_feature), value)
        else:
            func = lambda x: x.feature_vector(node_feature) == value

        root_ids = self.ids_where(node_type, func)
        while steps > 1:
            node_ids = self[self.root, node_type][node_type][
                self[self.root, node_type].isin(self.root, root_ids)
            ]
            func = lambda x: np.isin(x.index, node_ids)
            root_ids = self.ids_where(node_type, func)
            steps -= 1

        return root_ids

    def where(self, node_type, func):
        """Filter network to root nodes satisfying a predicate function.

        All graphs are reduced to a subset of edges related to those associated
        with the root nodes that satisfy the predicate function.

        Returns
        -------
        subnet : PubNet
            A new PubNet object that is subset of the original.

        See Also
        --------
        `PubNet.ids_where`
        `PubNet.containing`
        """
        root_ids = self.ids_where(node_type, func)
        return self[root_ids]

    def containing(self, node_type, node_feature, value, steps=1):
        """Filter network to root nodes with a given node feature.

        See Also
        --------
        `PubNet.ids_containing`
        `PubNet.where`
        """
        root_ids = self.ids_containing(node_type, node_feature, value, steps)
        return self[root_ids]

    def plot_distribution(
        self, node_type, node_feature, threshold=1, fname=None
    ):
        """Plot the distribution of the values of a node's feature.

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
        import matplotlib.pyplot as plt

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

        See Also
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

        self.nodes = [n for n in self.nodes if n not in nodes]

        edges = self._as_keys(edges) if edges is not None else []

        for edge in edges:
            self._edge_data.pop(edge)

        self.edges = [e for e in self.edges if e not in edges]

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

        return all(self[e].isequal(other[e]) for e in self.edges)

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

        Parameters
        ----------
        edges : list-like, optional
            A list of edge names

        Returns
        -------
        missing_edges : list
            Edges not in self.
        """
        if edges is None:
            return []

        if isinstance(edges[0], str):
            edges = [edges]

        return [key for key in self._as_keys(edges) if key not in self.edges]

    def _missing_nodes(self, nodes):
        """Find all node names in a list not in self.nodes.

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

        return [n for n in nodes if n not in self.nodes]

    def save_graph(
        self,
        name=None,
        nodes="all",
        edges="all",
        data_dir=None,
        file_format="tsv",
        overwrite=False,
    ):
        """Save a graph to disk.

        Parameters
        ----------
        name : str
            What to name the graph. If not set, defaults to graph's name.
        nodes : tuple or "all", default "all"
            A list of nodes to save. If "all", see notes.
        edges : tuple or "all", default "all"
            A list of edges to save. If "all", see notes.
        data_dir : str, optional
            Where to save the graph, defaults to the default data directory.
        file_format : {"tsv", "gzip", "binary"}, default "tsv"
            How to store the files.
        overwrite : bool, default False
            If true delete the current graph on disk. This may be useful for
            replacing a plain text representation with a binary representation
            if storage is a concern. WARNING: This can lose data if the self
            does not contain all the nodes/edges that are in the saved graph.
            Tries to perform the deletion as late as possible to prevent errors
            from erasing data without replacing it, but it may be safer to save
            the data to a new location then delete the graph (with
            `pubnet.storage.delete_graph`) after confirming the save worked
            correctly.

        Notes
        -----
        If nodes and edges are both "all" store the entire graph. If nodes is
        "all" and edges is a tuple, save all nodes in the list of
        edges. Similarly, if edges is "all" and nodes is a tuple, save all
        edges where both the start and end nodes are in the node list.

        See Also
        --------
        `pubnet.storage.default_data_dir`
        `load_graph`
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
        edges = [e for e in edges if len(self[e]) > 0]

        if name is None:
            name = self.name

        if name is None:
            raise ValueError(
                "Name must be set but is None. Pass a name to the"
                "function call or set the graphs name."
            )

        save_dir = graph_path(name, data_dir)

        if overwrite:
            delete_graph(name, data_dir)

        for n in nodes:
            self.get_node(n).to_file(save_dir, file_format=file_format)

        for e in edges:
            self.get_edge(e).to_file(save_dir, file_format=file_format)

    @classmethod
    def load_graph(
        cls,
        name: str,
        nodes: Optional[str | tuple[str, ...]] = "all",
        edges: Optional[str | tuple[tuple[str, str], ...]] = "all",
        root: str = "Publication",
        data_dir: Optional[str] = None,
        representation: str = "numpy",
    ):
        """Load a graph as a PubNet object.

        See `PubNet` for more information about parameters.

        Parameters
        ----------
        name : str
            Name of the graph, stored in `default_data_dir` or `data_dir`.
        nodes : tuple or "all", (default "all")
            A list of nodes to read in.
        edges : tuple or "all", (default "all")
            A list of pairs of nodes to read in.
        root : str, default "Publication
            The root node.
        data_dir : str, optional
            Where the graph is saved, defaults to default data directory.
        representation : {"numpy", "igraph"}, default "numpy"
            Which edge backend representation to use.

        Returns
        -------
        A PubNet object.

        Notes
        -----
        Node files are expected to be in the form f"{node_name}_nodes.tsv" and
        edge files should be of the form
        f"{node_1_name}_{node_2_name}_edges.tsv". The order nodes are supplied
        for edges does not matter, it will look for files in both orders.

        If nodes or edges is "all" it will look for all files in the directory
        that match the above file patterns. When one is "all" but the other is
        a list, it will only look for files containing the provided nodes. For
        example, if nodes = ("Author", "Publication", "Chemical") and edges =
        "all", it will only look for edges between those nodes and would ignore
        files such as "Publication_Descriptor_edges.tsv".

        Graph name is the name of the directory the graph specific files are
        found in. It is added to the end of the `data_dir`, so it is equivalent
        to passing `os.path.join(data_dir, name)` for `data_dir`, the reason to
        separate them is to easily store multiple separate graphs in the
        `default_data_dir` by only passing a `name` and leaving `data_dir` as
        default.

        Examples
        --------
        >>> net = pubnet.load_graph(
        ...     "author_net"
        ...     ("Author", "Publication"),
        ...     (("Author", "Publication"), ("Publication", "Chemical")),
        ... )

        See Also
        --------
        `pubnet.network.PubNet`
        `pubnet.storage.default_data_dir`
        `from_data`
        """
        if nodes is None:
            nodes = ()

        if edges is None:
            edges = ()

        assert isinstance(
            nodes, (str, tuple)
        ), "Nodes must be a string or a tuple."

        assert isinstance(
            edges, (str, tuple)
        ), 'Edges must be a tuple or "all".'

        if isinstance(nodes, str) and nodes != "all":
            raise TypeError('Nodes must be a tuple or "all"')
        if isinstance(edges, str) and edges != "all":
            raise TypeError('Edges must be a tuple of tuples or "all"')

        save_dir = graph_path(name, data_dir)
        if not os.path.exists(save_dir):
            raise FileNotFoundError(
                f'Graph "{name}" not found. Available graphs are: \n\t%s'
                % "\n\t".join(g for g in list_graphs(data_dir))
            )

        if (nodes == "all") and (edges != "all"):
            nodes = tuple({n for e in edges for n in e})

        if edges != "all":
            all_edge_files = edge_list_files(save_dir)
            edge_files = {
                edge_key(e[0], e[1]): edge_find_file(
                    e[0], e[1], all_edge_files
                )
                for e in edges
            }
        else:
            edge_files = edge_files_containing(nodes, save_dir)

        node_files = node_files_containing(nodes, save_dir)

        net_nodes = [Node.from_file(file) for file in node_files.values()]
        net_edges = [
            _edge.from_file(file, representation)
            for file in edge_files.values()
        ]

        return PubNet(root=root, nodes=net_nodes, edges=net_edges, name=name)

    @classmethod
    def from_data(
        cls,
        name: str,
        nodes: dict[str, Node] = {},
        edges: dict[str, Edge] = {},
        root: str = "Publication",
        representation: str = "numpy",
    ):
        """Make PubNet object from given nodes and edges.

        Parameters
        ----------
        name : str
            What to name the graph. This is used for saving graphs.
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
        `load_graph`
        """
        for n_name, n in nodes.items():
            nodes[n_name] = Node.from_data(n)

        for e_name, e in edges.items():
            start_id, end_id = edge_parts(e_name)
            edges[e_name] = _edge.from_data(
                e, e_name, {}, start_id, end_id, representation
            )

        return PubNet(root=root, nodes=nodes, edges=edges, name=name)
