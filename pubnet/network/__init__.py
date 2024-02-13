"""Object for storing publication data as a network.

Components
----------
A graph is made up of a list of node and a list of edges.
"""

import copy
import os
import re
from locale import LC_ALL, setlocale
from typing import Callable, Optional
from warnings import warn

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_list_like
from scipy.sparse import spmatrix

from pubnet.network import _edge
from pubnet.network._edge._base import Edge
from pubnet.network._node import Node
from pubnet.network._utils import edge_key, edge_parts, select_graph_components
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

    def __init__(
        self, nodes=set(), edges=set(), root="Publication", name=None
    ):
        self.root = root
        self.name = name

        if isinstance(nodes, str):
            nodes = {nodes}

        self._node_data = {}
        self._edge_data = {}

        for node in nodes:
            self.add_node(node)

        for edge in edges:
            self.add_edge(edge)

        edge_nodes = {n for e in self.edges for n in edge_parts(e)}

        for name in edge_nodes - self.nodes:
            self.add_node(None, name)

        if self.root not in self.nodes:
            warn(
                f"Constructing PubNet object without {self.root} nodes. "
                "This will limit the functionality of the data type."
            )

        self.id_dtype = _edge.id_dtype

    @property
    def nodes(self) -> set:
        """The set of all nodes in the PubNet object."""
        return set(self._node_data.keys())

    @property
    def edges(self) -> set:
        """The set of all edges in the PubNet object."""
        return set(self._edge_data.keys())

    def edges_containing(self, node: str):
        """Return the set of all edges containing the given node."""
        return {e for e in self.edges if node in edge_parts(e)}

    def select_root(self, new_root) -> None:
        """Switch the graph's root node.

        See `re_root` for modifying edges to reflect the new root.
        """
        if new_root in self.nodes:
            self.root = new_root
            return

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
        if isinstance(data, Node):
            node = data
        elif isinstance(data, str):
            node = Node.from_file(data)
        else:
            node = Node.from_data(data, name=name)

        node.name = name or node.name
        if node.name in self.nodes:
            raise ValueError(f"The node type {name} is already in network.")

        self._node_data[node.name] = node

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
            data = _edge.from_data(
                data, name, **keys, representation=representation
            )

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

        try:
            element = args[0]
        except TypeError:
            return self._slice(np.asarray([args]))

        if isinstance(element, str):
            return self.get_edge(*args)

        return self._slice(np.asarray(args))

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
            if self.root not in edge_parts(key):
                continue

            self.get_edge(key).set_data(
                self.get_edge(key)[
                    self.get_edge(key).isin(self.root, root_ids), :
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

                node_ids = edge[key]

            node_locs = np.isin(self.get_node(key).index, node_ids)
            self.get_node(key).set_data(self.get_node(key)[node_locs])

        return self

    def __repr__(self):
        setlocale(LC_ALL, "")

        def sep(name, col_len):
            return " " * (col_len - len(name))

        node_col = max((len(n) for n in self.nodes), default=0) + 4
        edge_col = max((len(e) for e in self.edges), default=0) + 4

        res = f"{self.name} Publication Network\nroot: {self.root}"
        res += "\n\nNode types:"
        for n in self.nodes:
            res += f"\n    {n}{sep(n, node_col)}({len(self._node_data[n]):n})"
        res += "\n\nEdge sets:"
        for e in self.edges:
            res += f"\n    {e}{sep(e, edge_col)}({len(self._edge_data[e]):n})"

        return res

    def ids_where(self, node_type: str, func, root: Optional[str] = None):
        """Get a list of the root node's IDs that match a condition.

        Parameters
        ----------
        node_type : str
          Name of the type of nodes to perform the search on.
        func : function
          A function that accepts a pandas.dataframe and returns a list of
          indices.
        root : str or None
          The root to use. If None (default) use the networks current root.

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
        root = root or self.root

        node_ids = nodes.index[node_idx]
        if node_type == root:
            root_ids = node_ids
        else:
            root_idx = self[root, node_type].isin(node_type, node_ids)
            root_ids = self[root, node_type][root][root_idx]

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

    def where(
        self,
        node_type: str,
        func: Callable[[pd.DataFrame], np.ndarray],
        in_place: bool = False,
        root: Optional[str] = None,
    ):
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
        old_root = self.root
        if root:
            self.select_root(root)

        root_ids = self.ids_where(node_type, func)
        if in_place:
            self._slice(root_ids, mutate=True)
            self.select_root(old_root)
            return None

        new_net = self[root_ids]
        new_net.select_root(old_root)

        return new_net

    def containing(self, node_type, node_feature, value, steps=1):
        """Filter network to root nodes with a given node feature.

        See Also
        --------
        `PubNet.ids_containing`
        `PubNet.where`

        """
        root_ids = self.ids_containing(node_type, node_feature, value, steps)
        return self[root_ids]

    def re_root(
        self,
        new_root: str,
        drop_unused: bool = True,
        counts: str = "drop",
        mode: str = "all",
    ) -> None:
        r"""Change the networks root, creating new edges.

        The root of the network should be the primary node type, which, at
        least most, edges contain. Re-rooting uses the edge between the current
        and new root as a key to map the new root to the other nodes in the
        network. For example, if the original root is "Publication" and there
        are edges between publications and authors, chemicals, and keywords,
        after re-rooting the network the edges will be between authors and
        publications, chemicals, and keywords.

        Parameters
        ----------
        new_root : str
            The node type in the network to base edges off.
        drop_unused : bool
            Whether to drop all edges that are not related to the new root.
        counts : str, {"drop", "absolute", "normalize"}
            Counts are the number of edges between root and the other edge
            type. For example if an author has three publications each of which
            are on a common chemical, the count between that author and
            chemical would be 3.

            When "drop" (default), the counts are not stored. Otherwise counts
            are stored as an edge feature "counts". If "absolute", store the
            raw counts, if "normalize" relative to the number of edges for each
            node in the new root. So if the above author also had an edge with
            1 other chemical, that authors counts would be 3/4 and 1/4.
        mode : str in {"all", "out", "in"}
            What direction to calculate the overlap in if the edge is directed.
            "all" creates a in and an out edge set. For example, references
            are directed, being referenced is different than referencing. So
            "all" produces an edge for root -- references out (referenced by
            the root) and root -- references in (root was referenced).

        See Also
        --------
        `PubNet.select_root` to change the root without modifying edges.

        """
        root_edges = [
            e
            for e in self.edges
            if self.root
            in (self.get_edge(e).start_id or self.get_edge(e).end_id)
        ]

        if drop_unused:
            self.drop(edges=self.edges.difference(root_edges))

        if new_root == self.root:
            return

        if edge_key(self.root, new_root) not in self.edges:
            raise AssertionError(
                "No edge set found linking the old root to the new root."
                " Cannot reroot."
            )

        if counts not in ("drop", "absolute", "normalize"):
            raise ValueError(counts)

        if counts == "normalize":
            counts = "normalize_other"

        mode_i = "in" if mode == "in" else "out"

        map_edge = self.get_edge(self.root, new_root)
        for e in self.edges - {map_edge.name}:
            self.add_edge(
                self.get_edge(e)._compose_with(map_edge, counts, mode_i)
            )
            if self.get_edge(e).isdirected and mode == "both":
                self.add_edge(
                    self.get_edge(e)._compose_with(map_edge, counts, "in")
                )

            self.drop(edges=e)

        self.select_root(new_root)

    def overlap(
        self,
        node_type: str | set[str] = "all",
        weights: Optional[str] = None,
        mutate: bool = True,
    ):
        r"""Calculate the overlap in neighbors between nodes.

        Creates new overlap edges with an overlap feature that contains the
        number of neighbors of `node_type` the nodes of the networks root
        have in common.

        Parameters
        ----------
        node_type : str or sequence of strings
            If "all", default, create overlap edges for all available edge
            sets. Available edge sets are those where one side is the root and,
            if a weight is provided (see below), has the required feature.
        weights : str, optional
            The name of a feature in the edge set to weight the overlap by. If
            None, the default, implicitly use 1 as the weight for all elements.
            If a string, only edges that contain that feature are considered.
        mutate : bool, optional
            If True (default) mutate the PubNet in place, The PubNet will
            contain all it's old edges plus the overlap edges. If False, return
            a new PubNet with only the overlap edges and root node.

        Example
        -------
        Calculate the number of chemicals each root node has in common with
        each other root node.

        >>> pubs.overlap("Chemical")
        >>> pubs[pubs.root, "ChemicalOverlap"].feature_vector("overlap")

        See Also
        --------
        `PubNet.select_root` for changing the network's root node type.
        `PubNet.re_root` for translating the current root edges to a new root.

        """

        def not_root(edge_key):
            n1, n2 = edge_parts(edge_key)
            if n1 == self.root:
                return n2
            return n1

        root_edges = {e for e in self.edges if self.root in edge_parts(e)}

        if isinstance(node_type, str):
            node_type = {node_type}

        if "all" not in node_type:
            root_edges = {e for e in root_edges if not_root(e) in node_type}

        if weights is not None:
            root_edges = {
                e for e in root_edges if weights in self.get_edge(e).features()
            }

        if not root_edges:
            raise ValueError(
                "Could not find any edge sets that fit the requirements."
            )

        if mutate:
            new_pubnet = self
        else:
            new_pubnet = PubNet(
                nodes={self[self.root]}, root=self.root, name="overlap"
            )

        for e in root_edges:
            new_pubnet.add_edge(
                self.get_edge(e).overlap(self.root, weights),
            )

        if not mutate:
            return new_pubnet

        return None

    def reduce_edges(
        self,
        func: Callable[[spmatrix, spmatrix], spmatrix],
        edge_feature: str,
        normalize: bool = False,
    ) -> Edge:
        """Reduce network edges on a feature.

        Reduce a group of edge sets by accumulating with a function. All edges
        to be reduced must have the provided edge feature. Each edge feature
        should have the same start and end node type otherwise results can not
        be interpreted.

        The method will try to be smart about selecting edges for which this
        operation make sense, but it is best to start with a PubNet with only
        edges that can be meaningfully combined.

        Parameters
        ----------
        func : callable
            A function that accepts to sparse matrices and returns one sparse
            matrix. The returned result should be some kind of combination of
            the inputs. Example: `lambda x acc: x + acc`
        edge_feature : str
            The name of a feature common to all edges that will be reduced.
            This feature will act as the data of the sparse matrices
        normalize : bool, optional
            Default False. If True, divide the results by the number of edges
            reduced.

        Returns
        -------
        new_edge : Edge
            An edge whose list of edges is equal to the union of all edge sets
            list of edges. The edge has a single feature with the same name as
            `edge_feature` with the resulting reduced data.

        """
        featured_edges = {
            e
            for e in self.edges
            if edge_feature in self.get_edge(e).features()
        }

        n_edges = len(featured_edges)
        if n_edges == 0:
            raise ValueError("No edge sets meet the requirements.")

        shape = (
            max(self.get_edge(e)[:, 0].max() for e in featured_edges) + 1,
            max(self.get_edge(e)[:, 1].max() for e in featured_edges) + 1,
        )

        def to_sparse(edge):
            return edge.to_sparse_matrix(
                row="from", weights=edge_feature, shape=shape
            )

        base_edge = self.get_edge(featured_edges.pop())
        acc = to_sparse(base_edge)
        for e in featured_edges:
            acc = func(to_sparse(self.get_edge(e)), acc)
            self.drop(edges=e)

        if normalize:
            acc = acc / n_edges

        return base_edge.from_sparse_matrix(
            acc,
            "Composite-" + edge_feature.title(),
            start_id=base_edge.start_id,
            end_id=base_edge.end_id,
            feature_name=edge_feature,
        )

    def plot_distribution(
        self, node_type, node_feature, threshold=0, max_n=20, fname=None
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
        max_n : int, optional
            The maximum number of bars to plot. If none, plot all.
        fname : str, optional
            The name of the figure.

        """
        import matplotlib.pyplot as plt

        node_ids, distribution = self[self.root, node_type].distribution(
            node_type
        )
        retain = distribution >= threshold
        distribution = distribution[retain]
        node_ids = node_ids[retain]

        node = self.get_node(node_type)
        names = node[np.isin(node.index, node_ids)].feature_vector(
            node_feature
        )

        indices = np.argsort(distribution)[-1::-1]
        names = np.take_along_axis(names, indices, axis=0)
        distribution = np.take_along_axis(distribution, indices, axis=0)

        if max_n is not None:
            names = names[:max_n]
            distribution = distribution[:max_n]

        fig, ax = plt.subplots()
        ax.bar(names, distribution)

        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        ax.set_xlabel(node_feature)
        ax.set_ylabel(f"{self.root} occurance")

        if fname:
            plt.savefig(fname)
        else:
            plt.show()

    def drop(self, nodes=set(), edges=set()):
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
        if isinstance(nodes, str):
            nodes = {nodes}
        if isinstance(edges, str):
            edges = {edges}

        assert len(self._missing_nodes(nodes)) == 0, (
            f"Node(s) {self._missing_nodes(nodes)} is not in network",
            f"\n\nNetwork's nodes are {self.nodes}.",
        )

        assert len(self._missing_edges(edges)) == 0, (
            f"Edge(s) {self._missing_edges(edges)} is not in network",
            f"\n\nNetwork's edges are {self.edges}.",
        )

        for node in nodes:
            self._node_data.pop(node)

        edges = self._as_keys(edges)

        for edge in edges:
            self._edge_data.pop(edge)

    def update(self, other):
        """Add the data from other to the current network.

        Behaves similar to Dict.update(), if other contains nodes or edges in
        this network, the values in other will replace this network's.

        This command mutates the current network and returns nothing.
        """
        self._node_data.update(other._node_data)
        self._edge_data.update(other._edge_data)

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

    def refresh_edges(self):
        """Recreate edge keys if they get out of sync with edge names."""
        self._edge_data = {
            self.get_edge(e).name: self.get_edge(e) for e in self.edges
        }

    def mutate_node(
        self,
        name: str,
        template_node: str,
        rule: Callable,
        feature_name: str = "",
        discard_used: bool = True,
    ) -> None:
        """Produce a new node from a modification of an existing node.

        Parameters
        ----------
        name : str
            The name of the to be created node. Can be the same as
            `template_node`, in which case, the node gets replaced with it's
            mutation.
        template_node : str
            Name of the node, in self, to mutate.
        rule : callable
            A function that takes the template node and returns a mapping from
            the template node's indices to the new nodes values. This should be
            a two dimensional array with one row for each new value. If there
            should be more than one feature, each row will be template node's
            index, feature 1, feature 2, ...
        feature_name : str
            What to name the new feature. If left as an empty list, the feature
            will be given the same name as the node (i.e. `name`).
        discard_used : bool, default True
            Whether the old node and it's edges should be discarded are kept in
            the PubNet. Cannot be False if `name == template_node`.

        See Also
        --------
        `PubNet.mutate_node_re`

        """
        if name == template_node:
            if not discard_used:
                raise ValueError(
                    "If new node is the same as template node, the template"
                    " node must be discarded. Set `discard_used` to True to"
                    " allow the template node to be discarded."
                )

            characters = [chr(i) for i in range(ord("a"), ord("z") + 1)]
            tmp_name = "".join(np.random.choice(characters, 20))
            self.mutate_node(
                tmp_name, template_node, rule, feature_name, discard_used
            )

            node = self._node_data.pop(tmp_name)
            node.rename_index(name + "ID")
            node.rename_column(tmp_name, template_node)
            self.add_node(node, name=name)
            for e in self.edges:
                edge = self.get_edge(e)
                if tmp_name in (edge.start_id, edge.end_id):
                    other = edge.other_node(tmp_name)
                    edge.name = edge_key(other, name)
                    edge.start_id = (
                        name if edge.start_id == tmp_name else edge.start_id
                    )
                    edge.end_id = (
                        name if edge.end_id == tmp_name else edge.end_id
                    )
            self.refresh_edges()
            return

        feature_name = feature_name or name
        old_to_new = rule(self.get_node(template_node))
        new_values, indices = np.unique(old_to_new[:, 1], return_inverse=True)
        self.add_node(pd.DataFrame({feature_name: new_values}), name=name)
        new_edge = _edge.from_data(
            np.concatenate(
                (
                    old_to_new[:, 0:1].astype(_edge.id_dtype),
                    np.expand_dims(indices, axis=1),
                ),
                axis=1,
                dtype=_edge.id_dtype,
            ),
            edge_key(template_node, name),
            start_id=template_node,
            end_id=name,
        )

        for e in self.edges:
            if template_node in edge_parts(e):
                self.add_edge(
                    new_edge._compose_with(self.get_edge(e), "drop", "all")
                )
                if discard_used:
                    self.drop(edges=e)

        if discard_used:
            self.drop(nodes=template_node)

    def mutate_node_re(
        self,
        name: str,
        pattern: str | re.Pattern,
        template_node: str,
        feature: str,
        feature_name: str = "",
        discard_used: bool = True,
    ) -> None:
        """Create new nodes from regex matches of an old node.

        A special case of `PubNet.mutate_node` for creating a new node based on
        regex matches on an existing node's values.

        Parameters
        ----------
        name : str
            The name of the to be created node.
        pattern : str
            A regex pattern that will be matched against the selected feature
            vector's values. To collect only a group from the pattern, use a
            named group. Can only have one named group per pattern.
        template_node : str
            Name of the node, in self, to mutate from.
        feature : str
            Name of the feature in `template_node` the values to match come
            from. The feature should contain string values.
        feature_name : str
            The name for the new feature. If left blank and the pattern
            contains a named group, the groups name will be used, otherwise the
            name of the new node will be used.
        discard_used : bool, default True
            Whether the template node and it's edges should be discarded after
            mutating.

        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        if pattern.groups > 1:
            raise ValueError("Cannot handle multiple capturing groups.")

        group_name = (
            None
            if not pattern.groupindex
            else list(pattern.groupindex.keys())[0]
        )
        feature_name = feature_name or group_name or name

        def rule(node):
            strings = node.feature_vector(feature)
            re_filter = (
                (i, re.search(pattern, s)) for i, s in zip(node.index, strings)
            )
            return np.fromiter(
                (
                    (i, m.group(group_name) if group_name else m.group())
                    for i, m in re_filter
                    if m is not None
                ),
                dtype=np.dtype((object, 2)),
            )

        self.mutate_node(
            name,
            template_node,
            rule,
            feature_name=feature_name,
            discard_used=discard_used,
        )

    def edges_to_igraph(self):
        """Convert all edge sets to the igraph backend."""
        for e in self.edges:
            self._edge_data[e] = _edge.from_edge(
                self.get_edge(e), representation="igraph"
            )

    def edges_to_numpy(self):
        """Convert all edge sets to the numpy backend."""
        for e in self.edges:
            self._edge_data[e] = _edge.from_edge(
                self.get_edge(e), representation="numpy"
            )

    def _as_keys(self, edges):
        """Convert a list of edges to their keys."""
        # A tuple of 2 strings is likely two edge parts that need to be
        # converted to an edge key, but it could also be two edge keys that
        # should not be converted.
        if (
            len(edges) == 2
            and not isinstance(edges, set)
            and isinstance(edges[0], str)
        ):
            try:
                _, _ = edge_parts(edges[0])
            except ValueError:
                edges = {edges}

        return {edge_key(*e) if len(e) == 2 else e for e in edges}

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
        return self._as_keys(edges) - self.edges

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
        return set(nodes) - self.nodes

    def repack(self, nodes: Optional[str | tuple[str, ...]] = None) -> None:
        """Repack the graphs indices.

        The indices for the requested nodes will be renumber to sequential
        values between 0 and node.shape[0]. All edges will be updated with the
        new indices.

        Parameters
        ----------
        nodes : str, tuple(str, ...), None
          If no nodes are provided (default), repack all nodes in the network.
          Otherwise repack the explicitly requested node(s).

        """

        def _repack_node(node: str) -> None:
            old_indices = self.get_node(node).index
            self.get_node(node)._reset_index()
            for edge in self.edges_containing(node):
                self.get_edge(edge)._reset_index(node, old_indices)

        nodes = nodes or tuple(self.nodes)
        if isinstance(nodes, str):
            nodes = (nodes,)

        for node in nodes:
            _repack_node(node)

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
        nodes: str | tuple[str, ...] = "all",
        edges: str | tuple[tuple[str, str], ...] = "all",
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
        graph_dir = graph_path(name, data_dir)
        if not os.path.exists(graph_dir):
            raise FileNotFoundError(
                f'Graph "{name}" not found. Available graphs are: \n\t%s'
                % "\n\t".join(g for g in list_graphs(data_dir))
            )

        node_files, edge_files = select_graph_components(
            nodes, edges, graph_dir
        )
        net_nodes = [
            Node.from_file(file)
            for file in node_files.values()
            if file is not None
        ]
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
