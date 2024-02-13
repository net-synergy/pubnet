"""Abstract base class for storing edges."""

import os
from locale import LC_ALL, setlocale
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse as sp

from pubnet.network._utils import edge_gen_file_name, edge_gen_header, edge_key


class Edge:
    """Provides a class for storing edges for `PubNet`.

    In the future it may support weighted edges and directed columns.

    Parameters
    ----------
    data : numpy.ndarray, igraph.Graph
        The edges as a list of existing edges.
    features : dict[str, numpy.ndarray]
        A dictionary of edge features.
    name : str
        Name of the edge set.
    start_id : str
        Name of edge start node type.
    end_id : str
        Name of edge end node type.
    dtype : type
        The type for storing the edge ids.

    Attributes
    ----------
    start_id : str
        The node type in column 0.
    end_id : str
        The node type in column 1.
    dtype : data type,
        The data type used.
    representation : {"numpy", "igraph"}
        Which representation the edges are stored as.
    is_weighted : bool
        Whether the edges are weighted.

    """

    def __init__(
        self,
        data,
        name: str,
        start_id: str,
        end_id: str,
        dtype: type,
        features: dict[str, NDArray[Any]] = {},
    ) -> None:
        self.dtype = dtype
        self.set_data(data)

        for feat_name, feat in features.items():
            self.add_feature(feat, feat_name)

        self.name = name
        self._n_iter = 0
        self.start_id = start_id
        self.end_id = end_id
        self.representation = "Generic"
        # Just assume it's directed if the two ids are the same. May not be a
        # great decision, but useful for now.
        self.isdirected = start_id == end_id

    def set_data(self, new_data) -> None:
        """Replace the edge's data with a new array."""
        self._data = new_data

    def __str__(self) -> str:
        setlocale(LC_ALL, "")

        col_names = [
            f"from: {self.start_id}",
            f"to: {self.end_id}",
        ] + self.features()
        col_len = max(max(len(n) for n in col_names) + 2, 8)

        def align_row(elements):
            def to_str(el):
                if isinstance(el, str):
                    return el

                if el == int(el):
                    return str(int(el))

                return f"{el:g}"

            cells = (
                to_str(el) + " " * (col_len - len(to_str(el)))
                for el in elements
            )
            return "".join(cells)

        header = align_row(col_names)

        if len(self) == 0:
            return "Empty edge set\n" + header

        n_edges = f"Edge set with {len(self):n} edges\n"
        if len(self) < 15:
            first_edges = len(self)
            last_edges = 0
        else:
            first_edges = 5
            last_edges = 5

        edges = "\n".join(
            align_row(line) for line in self[:first_edges].as_array()
        )
        if last_edges > 0:
            edges += f"\n{align_row(['...'] * len(col_names))}\n"
            edges += "\n".join(
                align_row(line) for line in self[-last_edges:].as_array()
            )
        return "\n".join((n_edges, header, edges))

    def __repr__(self) -> str:
        return self.__str__()

    def other_node(self, node_type: str) -> str:
        """Given a node type return the type on the other side of the edge."""
        if node_type == "to":
            return "from"

        if node_type == "from":
            return "to"

        types = {self.start_id, self.end_id}
        if node_type not in types:
            raise KeyError(node_type)

        if len(types) == 1:
            return self.start_id

        return (types.difference({node_type})).pop()

    def _column_to_indices(self, key: str | int) -> tuple[int, int]:
        """Return the index for the provided key and the other key."""
        if isinstance(key, int):
            if key == 0:
                primary = 0
                secondary = 1
            elif key == 1:
                primary = 1
                secondary = 0
            else:
                raise IndexError(
                    "Index out of range. Column index must be 0 or 1."
                )
        elif isinstance(key, str):
            if key == self.start_id or key == "from":
                primary = 0
                secondary = 1
            elif key == self.end_id or key == "to":
                primary = 1
                secondary = 0
            else:
                raise KeyError(
                    f'Key "{key}" not one of "{self.start_id}" or'
                    f' "{self.end_id}".'
                )
        else:
            raise TypeError("Id must be a string or integer.")

        return (primary, secondary)

    def _parse_key(self, key) -> tuple[Any, Any]:
        """Parse a key to get the correct row and column indices."""
        row_index = None
        col_index = None

        if isinstance(key, tuple):
            if len(key) > 2:
                raise IndexError(
                    "Index out of range. Can have at most two indices."
                )
            if len(key) == 2:
                col_index = key[1]

            row_index = key[0]

        elif isinstance(key, str):
            col_index = key
        else:
            row_index = key

        if col_index is not None and not isinstance(col_index, slice):
            col_index = self._column_to_indices(col_index)[0]

        return (row_index, col_index)

    def __getitem__(self, key):
        raise AbstractMethodError(self)

    def __iter__(self):
        self._n_iter = 0
        return self

    def __next__(self):
        if self._n_iter == len(self):
            raise StopIteration

        res = self[self._n_iter,]
        self._n_iter += 1
        return res

    def __len__(self) -> int:
        """Find number of edges."""
        raise AbstractMethodError(self)

    def __contains__(self, item: int) -> bool:
        raise AbstractMethodError(self)

    def isin(
        self, column: str | int, test_elements: ArrayLike
    ) -> NDArray[np.bool_]:
        """Find the elements of column that are in `test_elements`."""
        raise AbstractMethodError(self)

    @property
    def isweighted(self):
        """Test if graph is weighted."""
        return len(self.features) > 0

    def features(self):
        raise AbstractMethodError(self)

    def feature_vector(self, name: str):
        raise AbstractMethodError(self)

    def add_feature(self, feature, name):
        raise AbstractMethodError(self)

    def drop_feature(self, name: str):
        raise AbstractMethodError(self)

    def _assert_has_feature(self, name: str):
        if name not in self.features():
            raise KeyError(
                f"{name} not in features. Available features:\n\t"
                + "\t".join(self.features())
            )

    def isequal(self, other):
        """Determine if two edges are equivalent."""
        raise AbstractMethodError(self)

    def distribution(self, column):
        """Return the distribution of the nodes in column."""
        raise AbstractMethodError(self)

    def to_file(
        self,
        data_dir: str,
        edge_name: Optional[str | tuple[str, str]] = None,
        file_format: str = "tsv",
    ) -> None:
        """Save the edge to disk.

        Parameters
        ----------
        data_dir : str
            Where to store the graph.
        edge_name : str, optional
            The name of the edge. If None, use the edge's name.
        file_format : str {"tsv", "gzip", "binary"}
            How to store the edge (default "tsv"). The gzip method uses
            compresses the tsv. Binary uses numpy's npy file format or pickle
            depending on the edge backend.

        Returns
        -------
        None

        See Also
        --------
        `pubnet.storage.default_data_dir`
        `pubnet.network.PubNet.save_graph`
        `pubnet.network.load_graph`

        """
        if edge_name is None:
            edge_name = self.name

        ext = {"gzip": "tsv.gz", "tsv": "tsv"}

        if self.representation == "igraph":
            ext["binary"] = "pickle"
        elif self.representation == "numpy":
            ext["binary"] = "npy"

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if isinstance(edge_name, tuple):
            edge_name = edge_key(*edge_name)

        file_name, header_name = edge_gen_file_name(
            edge_name, ext[file_format], data_dir
        )
        header = edge_gen_header(self.start_id, self.end_id, self.features())

        if file_format == "binary":
            self._to_binary(file_name, header_name, header)
        else:
            # `np.savetxt` handles "gz" extensions so nothing extra to do.
            self._to_tsv(file_name, header)

    def _to_binary(self, file_name, header_name, header):
        """Save an edge to a binary file type."""
        raise AbstractMethodError(self)

    def _to_tsv(self, file_name, header):
        """Save an edge to a tsv."""
        raise AbstractMethodError(self)

    def get_edgelist(self):
        """Return a list of edges.

        See Also
        --------
        `as_array` to return any features in the edge set along with the edges.

        """
        raise AbstractMethodError(self)

    def as_array(self):
        """Return the edge list as a numpy array."""
        edges = self.get_edgelist()
        feats = tuple(
            np.expand_dims(np.asarray(self.feature_vector(f)), axis=1)
            for f in self.features()
        )
        return np.hstack((edges,) + feats)

    def as_igraph(self):
        """Return the edge as an igraph graph."""
        raise AbstractMethodError(self)

    def to_sparse_matrix(
        self, row=None, column=None, weights=None, shape=None
    ) -> sp.spmatrix:
        """Create a sparse matrix from edge list and a feature."""
        raise AbstractMethodError(self)

    def from_sparse_matrix(
        self,
        mat: sp.spmatrix,
        name: str,
        start_id: str,
        end_id: str,
        feature_name: Optional[str],
    ):
        """Create a new edge based on the sparse matrix."""
        mat = mat.tocoo()
        new_edge = self.__class__(
            np.stack((mat.row, mat.col), axis=1),
            name,
            start_id=start_id,
            end_id=end_id,
            dtype=self.dtype,
        )

        if feature_name is not None:
            new_edge.add_feature(mat.data, feature_name)

        return new_edge

    def _compose_with(self, other, counts: str, mode: str):
        """Use other to create a new edge set that transverses both edges.

        Bipartite edge sets can be treated as a mapping between two node types.
        If the edge set A--B maps a node of type a to a node of type b, then
        the composition of edge sets A--B and B--C should be a new map between
        node type a and node type c, A--C.

        Edges must contain one and only one node type in common, this node type
        is used as the key.
        """
        raise AbstractMethodError(self)

    def overlap(self, node_type: str, weights: Optional[str]):
        """Pairwise number of neighbors nodes have in common.

        For every pair of nodes of the given node type, calculate the number of
        neighbors they have in common.

        Parameters
        ----------
        node_type : str
            The name of one of the columns in the edge set.
        weights : str, optional
            If not none, used to weigh neighbors. Weights should be the name of
            a feature in the edge set.

        """
        raise AbstractMethodError(self)

    def similarity(self, target_publications, method="shortest_path"):
        """Calculate similarity between publications based on edge's overlap.

        Parameters
        ----------
        target_publications : ndarray
            An array of publications to return similarity between which must be
            a subset of all edges in `self.overlap`.
        method : {"shortest_path"}, default "shortest_path"
            The method to use for calculating similarity.

        Returns
        -------
        similarity : a 3 column 2d array
            Listing the similarity (3rd column) between all pairs of
            publications (1st--2nd column) in target_publications. Only
            non-zero similarities are listed.

        """
        all_methods = {
            "shortest_path": self._shortest_path,
            "pagerank": self._pagerank,
        }

        try:
            return all_methods[method](target_publications)
        except AbstractMethodError:
            raise NotImplementedError(
                f"Similarity method '{method}' not implemented for "
                f"'{type(self).__name__}'"
            )

    def _shortest_path(self, target_publications):
        raise AbstractMethodError(self)

    def _pagerank(self, target_publications):
        raise AbstractMethodError(self)

    def _duplicates_to_weights(self, weight_name: str) -> None:
        raise AbstractMethodError(self)

    def _reset_index(self, node: str, old_indices: np.ndarray) -> None:
        raise AbstractMethodError(self)


class AbstractMethodError(NotImplementedError):
    """Error for missing required methods in concrete classes."""

    def __init__(self, class_instance):
        self.class_name = type(class_instance).__name__

    def __str__(self):
        return f"Required method not implemented for {self.class_name}"
