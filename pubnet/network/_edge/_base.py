"""Abstract base class for storing edges."""

from locale import LC_ALL, setlocale
from math import ceil, log10
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Edge:
    """
    Provides a class for storing edges for `PubNet`.

    In the future it may support weighted edges and directed columns.

    Parameters
    ----------
    data : numpy.ndarray, igraph.Graph
        The edges as a list of existing edges.
    start_id : str
        Name of edge start node type.
    end_id : str
        Name of edge end node type.

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
    isweighted : bool
        Whether the edges are weighted.
    shape
    """

    def __init__(self, data, start_id: str, end_id: str, dtype: type) -> None:
        self.set(data)
        self._n_iter = 0
        self.start_id = start_id
        self.end_id = end_id
        self.dtype = dtype
        self.representation = "Generic"

        # Weighted not implemented yet
        self.isweighted = False

    def set(self, new_data) -> None:
        """Replace the edge's data with a new array."""
        self._data = new_data

    def __str__(self) -> str:
        setlocale(LC_ALL, "")

        n_edges = f"Edge set with {len(self):n} edges\n"
        columns = f"{self.start_id}\t{self.end_id}"

        if len(self) == 0:
            return "Empty edge set\n" + columns

        def sep(src: int) -> str:
            return (
                1
                + ceil((len(self.start_id) + 0.01) / 8)
                - ceil((log10(src) + 1.01) / 8)
            ) * "\t"

        if len(self) < 15:
            first_edges = len(self)
            last_edges = 0
        else:
            first_edges = 5
            last_edges = 5

        edges = "%s" % "\n".join(
            f"{e[0]}{sep(e[0])}{e[1]}" for e in self[:first_edges].as_array()
        )
        if last_edges > 0:
            edges += "\n.\n.\n.\n"
            edges += "%s" % "\n".join(
                f"{e[0]}{sep(e[0])}{e[1]}"
                for e in self[
                    len(self) - 1 : len(self) - (last_edges + 1) : -1
                ].as_array()
            )
        return "\n".join((n_edges, columns, edges))

    def __repr__(self) -> str:
        return self.__str__()

    def _column_to_int(self, key: Optional[str | int]) -> Optional[int]:
        if key is None:
            return key

        if isinstance(key, int):
            if key not in (0, 1):
                raise IndexError(
                    "Index out of range. Column index must be 0 or 1."
                )
            return key

        if isinstance(key, str):
            if key == self.start_id:
                return 0
            elif key == self.end_id:
                return 1
            else:
                raise KeyError(
                    f'Key "{key}" not one of "{self.start_id}" or'
                    f' "{self.end_id}".'
                )

        return key

    def _parse_key(self, key) -> Tuple[Optional[int], Optional[int]]:
        """Parse the key used in __getitem__ to determine the correct row and
        column indices."""

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

        col_index = self._column_to_int(col_index)

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
        """Find which elements from column are in the set of `test_elements`.
        """
        raise AbstractMethodError(self)

    def isequal(self, other):
        """Determine if two edges are equivalent."""
        raise AbstractMethodError(self)

    def distribution(self, column):
        """Return the distribution of the nodes in column."""

        raise AbstractMethodError(self)

    def to_file(self, edge_name, graph_name, data_dir, format):
        """Save the edge to disc."""
        raise AbstractMethodError(self)

    def as_array(self):
        """Return the edge list as a numpy array"""
        raise AbstractMethodError(self)

    def as_igraph(self):
        """Return the edge as an igraph graph"""
        raise AbstractMethodError(self)

    @property
    def overlap(self):
        """Pairwise number of neighbors nodes have in common."""
        if not hasattr(self, "_overlap"):
            setattr(self, "_overlap", self._calc_overlap())

        return self._overlap

    def _calc_overlap(self):
        raise AbstractMethodError(self)

    def similarity(self, target_publications, method="shortest_path"):
        """
        Calculate similarity between publications based on edge's overlap.

        Parameters
        ----------
        target_publication : ndarray
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


class AbstractMethodError(NotImplementedError):
    """Error for missing required methods in concrete classes."""

    def __init__(self, class_instance):
        self.class_name = type(class_instance).__name__

    def __str__(self):
        return f"Required method not implemented for {self.class_name}"
