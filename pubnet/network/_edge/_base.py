"""Abstract base class for storing edges."""

import os
import re


class Edge:
    """
    Provides a class for storing edges for `PubNet`.

    In the future it may support weighted edges and directed columns.

    Parameters
    ----------
    data : numpy.ndarray
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

    def __init__(self, data, start_id, end_id, dtype):
        self._data = data
        self.start_id = start_id
        self.end_id = end_id
        self.dtype = dtype
        self.representation = None

        # Weighted not implemented yet
        self.isweighted = False

    def set(self, new_data):
        """Replace the edge's data with a new array."""
        self._data = new_data

    def __str__(self):
        raise AbstractMethodError(self)

    def __repr__(self):
        raise AbstractMethodError(self)

    def __getitem__(self, key):
        raise AbstractMethodError(self)

    def isin(self, column, test_elements):
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

    @property
    def shape(self):
        """Find number of edges."""
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
