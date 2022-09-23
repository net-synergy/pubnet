"""Abstract base class for storing edges."""

import os
import re


class Edge:
    """Provides a class for storing edges for PubNet.

    Reads the data in from a file. The file should be in the form
    f"{edge[0]}_{edge[1]}_edges.tsv, where the order the node types
    are given in the edge argument is not important.

    As with the Node class it expects ID columns to be in Neo4j format
    f":START_ID({namespace})" and f":END_ID({namespace})". Start and
    end will be important only if the graph is directed. The
    `namespace` value provides the name of the node and will link to
    that node's ID column.

    In the future it may support weighted edges and directed columns.

    Arguments
    ---------
    edge : a list of two node types.
    data_dir : the directory the edge file is in.

    Attributes
    ----------
    start_id : the node type in column 0.
    end_id : the node type in column 1.
    """

    _start_id_re = re.compile(":START_ID\\((.*?)\\)")
    _end_id_re = re.compile(":END_ID\\((.*?)\\)")

    def __init__(self, edge, data_dir):
        assert len(edge) == 2, "Edge is defined by exactly two nodes."

        self._file_path = _edge_path(edge[0], edge[1], data_dir)

        with open(self._file_path, "r") as f:
            header_line = f.readline()

        self.start_id = self._start_id_re.search(header_line).groups()[0]
        self.end_id = self._end_id_re.search(header_line).groups()[0]
        self._data = None
        # Weighted edges implemented yet.
        self.isweighted = False

    def set(self, new_data):
        self._data = new_data

    def __str__(self):
        raise NotImplementedError(self._required_msg)

    def __repr__(self):
        raise NotImplementedError(self._required_msg)

    def __getitem__(self, key):
        raise NotImplementedError(self._required_msg)

    def isin(self, column, test_elements):
        """Find which elements from column are in the set of test_elements."""
        raise NotImplementedError(self._required_msg)

    @property
    def _required_msg(self):
        return f"Required method not implemented for Edge \
representation {self.representation}"

    @property
    def representation(self):
        """Name of Edge subclass."""
        try:
            return self._representation
        except AttributeError:
            raise AttributeError(
                f"{self.__class__} does not initialize \
'_representation' attribute."
            )

    @property
    def shape(self):
        """Find number of edges."""
        raise NotImplementedError(self._required_msg)

    @property
    def overlap(self):
        """Pairwise number of neighbors nodes have in common."""
        if not hasattr(self, "_overlap"):
            setattr(self, "_overlap", self._calc_overlap())

        return self._overlap

    def _calc_overlap(self):
        raise NotImplementedError(self._required_msg)

    def similarity(self, target_publications, method="shortest_path"):
        """Calculate similarity between publications based on edge's overlap.

        Arguments
        ---------
        target_publication : array, an array of publications to return
            similarity between which must be a subset of all edges in
            `self.overlap`.
        method : {'shortest_path'}, the method to use for
            calculating similarity.

        Returns
        -------
        similarity : a 3 column 2d array, listing the similarity (3rd
            column) between all pairs of publications (1st--2nd
            column) in target_publications. Only non-zero similarities
            are listed.
        """

        all_methods = {
            "shortest_path": self._shortest_path,
            "pagerank": self._pagerank,
        }

        try:
            return all_methods[method](target_publications)
        except NotImplementedError:
            raise NotImplementedError(
                f"Similarity method '{method}' not implemented for Edge \
representation '{self.representation}'"
            )

    def _shortest_path(self, target_publications):
        raise NotImplementedError(self._required_msg)

    def _pagerank(self, target_publications):
        raise NotImplementedError(self._required_msg)


def _edge_path(n1, n2, data_dir):
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

    if os.path.exists(edge_file_path(n1, n2)):
        file_path = edge_file_path(n1, n2)
    elif os.path.exists(edge_file_path(n2, n1)):
        file_path = edge_file_path(n2, n1)
    else:
        raise FileNotFoundError(
            f"No edge file for edges {n1}, {n2} found in \
{data_dir}.\
\n\nExpceted either file '{edge_file_path(n1, n2)}' or \
'{edge_file_path(n2, n1)}'"
        )

    return file_path
