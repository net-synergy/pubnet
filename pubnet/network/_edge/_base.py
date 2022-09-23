"""Abstract base class for storing edges."""

import os
import re

import numpy as np

from ..._similarity import numpy_metrics as _np_similarity


class Edge:
    """Provides a class for storing edges for PubNet.

    For now this is a wrapper around a numpy array with two
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

    def set(self, new_data):
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
            setattr(self, "_overlap", _np_similarity.overlap(self.data))

        return self._overlap

    def shortest_path(self, target_publications, overlap):
        return _np_similarity.shortest_path(target_publications, overlap)

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
