"""Implementation of the Edge class storing edges as numpy arrays."""

import numpy as np

from ..._similarity import numpy_metrics as _np_similarity
from ._base import Edge as _AbstractEdge


class Edge(_AbstractEdge):
    """An impelmentation of the Edge class that stores edges as numpy arrays.

    Uses arrays to list the non-zero edges in a sparse matrix form.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self._data = np.genfromtxt(
            self._file_path,
            # All edge values should be integer IDs.
            dtype=np.int64,
            skip_header=1,
        )

    def __str__(self):
        return (
            f"col 0: {self.start_id}\ncol 1: {self.end_id}\n{str(self._data)}"
        )

    def __repr__(self):
        return (
            f"col 0: {self.start_id}\ncol 1: {self.end_id}\n{repr(self._data)}"
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
            return self._data[:, key]

        return self._data[key]

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
