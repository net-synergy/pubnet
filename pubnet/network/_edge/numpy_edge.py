"""Implementation of the Edge class storing edges as numpy arrays."""

from ._base import Edge as _AbstractEdge


class Edge(_AbstractEdge):
    def __init__(self, *args):
        super().__init__(*args)
