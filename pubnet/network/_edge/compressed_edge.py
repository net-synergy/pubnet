"""Implementation of the Edge class storing edges in a compressed form."""


from ._base import Edge as _AbstractEdge


class Edge(_AbstractEdge):
    def __init__(self, *args):
        raise NotImplementedError("Compressed edge not implemented.")
