"""Implementation of the Edge class storing edges in a compressed form."""


from ._base import Edge


class CompressedEdge(Edge):
    def __init__(self, *args):
        raise NotImplementedError("Compressed edge not implemented.")
