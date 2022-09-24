"""Provides classes for storing graph edges as different representations. """

from .compressed_edge import CompressedEdge
from .numpy_edge import NumpyEdge

__all__ = ["CompressedEdge", "NumpyEdge"]
