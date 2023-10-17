"""Methods for creating PubNet networks from online sources.

Networks are stored on disk. If they have already been collected, the local
copies will be used.
"""

from .pubmed import from_pubmed

__all__ = ["from_pubmed"]
