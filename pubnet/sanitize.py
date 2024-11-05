"""Sanitize provides methods for cleaning and transforming common network data.

Most of these methods perform text based modification to data to either break
data into parts, convert to a different representation, or remove undesired
bits.
"""

__all__ = [
    "orcid",
    "abstract",
    "get_emails_from_affiliation",
    "get_first_initials_from_fore_name",
    "drop_retraction_publications",
    "drop_missing_last_names",
    "duplicates_to_weights",
]
import re

import numpy as np

from pubnet import PubNet


def orcid(net: PubNet, remove_bad: bool = True) -> None:
    """Remove the surrounding URL parts from ORCID strings.

    In real data, ORCID strings are often represented as either a URL with the
    ORCID contain in it or by just the ORCID itself. This method makes all
    ORCID strings only the number part.

    Parameters
    ----------
    net : PubNet
      The publication network to process.
    remove_bad : bool, (default True)
      If true, the ORCIDs will be tested using a checksum. ORCIDs that fail the
      checksum will be removed from the network.

    """

    def check_sum(orcid: str, tally: int = 0) -> bool:
        if len(orcid) == 1:
            return ((12 - (tally % 11)) % 11) == (
                10 if orcid[0] == "x" else int(orcid[0])
            )

        return check_sum(
            orcid[1:],
            tally if orcid[0] == "-" else (tally + int(orcid[0])) * 2,
        )

    orcid_re = re.compile(r"(?P<Identifier>\d{4}-\d{4}-\d{4}-\d{3}[0-9x]$)")
    net.mutate_node_re("Orcid", orcid_re, "Orcid", "Identifier")

    if remove_bad:
        nodes = net.get_node("Orcid")
        edges = net.get_edge("Author", "Orcid")
        good_orcids = np.fromiter(
            (check_sum(orc) for orc in nodes.feature_vector("Identifier")),
            dtype=np.bool_,
        )
        good_edges = np.isin(
            edges["Orcid"], np.arange(nodes.shape[0])[good_orcids]
        )
        edges.set_data(edges[good_edges, :])
        nodes.set_data(nodes[good_orcids])


def drop_retraction_publications(net: PubNet) -> None:
    """Drop publications that retract other publications."""
    net.where(
        "Abstract",
        lambda x: np.fromiter(
            (
                re.search("this retracts the article", a) is None
                for a in x.feature_vector("Abstract")
            ),
            dtype=np.bool_,
        ),
        in_place=True,
    )


def abstract(
    net: PubNet,
    remove_copyrightinformation: bool = True,
) -> None:
    """Strip HTML tags from the abstract.

    This can help with running language models.

    Parameters
    ----------
    net : PubNet
      The publication network to process.
    remove_copyrightinformation : bool, (default True)
      Whether to remove the copyrightinformation from the end of an abstract.

    """

    def clean_abstract(abstract):
        if remove_copyrightinformation:
            abstract = re.sub(
                r"\<copyrightinformation\>.*?\</copyrightinformation\>",
                "",
                abstract,
            )

        return re.sub(r"\<.*?\>", "", abstract).strip()

    net.get_node("Abstract")._data["Abstract"] = np.fromiter(
        (
            clean_abstract(a)
            for a in net.get_node("Abstract").feature_vector("Abstract")
        ),
        dtype=np.object_,
    )


def get_emails_from_affiliation(
    net: PubNet, keep_affiliation: bool = True
) -> None:
    """Find emails within affiliation strings.

    Parameters
    ----------
    net : PubNet
      The publication network to process.
    keep_affiliation : bool, (default True)
      Whether the affiliation string should be kept in the publication network
      (True) or discarded (False).

    """
    email_re = re.compile(
        r"(?P<Address>[a-zA-Z0-9_\.+-]+\s*@\s*(?:\.?[a-zA-Z0-9-]+)+)"
    )
    discard_affiliation = not keep_affiliation
    net.mutate_node_re(
        "Email",
        email_re,
        "Affiliation",
        "Affiliation",
        discard_used=discard_affiliation,
    )


def get_first_initials_from_fore_name(
    net: PubNet, keep_name: bool = True
) -> None:
    """Transform fore name to first initial.

    Parameters
    ----------
    net : PubNet
      The publication network to process.
    keep_name : bool, (default True)
      Whether the full fore names should be kept or discarded.

    """

    def rule(node):
        return np.fromiter(
            (
                (i, name[0])
                for i, name in zip(node.index, node.feature_vector("ForeName"))
                if isinstance(name, str)
            ),
            dtype=np.dtype((object, 2)),
        )

    net.mutate_node(
        "FirstInitial", "ForeName", rule, discard_used=not keep_name
    )


def drop_missing_last_names(net: PubNet) -> None:
    """Remove authors without a last name.

    In the pubmed data set, some none human's are added to the author list such
    as collaboratives. These tend to have different metadata associated with
    them then human authors do and can cause assumptions to fail (such as the
    assumption all author's have a last name).

    Parameters
    ----------
    net : PubNet
      The publication network to process.

    """
    old_root = net.root

    valid_authors = np.unique(net.get_edge("Author", "LastName")["Author"])
    valid_publications = np.unique(
        net.get_edge("Author", "Publication")["Publication"]
    )

    net.select_root("Author")
    net._slice(valid_authors, mutate=True)

    net.select_root("Publication")
    net._slice(valid_publications, mutate=True)

    net.select_root(old_root)


def duplicates_to_weights(net: PubNet, weight_name: str = "weights"):
    """Count the occurrences of an edge and set as weights."""
    for edge in net.edges:
        net.get_edge(edge)._duplicates_to_weights(weight_name)
