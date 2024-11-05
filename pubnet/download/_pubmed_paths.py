__all__ = [
    "expand_structure_dict",
    "node_list_to_file_names",
    "available_paths",
    "is_node_list",
    "sterilize_node_list",
]

import re

from pubnet.network._utils import (
    edge_gen_file_name,
    edge_key,
    node_gen_file_name,
)

_KNOWN_PATHS = {
    "Pmid": "/PubmedArticle/MedlineCitation/PMID",
    "Doi": "/PubmedArticle/PubmedData/ArticleIdList/ArticleId/[@IdType='doi']",
    "Pii": "/PubmedArticle/PubmedData/ArticleIdList/ArticleId/[@IdType='pii']",
    "Pmc": "/PubmedArticle/PubmedData/ArticleIdList/ArticleId/[@IdType='pmc']",
    "Mid": "/PubmedArticle/PubmedData/ArticleIdList/ArticleId/[@IdType='mid']",
    "Language": "/PubmedArticle/MedlineCitation/Article/Language",
    "Date": "/PubmedArticle/PubmedData/History/PubMedPubDate/Year",
    "Journal": "/PubmedArticle/MedlineCitation/Article/Journal/Title",
    "Author": "/PubmedArticle/MedlineCitation/Article/AuthorList",
    "Grant": "/PubmedArticle/MedlineCitation/Article/GrantList",
    "Chemical": "/PubmedArticle/MedlineCitation/ChemicalList/Chemical/NameOfSubstance/@UI",
    "Qualifier": "/PubmedArticle/MedlineCitation/MeshHeadingList/MeshHeading/QualifierName/@UI",
    "Descriptor": "/PubmedArticle/MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName/@UI",
    "Keyword": "/PubmedArticle/MedlineCitation/KeywordList/Keyword",
    "Reference": "/PubmedArticle/PubmedData/ReferenceList/Reference/ArticleIdList/ArticleId/[@IdType='pubmed']",
    "Title": "/PubmedArticle/MedlineCitation/Article/ArticleTitle",
    "Abstract": "/PubmedArticle/MedlineCitation/Article/Abstract",
}

_KNOWN_SUBPATHS = {
    "Author": {
        "LastName": "/Author/LastName",
        "ForeName": "/Author/ForeName",
        "Initials": "/Author/Initials",
        "Affiliation": "/Author/AffiliationInfo/Affiliation",
        "Orcid": "/Author/Identifier/[@Source='ORCID']",
    },
    "Grant": {
        "Id": "/Grant/GrantID",
        "Acronym": "/Grant/Acronym",
        "Agency": "/Grant/Agency",
        "Country": "/Grant/Country",
    },
}


def snake_case(word: str) -> str:
    return "".join(("_" + l.lower() if l.isupper() else l for l in word))[1:]


def available_paths() -> list[str]:
    return [snake_case(path) for path in _KNOWN_PATHS]


def node_list_to_file_names(
    node_list, key_node: str, data_dir: str = "", prefix: str = ""
) -> list[str]:
    """Return a list of all files needed to represent the graph."""
    # FIXME: Node list contains dictionaries, which are not handled.
    # Additionally I don't think "root" or "key" exist anymore. So need another
    # method to test for "root".
    files = [
        node_gen_file_name(f + "_" + prefix if prefix else f, "tsv", data_dir)
        for f in node_list
        if f not in ("root", "key")
    ] + [
        edge_gen_file_name(edge_key(f, key_node), "tsv", data_dir)[0]
        for f in node_list
        if f not in ("root", "key", key_node)
    ]

    for el in node_list:
        if isinstance(el, dict):
            files += node_list_to_file_names(
                el["value"], el["name"], data_dir, prefix=el["name"]
            )

    return files


def _titlecase(name):
    if name.islower():
        return name.title().replace("_", "")

    return name


def sterilize_node_list(node_list):
    for i, el in enumerate(node_list):
        if isinstance(el, str):
            node_list[i] = _titlecase(el)
        else:
            node_list[i]["name"] = _titlecase(el["name"])
            if isinstance(el["value"], str):
                node_list[i]["value"] = _titlecase(el["value"])
            else:
                node_list[i]["value"] = sterilize_node_list(el["value"])

    return node_list


def expand_structure_dict(struct):
    """Expand a structure dictionary using predefined paths.

    Create a full path structure for read_xml from a list of names.
    """
    if not _is_path_structure(struct):
        struct = _node_list_to_path_structure(struct)

    for key, val in struct.items():
        if isinstance(val, str) and val.split("/")[0] in _KNOWN_PATHS:
            path = val.split("/")[0]
            struct[key] = val.replace(path, _KNOWN_PATHS[path])

        if isinstance(val, dict):
            struct[key] = expand_structure_dict(val)

    root_path_is_list = re.match("(^.*List)(.*$)", struct["root"])
    if root_path_is_list:
        struct["root"] = root_path_is_list.groups()[0]
        for key, val in struct.items():
            if key == "root":
                continue

            struct[key] = root_path_is_list.groups()[1] + val

    return struct


def _is_path_structure(obj) -> bool:
    if isinstance(obj, dict) and "root" in obj and "key" in obj:
        return True

    return False


def _node_list_to_path_structure_i(node_list, root, key, known_paths):
    struct = {"root": root, "key": key}
    for node in node_list:
        if isinstance(node, dict):
            if isinstance(node["value"], str):
                try:
                    # Just use a different name for known path.
                    struct[node["name"]] = known_paths[node["value"]]
                except KeyError:
                    # Name a user-defined path
                    struct[node["name"]] = node["value"]
                continue

            group_key = "/condensed"
            if "grouping" in node and node["grouping"] == "relational":
                group_key = "/auto_index"

            if known_paths[node["name"]].endswith("List"):
                group_key = "/" + node["name"] + group_key

            struct[node["name"]] = _node_list_to_path_structure_i(
                node["value"],
                known_paths[node["name"]],
                group_key,
                _KNOWN_SUBPATHS[node["name"]],
            )
        elif isinstance(node, str):
            if node.lower() in ("condensed", "relational"):
                continue

            try:
                struct[node] = known_paths[node]
            except KeyError:
                raise KeyError(f"{node} is not a known path.")
        else:
            raise TypeError("Got a node value that was not a str or dict.")

    return struct


def _node_list_to_path_structure(node_list):
    root = "//PubmedArticleSet"
    key = _KNOWN_PATHS["Pmid"]
    return _node_list_to_path_structure_i(node_list, root, key, _KNOWN_PATHS)


def is_node_list(obj: list) -> bool:
    """Determine if object conforms to node_list protocol.

    To be a node_list object:
    - Must be a list.
    - Each element of the list must be either a string or a dictionary.
    - If an element is a dictionary it must contain keys "name" and "value".
    - If element is a dictionary it can optionally contain the key "grouping".
    - Any other dictionary keys are allowed but will be ignored.
    """
    if not isinstance(obj, list):
        return False

    for el in obj:
        if not isinstance(el, dict | str):
            return False

        if isinstance(el, dict) and ("name" not in el or "value" not in el):
            return False

    return True
