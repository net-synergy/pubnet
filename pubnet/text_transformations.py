"""Methods for transforming string features to vector representations.

Some of the methods in this module require the language model (LM) dependency
group to be install.
"""

__all__ = ["specter", "string_to_vec"]

from typing import Optional

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, FlaxAutoModel

from pubnet import PubNet
from pubnet.network import _utils


def specter(
    net: PubNet,
    node: str,
    feature: Optional[str] = None,
    batch_size: Optional[int] = None,
    root: Optional[str] = None,
    weight_name: str = "embedding",
    max_length: int = 512,
) -> None:
    """Create specter text embeddings for the given node type.

    Embeddings are created against the network's root (or the provided root).
    I.e. if the network's root is Publication, the embeddings are created as
    edges between publications and their associated text based node.

    Parameters
    ----------
    net : PubNet
      The network to add the embeddings to (modified in place).
    node : str
      The name of a node that has edges with the root.
    feature : str or None
      The name of one of the node's text based features. If None (default) use
      the node name as the feature name.
    batch_size : int or None
      By default (None) transforms all data at once. For a large amount of
      data, this can use up all available memory. If transforming will be done
      in batches of at most that size.
    root : str or None
      If None (default) use the network's default root, otherwise, treat this
      as the network's root.
    weight_name: str
      Name to give the new edge feature (default "embedding").
    max_length: int
      Maximum size of the tokenized abstract. Abstracts larger than this will
      be truncated.

    """
    root = root or net.root
    feature = feature or node

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model = FlaxAutoModel.from_pretrained("allenai/specter")

    # Ensure node's index is equivalent to node's position.
    net.repack(node)
    feature_vec = list(net.get_node(node).feature_vector(feature))
    batch_size = batch_size or len(feature_vec)

    batch_weights = []
    start_idx = 0
    end_idx = min(batch_size, len(feature_vec))
    progress = tqdm(total=len(feature_vec))
    while start_idx < end_idx:
        inputs = tokenizer(
            feature_vec[start_idx:end_idx],
            return_tensors="jax",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        outputs = model(**inputs)
        batch_weights.append(np.asarray(outputs.last_hidden_state[:, 0, :]))
        progress.update(end_idx - start_idx)

        start_idx = end_idx
        end_idx = min(end_idx + batch_size, len(feature_vec))

    progress.close()

    weights = np.concatenate(batch_weights, axis=0)
    edges = net.get_edge(root, node)
    weights = weights[edges[node]]

    new_edge_data = np.zeros((weights.shape[0] * weights.shape[1], 2))
    new_edge_data[:, 0] = edges[root].repeat(weights.shape[1])
    new_edge_data[:, 1] = np.tile(np.arange(weights.shape[1]), len(edges))
    weights = weights.reshape((weights.shape[0] * weights.shape[1],))

    net.add_edge(
        new_edge_data,
        _utils.edge_key(root, node + "_embedding"),
        features={weight_name: weights},
        start_id=root,
        end_id=node,
    )


def string_to_vec(
    net: PubNet,
    node: str,
    feature: Optional[str] = None,
    weight_name: str = "weight",
    root: Optional[str] = None,
):
    """Convert a string to a sparse vector of the characters it contains.

    The ascii code of the element positions represent the ascii characters and
    the weights represent the number of times that letter occured. Order of
    characters is lost, so anagrams are equivalent.

    Parameters
    ----------
    net : PubNet
      The network to modify.
    node : str
      Which node to get the feature from.
    feature : str, None
      The feature to convert (must be text based). If None (default) use the
      feature named after the node.
    weight_name : str (default "weight")
      What to name the resulting edge feature (letter occurrence).
    root : str, None
      Whether to use the network's current root (default) or a different node
      as the root.

    """
    root = root or net.root
    feature = feature or node

    net.repack(node)
    edges = net.get_edge(node, root)
    if edges.start_id == root:
        root_pos = 0
        node_pos = 1
    else:
        root_pos = 1
        node_pos = 0

    feat = net.get_node(node).feature_vector(feature)
    new_edge_data = np.fromiter(
        (
            (edge[root_pos], (ord(letter) - ord("a")))
            for edge in edges
            # Skip NaNs which are type float.
            if feat[edge[node_pos]]
            and not isinstance(feat[edge[node_pos]], float)
            for letter in feat[edge[node_pos]]
            # Ignore spaces and special characters
            if ord(letter) >= ord("a")
        ),
        dtype=np.dtype((edges.dtype, 2)),
    )

    net.add_edge(
        new_edge_data,
        _utils.edge_key(root, node + "_letters"),
        start_id=root,
        end_id=node,
    )
    net.get_edge(root, node + "_letters")._duplicates_to_weights(weight_name)
