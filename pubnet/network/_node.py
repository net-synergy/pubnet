"""Class for storing node data."""

import os

import numpy as np
import pandas as pd

from pubnet.network._utils import (
    node_file_parts,
    node_gen_file_name,
    node_gen_id_label,
    node_id_label_parts,
)

__all__ = ["Node"]


class Node:
    """
    Class for storing node data for PubNet class.

    Provides a wrapper around a panda dataframe adding in information
    about the ID column, which is identified by the special syntax
    f"{name}:ID({namespace})" in order to be compatible with Neo4j
    data.  Here the value `namespace` refers to the node so it's not
    important since we already know the the node.

    This class should primarily be initialized through `PubNet` methods.

    Parameters
    ----------
    data : pandas.DataFrame
        `DataFrame` containing node's features.
    id : str, default "detect"
        The `data` column to use as the ID. If `"detect"`, determine the id
        column based on the above mentioned Neo4j syntax. If the provided
        column name doesn't exist, the column will be generated as
        `1:len(data)`.
    features : "all" or list of str, default "all"
        A list of the columns to keep. If "all" keep all columns.

    Attributes
    ----------
    id : str
        The name of the node id. This is the feature that will be used in edges
        to link to the node.
    features
    columns
    shape
    """

    def __init__(self, data, id=None, name=None, features="all"):
        self._data = data
        self.id = id
        self.name = name
        if data is None:
            self._data = pd.DataFrame()
            self.id = None
            return

        if features != "all":
            assert isinstance(
                features, list
            ), 'Features must be a list or "all"'
            try:
                self._data = self._data[features]
            except KeyError as err:
                raise KeyError(
                    "One or more selected feature not in data.\n\n\tSelected"
                    f" features: {features}\n\tData's features:"
                    f" {self._data.columns}"
                ) from err

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"{self.name} nodes\n\n" + repr(self._data)

    def __getitem__(self, key):
        def genNode(new_data):
            return Node(pd.DataFrame(new_data), self.id, self.name)

        if key is None:
            # When node is empty, self.id == None.
            return genNode(pd.Series(dtype=pd.Float64Dtype))

        if isinstance(key, str):
            return genNode(self._data[key])

        if isinstance(key, int):
            return genNode(self._data[self._data.columns[key]])

        if isinstance(key, tuple):
            assert (
                len(key) <= 2
            ), f"Nodes are 2d; {key} has too many dimensions."
            rows = key[0]
            columns = key[1]
        elif isinstance(key, list) and isinstance(key[0], str):
            columns = key
            rows = slice(None)
        else:
            rows = key
            columns = slice(None)

        if isinstance(columns, int):
            new_data = self._data[self._data.columns[columns]]
        else:
            new_data = self._data

        if isinstance(rows, int):
            return genNode(new_data[rows : (rows + 1)])

        if not isinstance(rows, slice):
            if isinstance(rows, pd.Series):
                is_mask = isinstance(rows.values[0], (bool, np.bool_))
            else:
                is_mask = isinstance(rows[0], (bool, np.bool_))

            if is_mask:
                return genNode(new_data.loc[rows])

        return genNode(new_data[rows])

    def __len__(self):
        return len(self._data)

    def set_data(self, new_data):
        if isinstance(new_data, Node):
            self._data = new_data._data
        elif isinstance(new_data, pd.DataFrame):
            self._data = new_data
        else:
            raise ValueError("New data is not a dataframe")

    @property
    def features(self):
        """A list of all the node's features."""
        return self._data.columns

    @property
    def columns(self):
        """Alias for features to correspond with dataframe terminology."""
        return self.features

    @property
    def shape(self):
        """A tuple with number of rows and number of features."""
        return self._data.shape

    @property
    def index(self):
        return np.asarray(self._data.index)

    def feature_vector(self, name):
        return self._data[name].values

    def get_random(self, n=1, seed=None):
        """
        Sample rows in `Node`.

        Parameters
        ----------
        n : positive int, default 1
            Number of nodes to sample.
        seed : positive int, optional
            Random seed for reproducibility. If not provided, seed is select at
            random.

        Returns
        -------
        nodes : dataframe
            Subset of nodes.
        """

        rng = np.random.default_rng(seed=seed)
        return self._data.loc[rng.integers(0, self._data.shape[0], size=(n,))]

    def isequal(self, node_2):
        """Test if two `Node`s have the same values in all their columns."""

        if not (self.features == node_2.features).all():
            return False

        for feature in self.features:
            if not (
                self.feature_vector(feature) == node_2.feature_vector(feature)
            ).all():
                return False

        return True

    def to_file(
        self,
        data_dir,
        format="tsv",
    ):
        """
        Save the `Node` to file.

        The node will be saved to a graph (a directory in the `data_dir` where
        the graphs nodes and edges are stored).

        Parameters
        ----------
        data_dir : str
            Where the graph is stored.
        format : {"tsv", "gzip", "binary"}, default "tsv"
            the format to save the file as. The binary format uses apache
            feather.

        See also
        --------
        `from_file`
        `pubmed.storage.default_data_dir`
        `pubmed.network.pubnet.save_graph`
        `pubmed.network.pubnet.load_graph`
        """

        ext = {"binary": "feather", "gzip": "tsv.gz", "tsv": "tsv"}
        file_path = node_gen_file_name(self.name, ext[format], data_dir)

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if format == "binary":
            self._data.reset_index().to_feather(file_path)
        else:
            # `to_csv` will infer whether to use gzip based on extension.
            self._data.to_csv(
                file_path,
                sep="\t",
                index_label=node_gen_id_label(self.id, self.name),
            )

    @classmethod
    def from_file(cls, file_name, *args):
        """
        Read a `Node` in from a file

        The node will be saved to a graph (a directory in the `data_dir` where
        the graphs nodes and edges are stored).

        Parameters
        ----------
        file_name : str
           Path to the file containing the node.

        Returns
        -------
        node : Node

        Other Parameters
        ----------------
        *args
            All other args are passed forward to the `Node` class.

        See Also
        --------
        `Node`
        `Node.to_file`
        `from_data`
        `pubmed.storage.default_data_dir`
        `pubmed.network.pubnet.save_graph`
        `pubmed.network.pubnet.load_graph`
        """

        name, ext = node_file_parts(file_name)
        if ext == "feather":
            data = pd.read_feather(file_name)
            data.set_index(data.columns[0], inplace=True)
        else:
            data = pd.read_table(file_name, index_col=0, memory_map=True)
            # Prefer name in header to that in filename if available (but they
            # *should* be the same).
            id, name = node_id_label_parts(data.index.name)
            data.index.name = id

        id = data.index.name

        return cls.from_data(data, id, name, *args)

    @classmethod
    def from_data(cls, data, *args):
        """
        Create a node from a DataFrame.

        Paramaters
        ----------
        Data, DataFrame

        Returns
        -------
        node, Node

        Other Parameters
        ----------------
        *args
            All other args are passed forward to the `Node` class.

        See Also
        --------
        `Node`
        `from_file` : read a `Node` from file.
        `Node.to_file` : save a `Node` to file.
        """

        return Node(data, *args)
