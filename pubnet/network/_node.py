"""Class for storing node data."""

import os
import re

import numpy as np
import pandas as pd


class Node:
    """Class for storing node data for PubNet class.

    Provides a wrapper around a panda dataframe adding in information
    about the ID column, which is identified by the special syntax
    f"{name}:ID({namespace})" in order to be compatible with Neo4j
    data.  Here the value `namespace` refers to the node so it's not
    important since we already know the the node.
    """

    _id_re = re.compile("(.*):ID\\(.*?\\)")

    def __init__(self, node, data_dir="."):
        if node is None:
            self._data = pd.DataFrame()
            self.id = None
            return

        self._data = pd.read_csv(
            os.path.join(data_dir, f"{node}_nodes.tsv"),
            delimiter="\t",
        )
        id_column = list(
            filter(
                lambda x: x is not None,
                [self._id_re.search(name) for name in self._data.columns],
            )
        )[0]
        old_id = id_column.group().replace("(", "\\(").replace(")", "\\)")
        self.id = id_column.groups()[0]
        self._data.columns = self._data.columns.str.replace(
            old_id, self.id, regex=True
        )

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    def __getitem__(self, key):
        if key is None:
            # When node is empty, self.id == None.
            return pd.Series(dtype=pd.Float64Dtype)

        if isinstance(key, str):
            return self._data[key]

        if isinstance(key, int):
            return self._data[self._data.columns[key]]

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

        if not isinstance(rows, slice):
            if (len(rows) > 1) and isinstance(rows[0], (bool, np.bool8)):
                return self._data[columns].loc[rows]

        return self._data[columns][rows]

    def __len__(self):
        return len(self._data)

    def set(self, new_data):
        self._data = new_data

    @property
    def features(self):
        return self._data.columns

    @property
    def columns(self):
        return self.features

    @property
    def shape(self):
        return self._data.shape

    def get_random(self, n=1, seed=None):
        """Sample n nodes.

        Arguments
        ---------
        n : positive int, number of nodes to sample (default 1).
        seed : positive int, random seed for reproducibility (default None).
            If None seed is select at random.

        Returns
        -------
        nodes : dataframe, subset of nodes.
        """

        rng = np.random.default_rng(seed=seed)
        return self._data.loc[rng.integers(0, self._data.shape[0], size=(n,))]
