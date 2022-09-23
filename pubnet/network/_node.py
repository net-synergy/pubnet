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

    def __init__(self, node, data_dir):
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
        return self._data[key]

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
