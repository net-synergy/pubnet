import time

import numpy as np
import pandas as pd
import pubnet
import pytest

class TimeEdges:
    params = [('igraph',100),('igraph',1000),('igraph',10000),('numpy',100),('numpy',1000),('numpy',10000)]
        
    def setup(self,n):
        global simple_pubnet 
        simple_pubnet = pubnet.from_dir(
            graph_name="graphs",
            nodes=("Author", "Publication"),
            edges=(("Author", "Publication")),
            data_dir="../",
            root="Publication",
            representation=n[0],
        )        
        
        
        
        
    def time_finds_start_id(self):
        for e in self.simple_pubnet.edges:
            self.simple_pubnet[e].start_id == "Publication"
    time_finds_start_id.setup = setup


    def time_finds_end_id(self):
        expected = ["Author", "Chemical"]
        for e in self.simple_pubnet.edges:
            self.simple_pubnet[e].end_id in expected

    def time_overlap(self):
        expected = np.array(
            [
                [1, 2, 2],
                [1, 3, 2],
                [1, 4, 1],
                [1, 5, 1],
                [2, 3, 1],
                [2, 4, 1],
                [2, 5, 1],
                [3, 5, 1],
                [4, 5, 1],
                [4, 6, 1],
                [5, 6, 1],
            ]
        )
        np.array_equal(
            self.simple_pubnet["Author", "Publication"].overlap, expected
        )


class TimeNodes:
    simple_pubnet =  pubnet.from_dir(
            graph_name="graphs",
            nodes=("Author", "Publication", "Descriptor", "Chemical"),
            edges=(("Author", "Publication"),("Descriptor", "Publication"),("Chemical", "Publication")),
            data_dir="../",
            root="Publication",
            representation="numpy",
        )   
    
    author_node =  simple_pubnet["Author"]
    
    
    def time_finds_namespace(self):
        self.author_node.id == "AuthorId"

    def time_slice_column(self):
        self.author_node["LastName"][0] == "Smith"

    def time_slice_columns(self):
        features = ["LastName", "ForeName"]
        (self.author_node[features].columns.values == features).all()

    def time_slice_column_by_index(self):
        self.author_node[0] is self.author_node[self.author_node.features[0]]
        self.author_node[1] is self.author_node[self.author_node.features[1]]

    def time_slice_rows_by_index(self):
        expected = pd.DataFrame(
            {
                "AuthorId": [1, 2],
                "LastName": ["Smith", "Kim"],
                "ForeName": ["John", "John"],
            }
        )
        actual = self.author_node[0:2]

        for feature in self.author_node.features:
            (actual[feature].values == expected[feature].values).all()

    def time_slice_rows_by_mask(self):
        actual = self.author_node[self.author_node["LastName"] == "Smith"]
        expected = pd.DataFrame(
            {
                "AuthorId": [1, 3],
                "LastName": ["Smith", "Smith"],
                "ForeName": ["John", "Jane"],
            }
        )

        for feature in self.author_node.features:
            (actual[feature].values == expected[feature].values).all()

    def time_slice_rows_and_columns(self):
        actual = {
            "Slices": self.author_node[0:2, 0:2],
            "Slice + List": self.author_node[0:2, ["AuthorId", "LastName"]],
            "Mask + Slice": self.author_node[
                self.author_node["ForeName"] == "John", 0:2
            ],
        }
        expected = pd.DataFrame(
            {"AuthorId": [1, 2], "LastName": ["Smith", "Kim"]}
        )

        for node in actual.values():
            for feature in expected.columns:
                (node[feature].values == expected[feature].values).all()


class TimeNetwork:
    

    simple_pubnet =  pubnet.from_dir(
            graph_name="graphs",
            nodes=("Author", "Publication", "Descriptor", "Chemical"),
            edges=(("Author", "Publication"),("Descriptor", "Publication"),("Chemical", "Publication")),
            data_dir="/mnt/c/Users/lenysm/Desktop/pubnet",
            root="Publication",
            representation="numpy",
        )        
    other_pubnet = pubnet.from_dir(
        "other_pubnet",
        ("Chemical",),
        (("Publication", "Chemical"),),
        data_dir="/mnt/c/Users/lenysm/Desktop/pubnet/tests/data",
    )
    
   

    def time_creates_empty_nodes_for_missing_edge_nodes(self):
        len(self.simple_pubnet["Chemical"]) == 0

    def time_filter_to_single_publication_id(self):
        publication_id = 1
        subnet = self.simple_pubnet[publication_id]

        expected_authors = np.asarray([1, 2, 3])

        subnet["Author", "Publication"].shape[0] == 3
        subnet["Chemical", "Publication"].shape[0] == 2
        np.array_equal(
            np.unique(subnet["Author"][subnet["Author"].id]), expected_authors
        )
        np.array_equal(
            np.unique(subnet["Publication"][subnet["Publication"].id]),
            np.asarray([publication_id]),
        )

    def time_filter_to_publicaiton_ids(self):
        publication_ids = np.asarray([1, 2], dtype=self.simple_pubnet.id_dtype)
        subnet = self.simple_pubnet[publication_ids]

        expected_authors = np.asarray([1, 2, 3])

        subnet["Author", "Publication"].shape[0] == 5
        subnet["Chemical", "Publication"].shape[0] == 4
        np.array_equal(
            np.unique(subnet["Author"][subnet["Author"].id]), expected_authors
        )
        np.array_equal(
            np.unique(subnet["Publication"][subnet["Publication"].id]),
            publication_ids,
        )

    def time_filter_twice(self):
        publication_ids_1 = np.asarray([4, 6], dtype=self.simple_pubnet.id_dtype)
        publication_ids_2 = 4

        subnet_1 = self.simple_pubnet[publication_ids_1]
        subsubnet = subnet_1[publication_ids_2]
        subnet_2 = self.simple_pubnet[publication_ids_2]

        subsubnet.isequal(subnet_2)

    def time_filter_to_author(self):
        subnet = self.simple_pubnet.containing("Author", "LastName", "Smith")
        expected_publication_ids = np.asarray([1, 2, 3, 5])

        np.array_equal(
            np.unique(subnet["Author", "Publication"]["Publication"]),
            expected_publication_ids,
        )
        np.array_equal(
            np.unique(subnet["Chemical", "Publication"]["Publication"]),
            expected_publication_ids,
        )

        np.array_equal(
            np.unique(subnet["Publication"][subnet["Publication"].id]),
            expected_publication_ids,
        )

    def time_filter_to_author_multiple_steps(self):
        publication_ids = self.simple_pubnet.ids_containing(
            "Author", "LastName", "Smith", steps=2
        )
        subnet = self.simple_pubnet[publication_ids]
        expected_publication_ids = np.asarray([1, 2, 3, 4, 5, 6])

        np.array_equal(
            np.unique(subnet["Author", "Publication"]["Publication"]),
            expected_publication_ids,
        )
        np.array_equal(
            np.unique(subnet["Chemical", "Publication"]["Publication"]),
            expected_publication_ids,
        )

    def time_update(self):
        expected_nodes = set(self.simple_pubnet.nodes).union(
            set(self.other_pubnet.nodes)
        )

        expected_edges = set(self.simple_pubnet.edges).union(
            set(self.other_pubnet.edges)
        )

        self.other_pubnet.drop("Publication")
        self.simple_pubnet.update(self.other_pubnet)
        set(self.simple_pubnet.nodes) == expected_nodes
        set(self.simple_pubnet.edges) == expected_edges
        # other's Chemical-Publication edge shadowed simple_pubnet's.
        self.simple_pubnet["Chemical", "Publication"].isequal(
            self.other_pubnet["Chemical", "Publication"]
        )