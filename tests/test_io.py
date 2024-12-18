import pytest

from pubnet import PubNet

from ._test_fixtures import simple_pubnet


class TestIO:
    @pytest.mark.filterwarnings("ignore:Constructing PubNet object")
    @pytest.mark.parametrize("file_format", ["tsv", "gzip", "binary"])
    def test_edge_io(self, simple_pubnet, tmp_path, file_format):
        simple_pubnet.save_graph(
            "edge",
            nodes=None,
            edges=(("Author", "Publication"),),
            data_dir=tmp_path,
            file_format=file_format,
        )

        representation = simple_pubnet.get_edge(
            "Author", "Publication"
        ).representation
        new = PubNet.load_graph(
            "edge",
            data_dir=tmp_path,
            representation=representation,
        )

        assert simple_pubnet.get_edge("Author", "Publication").isequal(
            new.get_edge("Author", "Publication")
        )

    @pytest.mark.parametrize("file_format", ["tsv", "gzip", "binary"])
    def test_node_io(self, simple_pubnet, tmp_path, file_format):
        simple_pubnet.save_graph(
            "node",
            nodes=("Publication", "Author"),
            edges=None,
            data_dir=tmp_path,
            file_format=file_format,
        )

        new = PubNet.load_graph("node", data_dir=tmp_path)

        assert simple_pubnet.get_node("Publication").isequal(
            new.get_node("Publication")
        )
        assert simple_pubnet.get_node("Author").isequal(new.get_node("Author"))

    @pytest.mark.parametrize("file_format", ["tsv", "gzip", "binary"])
    def test_graph_io(self, simple_pubnet, tmp_path, file_format):
        simple_pubnet.save_graph(
            "graph", data_dir=tmp_path, file_format=file_format
        )

        representation = simple_pubnet.get_edge(
            "Author", "Publication"
        ).representation
        new = PubNet.load_graph(
            "graph",
            data_dir=tmp_path,
            representation=representation,
        )

        assert simple_pubnet.isequal(new)

    @pytest.mark.parametrize("file_format", ["tsv", "gzip", "binary"])
    def test_graph_reads_writes_features(
        self, simple_pubnet, tmp_path, file_format
    ):
        simple_pubnet.add_edge(
            simple_pubnet.get_edge("Author", "Publication").overlap(
                "Publication"
            )
        )
        simple_pubnet.save_graph(
            "graph",
            edges=(("Publication", "AuthorOverlap"),),
            data_dir=tmp_path,
            file_format=file_format,
        )

        representation = simple_pubnet.get_edge(
            "Author", "Publication"
        ).representation
        new = PubNet.load_graph(
            "graph", data_dir=tmp_path, representation=representation
        )

        assert simple_pubnet.get_edge("Publication", "AuthorOverlap").isequal(
            new.get_edge("Publication", "AuthorOverlap")
        )
