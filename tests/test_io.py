import pytest

from pubnet import PubNet

from ._test_fixtures import simple_pubnet


class TestIO:
    @pytest.mark.filterwarnings("ignore:Constructing PubNet object")
    @pytest.mark.parametrize("format", ["tsv", "gzip", "binary"])
    def test_edge_io(self, simple_pubnet, tmp_path, format):
        simple_pubnet.save_graph(
            "edge",
            nodes=None,
            edges=(("Author", "Publication"),),
            data_dir=tmp_path,
            format=format,
        )

        representation = simple_pubnet["Author", "Publication"].representation
        new = PubNet.load_graph(
            "edge",
            data_dir=tmp_path,
            representation=representation,
        )

        assert simple_pubnet["Author", "Publication"].isequal(
            new["Author", "Publication"]
        )

    @pytest.mark.parametrize("format", ["tsv", "gzip", "binary"])
    def test_node_io(self, simple_pubnet, tmp_path, format):
        simple_pubnet.save_graph(
            "node",
            nodes=("Publication", "Author"),
            edges=None,
            data_dir=tmp_path,
            format=format,
        )

        new = PubNet.load_graph("node", data_dir=tmp_path)

        assert simple_pubnet["Publication"].isequal(new["Publication"])
        assert simple_pubnet["Author"].isequal(new["Author"])

    @pytest.mark.parametrize("format", ["tsv", "gzip", "binary"])
    def test_graph_io(self, simple_pubnet, tmp_path, format):
        simple_pubnet.save_graph("graph", data_dir=tmp_path, format=format)

        representation = simple_pubnet["Author", "Publication"].representation
        new = PubNet.load_graph(
            "graph",
            data_dir=tmp_path,
            representation=representation,
        )

        assert simple_pubnet.isequal(new)
