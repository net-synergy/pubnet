import os
import tempfile

from pubnet import PubNet


def simple_pubnet(representation, n_nodes):
    data_dir = os.path.dirname(__file__)
    net = PubNet.load_graph(
        "graphs",
        nodes=("Author", "Publication", "Descriptor", "Chemical"),
        edges=(
            ("Author", "Publication"),
            ("Descriptor", "Publication"),
            ("Chemical", "Publication"),
        ),
        root="Publication",
        data_dir=data_dir,
        representation=representation,
    )
    random_nodes = net["Author"].get_random(n_nodes)
    return net.containing("Author", "AuthorId", random_nodes["AuthorId"])


def mktmpdir(*args):
    return tempfile.TemporaryDirectory(prefix="pubnet_benchmarks")
