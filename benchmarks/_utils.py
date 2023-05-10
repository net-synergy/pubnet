import os

from pubnet import from_dir


def simple_pubnet(representation, n_nodes):
    data_dir = os.path.dirname(__file__)
    net = from_dir(
        graph_name="graphs",
        nodes=("Author", "Publication", "Descriptor", "Chemical"),
        edges=(
            ("Author", "Publication"),
            ("Descriptor", "Publication"),
            ("Chemical", "Publication"),
        ),
        data_dir=data_dir,
        root="Publication",
        representation=representation,
    )
    random_nodes = net["Author"].get_random(n_nodes)
    return net.containing("Author", "AuthorId", random_nodes["AuthorId"])
