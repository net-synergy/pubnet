from pubnet import PubNet

data_dir = "example/graphs"
nodes = ("Author", "Publication", "Descriptor", "Chemical")
edges = (
    ("Author", "Publication"),
    ("Descriptor", "Publication"),
    ("Chemical", "Publication"),
)

publications = PubNet(nodes, edges, data_dir)
pubids = publications.publications_containing(
    "Author", "LastName", ["Szymanski", "Smith"], steps=2
)

subnet = publications[pubids]
similarity_metric = subnet["Author", "Publication"].shortest_path
subnet["Author", "Publication"].similarity(
    similarity_metric,
    publications.publications_containing(
        "Author", "LastName", ["Szymanski", "Smith"]
    ),
)
