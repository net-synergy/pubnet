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
    "Author", "LastName", "Szymanski"
)

new_publications = publications[pubids]
