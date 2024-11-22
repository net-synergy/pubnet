import pubnet

nodes = ("Author", "Publication", "Descriptor", "Chemical")
edges = (
    ("Author", "Publication"),
    ("Descriptor", "Publication"),
    ("Chemical", "Publication"),
)

publications_np = pubnet.load_graph(
    "pubmed",
    nodes=nodes,
    edges=edges,
    representation="numpy",
)

publications_ig = pubnet.load_graph(
    "pubmed",
    nodes=nodes,
    edges=edges,
    representation="igraph",
)

publications_ig[10627536]
publications_np[10627536]

eig = publications_ig.get_edge("Author", "Publication")
enp = publications_np.get_edge("Author", "Publication")

list(publications_ig.get_edge("Author", "Publication")[1:5, "Author"])
publications_np.get_edge("Author", "Publication").isin(
    "Publication", range(10627536, 10627600)
)
publications_ig.get_edge("Author", "Publication").isin(
    "Publication", range(10627536, 10627600)
)

publications_ig[range(10627536, 11000000)]
publications_np[range(10627536, 11000000)]

last_names = list(
    publications_np.get_node("Author").get_random(n=4, seed=1)["LastName"]
)
subnet = publications_np.containing("Author", "LastName", last_names, steps=2)
subnet.re_root("Author", counts="normalize")
ovr = subnet.overlap({"Chemical", "Publication"}, mutate=False)
ovr.reduce_edges(lambda x, acc: x + acc, "overlap", normalize=True).as_igraph()

author_ovr_np = subnet.get_edge("Author", "Publication").overlap("Author")

subnet = publications_ig.containing("Author", "LastName", last_names, steps=2)
author_ovr_ig = subnet.get_edge("Author", "Publication").overlap("Author")

# sim = subnet["Author", "Publication"].similarity(
#     publications_np.containing("Author", "LastName", last_names)
# )

net = pubnet.load_graph("tmp_test")
net2 = net.copy()
net2.repack()
net2.save_graph(
    name="testing", file_format="binary", overwrite=True, keep_index=True
)


net = pubnet.load_graph(
    "simple_pubnet",
    ("Author", "Publication"),
    (("Publication", "Author"), ("Publication", "Chemical")),
    data_dir="tests/data",
)

subnet = net[1]
subnet.get_node("Author")
