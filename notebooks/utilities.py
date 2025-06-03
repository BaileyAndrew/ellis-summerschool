from typing import Optional
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import igraph as ig
import random
import numpy as np

shape_palette = 100*[
   "circle",
   "rect",
   "diamond",
   "triangle",
   "rect",
   "circle",
   "triangle"
]
color_palette = 100*[
   "#FF9A96",
   "#264589",
   "#FF9A96",
   "#264589",
   "#FF9A96",
   "#264589",
   "#FF9A96"
]


def plot_genes(
   name: str,
   adata: ad.AnnData,
   no_edge_weights: bool = True,
   layout: Optional[str] = None,
   graph_type = 'var_nc_gmgm_connectivities',
   color_type = None,
   fig = None,
   ax = None,
   var_names = None
) -> tuple[plt.Figure, plt.Axes]:
    random.seed(1)
    if graph_type in adata.varp:
        graph = adata.varp[graph_type]
        try:
            graph = graph.toarray()
        except:
            pass
        np.fill_diagonal(graph, 0)
    else:
        raise Exception(f"Invalid graph type: {graph_type}")
    how_many = graph.shape[0]

    try:
        # If the graph is empty, this throws an error!
        graph = ig.Graph.Weighted_Adjacency(graph, mode="undirected")
    except:
        graph = ig.Graph.Weighted_Adjacency(graph)
    if var_names is None:
        graph.vs["label"] = adata.var_names
    else:
        graph.vs["label"] = var_names
    graph.vs["label_size"] = 5

    # Get connected components of the graph
    components = graph.components()

    # Color vertex by component
    if color_type is None:
        graph.vs["color"] = [color_palette[i] for i in components.membership]
    else:
        graph.vs["color"] = [color_palette[0] if is_true else color_palette[1] for is_true in adata.var[color_type]]

    # Change  vertex shape by component as well
    graph.vs["shape"] = [shape_palette[i] for i in components.membership]

    graph.es["label"] = [f"{w:.2E}" for w in graph.es["weight"]]
    graph.es["label_size"] = 5

    if layout is None:
        layout = 'circle' if how_many > 15 else 'kk'

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    elif fig is None or ax is None:
        raise Exception("Either both fig and ax should be specified, or neither")
    ig.plot(
        graph,
        vertex_size=40,
        target=ax,
        edge_color="black",
        layout=layout,
        **({} if no_edge_weights else {'edge_label': graph.es["label"]}),
    )
    ax.set_facecolor("white")
    ax.set_title(name)

    if color_type is None:
        ax.set_xlabel("Colored/shaped by connected components")
    else:
        ax.set_xlabel(f"Colored by {color_type}")


    #fig.suptitle(f"Connections of {name}")
    return fig, ax


def plot_cells(
   name: str,
   adata: ad.AnnData,
   no_edge_weights: bool = True,
   layout: Optional[str] = None,
   graph_type ='var_nc_gmgm_connectivities',
   fig = None,
   ax = None,
   legend_loc = 'upper right'
) -> tuple[plt.Figure, plt.Axes]:
    random.seed(1)
    if graph_type in adata.obsp:
        graph = adata.obsp[graph_type]
        try:
            graph = graph.toarray()
        except:
            pass
        np.fill_diagonal(graph, 0)
    else:
        raise Exception(f"Invalid graph type: {graph_type}")
    how_many = graph.shape[0]

    try:
        # If the graph is empty, this throws an error!
        graph = ig.Graph.Weighted_Adjacency(graph, mode="undirected")
    except:
        graph = ig.Graph.Weighted_Adjacency(graph)

    # Get connected components of the graph
    components = graph.components()

    # Map cell types to integers and use igraph’s default palette
    cell_types = adata.obs['cell_type']
    type_to_color = {ctype: adata.uns['cell_type_colors'][i % len(adata.uns['cell_type_colors'])] for i, ctype in enumerate(cell_types.unique())}
    graph.vs["color"] = cell_types.map(type_to_color).tolist()
    graph.vs["edge_color"] = graph.vs["color"]
    #graph.vs["label"] = cell_types
    #graph.vs["label_size"] = 5
    #graph.vs["label"] = cell_types.str[0]
    #graph.vs["label_size"] = 5

    # Change  vertex shape by component as well
    #graph.vs["shape"] = [shape_palette[i] for i in components.membership]
    
    type_to_shape = {ctype: shape_palette[i % len(shape_palette)] for i, ctype in enumerate(cell_types.unique())}
    graph.vs["shape"] = cell_types.map(type_to_shape).tolist()

    graph.es["label"] = [f"{w:.2E}" for w in graph.es["weight"]]
    graph.es["label_size"] = 5

    if layout is None:
        layout = 'circle' if how_many > 15 else 'kk'

    ig.plot(
        graph,
        vertex_size=10,
        target=ax,
        edge_color="black",
        layout=layout,
        **({} if no_edge_weights else {'edge_label': graph.es["label"]}),
    )
    ax.set_facecolor("white")
    ax.set_title(name)
    ax.set_xlabel("Colored/shaped by cell type")

    patches = [
        mpatches.Patch(color=color, label=label)
        for color, label
        in zip(adata.uns['cell_type_colors'], cell_types.unique())
    ]
    ax.legend(handles=patches, loc=legend_loc)
    return fig, ax