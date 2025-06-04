from typing import Optional, Literal
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import igraph as ig
import random
import numpy as np
import pandas as pd

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

# This is a helper function to generate our results for each cluster; run it, but don't worry too much about understanding the internals
def plot_graph(
    adata: ad.AnnData,
    module: str,
    key: str,
    pathway_df: pd.DataFrame,
    graph_type: Literal["glasso", "gmgm"],
    obsp_or_varp: Literal["obsp", "varp"] = "obsp",
    vertex_names: Optional[str] = None,
    top_genes: int = 25,
    top_pathways: int = 10,
    display_centrality: Literal["degree", "harmonic", "eigenvector_centrality", "betweenness"] = "degree",
    save: bool = True,
) -> tuple[tuple[plt.Figure, plt.Axes], ig.Graph]:
    leiden_string = f"leiden_{graph_type}"
    key = f"{key}_connectivities"
    # Create the graph for the specific module
    if obsp_or_varp == "obsp":
        module_idx = adata.obs[leiden_string] == module
        name_source = adata[module_idx].obs_names
        id = adata[module_idx].obs_names
        if vertex_names is not None:
            name_source = adata[module_idx].obs[vertex_names]
        full_graph = adata[module_idx].obsp[key]
    elif obsp_or_varp == "varp":
        module_idx = adata.var[leiden_string] == module
        name_source = adata[module_idx].var_names
        id = adata[module_idx].var_names
        if vertex_names is not None:
            name_source = adata[module_idx].var[vertex_names]
        full_graph = adata[module_idx].varp[key]
    else:
        raise ValueError("obsp_or_varp must be either 'obsp' or 'varp'")
    
    full_graph = ig.Graph.Weighted_Adjacency(full_graph.toarray(), mode="undirected")
    full_graph.vs["label"] = name_source
    full_graph.vs["label_size"] = 10

    # Measure how central each gene is
    centrality_measures = pd.DataFrame(
            index=id,
            data={
                "names": full_graph.vs["label"],
                "degree": full_graph.degree(),
                "harmonic": full_graph.harmonic_centrality(),
                "eigenvector_centrality": full_graph.eigenvector_centrality(),
                "betweenness": full_graph.betweenness(),
            }
    )

    full_graph.vs["degree"] = centrality_measures["degree"]
    full_graph.vs["harmonic"] = centrality_measures["harmonic"]
    full_graph.vs["eigenvector_centrality"] = centrality_measures["eigenvector_centrality"]
    full_graph.vs["betweenness"] = centrality_measures["betweenness"]

    centrality_measures = centrality_measures.sort_values(display_centrality, ascending=False)
    ranked = centrality_measures[display_centrality]

    # Subset the graph to just the most central genes
    name_source = centrality_measures.head(top_genes)["names"]
    idxs = centrality_measures.head(top_genes).index
    if obsp_or_varp == "obsp":
        graph = adata[idxs].obsp[key]
    elif obsp_or_varp == "varp":
        graph = adata[idxs].varp[key]
    else:
        raise ValueError("obsp_or_varp must be either 'obsp' or 'varp'")
    

    # Plot and save the graph
    graph = ig.Graph.Weighted_Adjacency(graph.toarray(), mode="undirected")
    graph.vs["label"] = name_source
    graph.vs["label_size"] = 7
    fig, ax = plt.subplots(figsize=(7, 7))
    ig.plot(
        graph,
        vertex_size=50,
        target=ax,
        edge_color="black",
    )
    ax.set_facecolor("grey")
    ax.set_title(f"Module {module}; top {top_genes} genes by {display_centrality}")
    if save:
        # Check if m{module} exists; if not, create it
        if not os.path.exists(f"../results/{graph_type}/m{module}"):
            os.makedirs(f"../results/{graph_type}/m{module}")
        else:
            # If it already exists, remove the old contents
            shutil.rmtree(f"../results/{graph_type}/m{module}")
            os.makedirs(f"../results/{graph_type}/m{module}")
       
        plt.savefig(f"../results/{graph_type}/m{module}/graph.png")

        # Save the centrality measures
        centrality_measures.to_csv(f"../results/{graph_type}/m{module}/centrality_measures.csv")

        # Save the list of top N genes
        centrality_measures["names"].head(top_genes).to_csv(
            f"../results/{graph_type}/m{module}/top_genes.csv",
            index=False,
            header=False,
            sep=",",
        )

        # Select the top 10 pathways for relevant sources
        module_df = pathway_df[pathway_df["query"] == f"Module {module}"]
        res = pd.concat([
            module_df[module_df["source"] == "GO:BP"].head(top_pathways)[["source", "native", "name", "p_value"]],
            module_df[module_df["source"] == "GO:CC"].head(top_pathways)[["source", "native", "name", "p_value"]],
            module_df[module_df["source"] == "GO:MF"].head(top_pathways)[["source", "native", "name", "p_value"]],
            module_df[module_df["source"] == "KEGG"].head(top_pathways)[["source", "native", "name", "p_value"]],
        ])
        res.to_csv(f"../results/{graph_type}/m{module}/top_pathways.csv", index=False)

    return (fig, ax), graph