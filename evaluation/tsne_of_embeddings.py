import argparse  # Import argparse
import itertools
import json
import os
import sys
from functools import partial

import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import umap
from openTSNE import TSNE
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

sys.path.append(".")
sys.path.append("..")
from data.wordnet_tree import WNTree

with open("data/misc_dataset_files/imagenet_labels.txt", "r") as f:
    lines = f.readlines()

class_names = {l.strip().split(" ")[0]: l.strip().split(" ")[-1] for l in lines}

in_tree = WNTree.load("data/misc_dataset_files/imagenet21k+1k_masses_tree.json")


def _load_embedding(filename, folder_path):
    filepath = os.path.join(folder_path, filename)
    class_id = filename.split("_")[0]
    try:
        embedding = np.load(filepath)
        if len(embedding.shape) > 1:
            embedding = embedding[0]
        # print(f"embedding shape: {embedding.shape}")
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Extract class ID from filename (part before the first '_')
        return embedding, filename, class_id

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None, filename, class_id


def plot_interactive_tsne_with_args(
    folder_path,
    output_html_path=None,
    max_embeddings=10000,
    random_seed=42,
    pca_components=None,
    umap_n_neighbors=15,
    umap_metric="euclidean",
    umap_min_dist=0.1,
    mapper="umap",
):
    """Plot embeddings as t-SNE.

    Generates an interactive t-SNE plot using Plotly from high-dimensional
    embeddings, colored from a continuous distribution (e.g., Viridis) and
    marked by class. The class ID is extracted from the filename
    (part before the first '_'). Includes optional PCA preprocessing.

    Args:
        folder_path (str): The path to the folder containing .emb.npy files.
        output_html_path (str, optional): Path to save the interactive HTML figure.
                                          If None, the plot is shown in a browser.
        max_embeddings (int): Maximum number of embeddings to use for plotting.
                              If more embeddings are found, a random subset is taken.
        random_seed (int): Seed for reproducibility when sampling, PCA, and t-SNE.
        pca_components (int, optional): Number of PCA components to reduce
                                        embedding dimensionality before t-SNE.
                                        If None or greater than embedding dim, PCA is skipped.
    """
    print(f"Loading embeddings from {folder_path}")
    embedding_files = [f for f in os.listdir(folder_path) if f.endswith(".emb.npy")]
    # print(len(embedding_files), len(os.listdir(folder_path)))
    # print(list(os.listdir(folder_path))[:100])
    # exit()

    if len(embedding_files) > max_embeddings:
        print(f"Found {len(embedding_files)} embeddings. Reducing to {max_embeddings}")
        np.random.seed(random_seed)
        indices = np.random.choice(len(embedding_files), max_embeddings, replace=False)
        embedding_files = [embedding_files[idx] for idx in indices]

    # Load embeddings and extract class IDs
    # for filename in tqdm(embedding_files, desc="Load embedding files"):
    #     filepath = os.path.join(folder_path, filename)

    res = process_map(
        partial(_load_embedding, folder_path=folder_path),
        embedding_files,
        max_workers=20,
        chunksize=25,
        desc="Load embedding files",
    )
    res = [(emb, name, cls_id) for emb, name, cls_id in res if emb is not None]
    all_embeddings_list, all_file_names, all_class_ids = zip(*res)

    if not all_embeddings_list:
        print(f"No .emb.npy files found in {folder_path}.")
        return

    concatenated_embeddings = np.vstack(all_embeddings_list)
    print(f"All embeddings: {concatenated_embeddings.shape}")

    embeddings_to_plot = concatenated_embeddings
    file_names_to_plot = all_file_names
    class_ids_to_plot = all_class_ids

    # Optional PCA preprocessing
    original_dim = embeddings_to_plot.shape[1]
    if pca_components is not None and pca_components < original_dim:
        print(f"Performing PCA to reduce dimensionality from {original_dim} to {pca_components} components.")
        pca = PCA(n_components=pca_components, random_state=random_seed, svd_solver="randomized")
        embeddings_processed = pca.fit_transform(embeddings_to_plot)
    else:
        embeddings_processed = embeddings_to_plot
        if pca_components is not None and pca_components >= original_dim:
            print(
                f"PCA components ({pca_components}) are not less than embedding dimensionality ({original_dim})."
                " Skipping PCA."
            )
        else:
            print(f"No PCA specified. Using original embedding dimensionality {original_dim}.")

    if mapper == "umap":
        # Perform UMAP dimensionality reduction
        try:
            print(
                f"Running UMAP on {embeddings_processed.shape[0]} embeddings with n_neighbors={umap_n_neighbors},"
                f" min_dist={umap_min_dist}, metric='{umap_metric}'..."
            )
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
            )
            results = reducer.fit_transform(embeddings_processed)
            print("UMAP computation complete.")
        except Exception as e:
            print(f"Error during UMAP computation: {e}")
            return
    elif mapper == "tsne":
        print(f"Running TSNE on {embeddings_processed.shape[0]} embeddings")
        tsne = TSNE(n_components=2, n_jobs=20, perplexity=30, random_state=random_seed)
        results = tsne.fit(embeddings_processed)
        print("TSNE computation complete")
    else:
        raise NotImplementedError(f"Mapper {mapper} not implemented")

    # Map class IDs to unique colors from a continuous colormap and unique markers
    parent_ids_to_plot = [
        15388,  # animal
        17222,  # plant
        12992868,  # fungus
        3575240,  # instrumentality
        4341686,  # structure
        3076708,  # commodity
        13086908,  # plant part
        9287968,  # geological formation
        3309808,  # fabric
    ]

    if output_html_path is not None:
        out_html_bare_path = ".".join(output_html_path.split(".")[:-1])
        umap_path = out_html_bare_path + f".{args.mapper}.npy"
        meta_path = out_html_bare_path + ".metadata.json"
        print(f"saving {args.mapper} results at {umap_path} and metadata at {meta_path}")
        np.save(umap_path, results)
        with open(meta_path, "w") as f:
            json.dump(dict(files=file_names_to_plot, classes=class_ids_to_plot, parent_classes=parent_ids_to_plot), f)

    unique_class_ids = sorted(list(set(parent_ids_to_plot)))
    num_unique_classes = len(unique_class_ids)

    colormap = cm.viridis

    markers = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
        "star-diamond",
        "hourglass",
        "bowtie",
    ]
    marker_cycle = itertools.cycle(markers)

    class_marker_map = {class_id: next(marker_cycle) for class_id in unique_class_ids}

    class_color_map = {}
    for i, class_id in enumerate(unique_class_ids):
        norm_value = i / (num_unique_classes - 1) if num_unique_classes > 1 else 0.5
        rgba_color = colormap(norm_value)
        rgb_color = f"rgb({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)})"
        class_color_map[class_id] = rgb_color

    # Create a Plotly figure
    fig = go.Figure()

    # Add a trace for each class
    for class_id in tqdm(unique_class_ids, desc="plotting classes"):
        subtree = in_tree.subtree(class_id)
        class_indices = [i for i, cid in enumerate(class_ids_to_plot) if cid in subtree]
        parent_name = in_tree[class_id].print_name
        tqdm.write(f"class {parent_name}: {len(class_indices)} ims")

        hover_texts = []
        for i in class_indices:
            filename = file_names_to_plot[i]
            current_class_id = class_ids_to_plot[i]
            imagenet_name = class_names.get(current_class_id, "N/A")  # Get name or "N/A"
            hover_texts.append(
                f"Filename: {filename}<br>Class ID: {current_class_id}<br>ImageNet Name: {imagenet_name}"
            )

        fig.add_trace(
            go.Scatter(
                x=results[class_indices, 0],
                y=results[class_indices, 1],
                mode="markers",
                name=f"Class: {parent_name}",
                text=hover_texts,
                hoverinfo="text",
                marker=dict(size=8, opacity=0.8, color=class_color_map[class_id], symbol=class_marker_map[class_id]),
            )
        )

    fig.update_layout(
        title=f"Interactive t-SNE Plot of Embeddings by Class (N={len(file_names_to_plot)})",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            title="Classes", itemsizing="constant", orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99
        ),
    )

    if output_html_path:
        print(f"Exporting plot to HTML: {output_html_path}")
        fig.write_html(output_html_path)
    else:
        fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an interactive t-SNE plot from high-dimensional embeddings.")
    parser.add_argument("--folder-path", type=str, required=True, help="Path to the folder containing .emb.npy files.")
    parser.add_argument(
        "-out",
        "--output-html-path",
        type=str,
        default=None,
        help=(
            "Optional: Path to save the interactive HTML figure (e.g., 'tsne_plot.html'). If not specified, the plot is"
            " shown in a browser."
        ),
    )
    parser.add_argument(
        "--max-embeddings",
        type=int,
        default=10000,
        help=(
            "Maximum number of embeddings to use for plotting. If more are found, a random subset is taken. Default:"
            " 10000"
        ),
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Seed for reproducibility of sampling, PCA, and t-SNE. Default: 42"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "Number of PCA components to reduce embedding dimensionality before t-SNE. "
            "If None or greater than embedding dimension, PCA is skipped. Default: None"
        ),
    )
    mapper_group = parser.add_mutually_exclusive_group(required=False)
    mapper_group.add_argument(
        "--tsne", action="store_const", dest="mapper", const="tsne", help="Use t-SNE for dimensionality reduction."
    )
    mapper_group.add_argument(
        "--umap",
        action="store_const",
        dest="mapper",
        const="umap",
        help="Use UMAP for dimensionality reduction (default).",
    )

    args = parser.parse_args()
    if args.mapper is None:
        args.mapper = "umap"
    print(args)

    plot_interactive_tsne_with_args(
        folder_path=args.folder_path,
        output_html_path=args.output_html_path,
        max_embeddings=args.max_embeddings,
        random_seed=args.random_seed,
        pca_components=args.pca_components,
        mapper=args.mapper,
    )
