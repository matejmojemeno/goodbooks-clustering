import plotly.express as px
import umap
from sklearn.manifold import TSNE


def visualize_clusters_umap(books, distance_matrix):
    umap_reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="precomputed", random_state=42
    )
    umap_embedding = umap_reducer.fit_transform(distance_matrix)

    books["cluster_size"] = books["cluster"].map(books["cluster"].value_counts())

    fig = px.scatter(
        books,
        x=umap_embedding[:, 0],
        y=umap_embedding[:, 1],
        color="cluster",
        hover_data={
            "cluster_name": True,
            "cluster_size": True,
            "title": True,
            "cluster": True,
            "keywords": True,
        },
        color_continuous_scale="Viridis",
        title="UMAP Cluster Visualization",
    )
    fig.show()


def visualize_clusters(books, distance_matrix, embeddings_weight, save=False):
    tsne = TSNE(n_components=2, metric="precomputed", random_state=42, init="random")
    tsne_embedding = tsne.fit_transform(distance_matrix)

    books["cluster_size"] = books["cluster"].map(books["cluster"].value_counts())

    fig = px.scatter(
        books,
        x=tsne_embedding[:, 0],
        y=tsne_embedding[:, 1],
        color="cluster",
        hover_data={
            "cluster_name": True,
            "cluster_size": True,
            "title": True,
            "cluster": True,
            "keywords": True,
        },
        color_continuous_scale="Viridis",
    )
    if save:
        fig.write_html(f"./output/2d_cluster_{embeddings_weight}.html")
    else:
        fig.show()


def visualize_clusters_3d(books, distance_matrix, embeddings_weight, save=False):
    tsne = TSNE(n_components=3, metric="precomputed", random_state=42, init="random")
    tsne_embedding = tsne.fit_transform(distance_matrix)

    books["cluster_size"] = books["cluster"].map(books["cluster"].value_counts())

    fig = px.scatter_3d(
        books,
        x=tsne_embedding[:, 0],
        y=tsne_embedding[:, 1],
        z=tsne_embedding[:, 2],
        color="cluster",
        hover_data={
            "cluster_name": True,
            "cluster_size": True,
            "title": True,
            "cluster": True,
            "keywords": True,
        },
        color_continuous_scale="Viridis",
    )

    if save:
        fig.write_html(f"./output/3d_cluster_{embeddings_weight}.html")
    else:
        fig.show()