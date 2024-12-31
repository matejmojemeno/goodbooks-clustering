import matplotlib.pyplot as plt
import plotly.express as px
import umap
from sklearn.manifold import TSNE


def visualize_clusters_umap(
    books, distance_matrix, gower, embeddings, interact, save=False
):
    """Visualize the clusters using UMAP."""
    reducer = umap.UMAP(metric="precomputed", random_state=42)
    umap_embedding = reducer.fit_transform(distance_matrix)

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
    )
    fig.update_layout(
        title=f"Gower: {gower}, Embeddings: {embeddings}, Interactions: {interact}"
    )
    if save:
        fig.write_html(
            f"./output/umap_{int(gower)}_{int(embeddings)}_{int(interact)}.html"
        )
    else:
        fig.show()
