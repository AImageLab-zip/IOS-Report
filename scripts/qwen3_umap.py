import os
import numpy as np
from tqdm import tqdm
import umap

from src.qwen3_vl_embedder import Qwen3VLImageEmbedder
from src.dataset.ferrara_dump_dataset import FerraraDumpDataset

EMBEDDING_CACHE_PATH = "/work/grana_maxillo/IOS_DraftReport/embeddings_cache/qwen3_vl_umap_embeddings.npy"
def umap_embeddings(embeddings, n_neighbors=200, min_dist=0.01, n_components=2, random_state=42):
    # Initialize UMAP reducer
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
    )

    embedding_umap = reducer.fit_transform(embeddings)
    
    return embedding_umap

def kmeans_clustering(embeddings, n_clusters=5, random_state=42):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels

def plot_umap(embedding_umap, cluster_labels):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=cluster_labels, s=5, alpha=0.7)
    plt.title("UMAP Projection of Image Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    plt.savefig("qwen3_vl_image_embeddings_umap.png", dpi=300)
    plt.show()

def compute_embeddings():
    if os.path.exists(EMBEDDING_CACHE_PATH):
        print(f"Loading cached embeddings from {EMBEDDING_CACHE_PATH}...")
        all_embeddings = np.load(EMBEDDING_CACHE_PATH)
        print(f"Loaded {all_embeddings.shape} embeddings from cache.")
        return all_embeddings
    
    all_embeddings = []
    for idx in tqdm(range(len(dataset))):
        patient = dataset[idx]
        for image in patient['images']:
            embedding = embedder.get_class_token(image)
            all_embeddings.append(embedding)
    all_embeddings = [emb.squeeze().detach().cpu().float().numpy() for emb in all_embeddings]
    all_embeddings = np.stack(all_embeddings, axis=0)
    
    np.save(EMBEDDING_CACHE_PATH, all_embeddings)
    
    return all_embeddings

if __name__ == "__main__":
    embedder = Qwen3VLImageEmbedder()
    dataset = FerraraDumpDataset(
        root_dir="/work/grana_maxillo/Dataset_FerraraDump_ToothFairy4M/",
        preload=False,
        load_meshes=False,
        load_images=True,
        num_workers=8,
        image_size=(512, 512),
    )
    
    all_embeddings = compute_embeddings()
            
    embedding_umap = umap_embeddings(all_embeddings, n_neighbors=42, min_dist=0.01, n_components=2)
    kmeans_labels = kmeans_clustering(embedding_umap, n_clusters=10)
    plot_umap(embedding_umap, kmeans_labels)