import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def visualisation_one_space(embeddings,title, max_words=5000, with_word = False ,vectors = None, selected_words = None):
    """
        Loading and visualising embedding
    """

    # Conversion en matrices NumPy
    #arabic_embeddings = np.asarray(arabic_embeddings)
    #embeddings = vectors.vectors
    language_embeddings = np.asarray(embeddings)

    #print("Arabic shape:", arabic_embeddings.shape)
    print("shape:", language_embeddings.shape)

    #assert arabic_embeddings.ndim == 2
    assert language_embeddings.ndim == 2
    #assert arabic_embeddings.shape[1] == french_embeddings.shape[1]
    #assert arabic_embeddings.shape[0] >= 2
    assert language_embeddings.shape[0] >= 2

    pca = PCA(n_components=2)

    space_2d = pca.fit_transform(language_embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        space_2d[:, 0],
        space_2d[:, 1],
        alpha=0.4,
        s=5
    )
    if not with_word : 
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(alpha=0.2)
        plt.show()
    else :
        if selected_words and vectors: 
            for word in selected_words:
                if word in vectors:
                    idx = vectors.key_to_index[word]

                    plt.scatter(
                        space_2d[idx, 0],
                        space_2d[idx, 1],
                        s=40
                    )

                    plt.annotate(
                        word,
                        (space_2d[idx, 0], space_2d[idx, 1]),
                        fontsize=10
                    )

            plt.title("Selected French Words in NLLB Embedding Space")
            plt.show()





def visualisation_2_space(arabic_embedding,french_embedding) :

    # Conversion en matrices NumPy
    arabic_embeddings = np.asarray(arabic_embedding)
    french_embeddings = np.asarray(french_embedding)

    print("Arabic shape:", arabic_embeddings.shape)
    print("French shape:", french_embeddings.shape)

    assert arabic_embeddings.ndim == 2
    assert french_embeddings.ndim == 2
    assert arabic_embeddings.shape[1] == french_embeddings.shape[1]
    assert arabic_embeddings.shape[0] >= 2
    assert french_embeddings.shape[0] >= 2

    # Concaténer les deux espaces
    combined_embeddings = np.vstack([
        arabic_embeddings,
        french_embeddings
    ])

    # Créer les labels
    labels = np.array(
        ["Arabic"] * len(arabic_embeddings) +
        ["French"] * len(french_embeddings)
    )

    # PCA commune
    pca_combined = PCA(n_components=2)

    combined_2d = pca_combined.fit_transform(combined_embeddings)

    # Séparer les coordonnées
    arabic_combined_2d = combined_2d[labels == "Arabic"]
    french_combined_2d = combined_2d[labels == "French"]

    # Visualisation
    plt.figure(figsize=(12, 8))

    plt.scatter(
        arabic_combined_2d[:, 0],
        arabic_combined_2d[:, 1],
        alpha=0.4,
        s=5,
        label="Arabic"
    )

    plt.scatter(
        french_combined_2d[:, 0],
        french_combined_2d[:, 1],
        alpha=0.4,
        s=5,
        label="French"
    )

    plt.title("Arabic and French Embedding Spaces - Joint PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()




def plot_similar_word(src_words, src_word2id, src_emb, tgt_words, tgt_word2id, tgt_emb, pca):

    Y = []
    word_labels = []
    for sw in src_words:
        Y.append(src_emb[src_word2id[sw]])
        word_labels.append(sw)
    for tw in tgt_words:
        Y.append(tgt_emb[tgt_word2id[tw]])
        word_labels.append(tw)

    # find tsne coords for 2 dimensions
    Y = pca.transform(Y)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.figure(figsize=(10, 8), dpi=80)
    plt.scatter(x_coords, y_coords, marker='x')

    for k, (label, x, y) in enumerate(zip(word_labels, x_coords, y_coords)):
        color = 'blue' if k < len(src_words) else 'red'  # src words in blue / tgt words in red
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=19,
                     color=color, weight='bold')

    plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
    plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)
    plt.title('Visualization of the multilingual word embedding space')

    plt.show()



def check_embedding_orthogonality(embeddings, name="Embeddings",
                                  sample_size=2000, random_state=42):
    """
    Analyse l'orthogonalité entre les vecteurs d'un espace d'embeddings.

    Args:
        embeddings: matrice (n_words, embedding_dim)
        name: nom de l'espace
        sample_size: nombre de vecteurs échantillonnés
        random_state: graine aléatoire

    Returns:
        Statistiques des similarités cosinus entre vecteurs distincts.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError("Les embeddings doivent être une matrice 2D.")

    # Supprimer les vecteurs nuls
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings[norms > 1e-12]

    n = len(embeddings)

    if n < 2:
        raise ValueError("Il faut au moins deux vecteurs non nuls.")

    # Échantillonnage si le vocabulaire est trop grand
    rng = np.random.default_rng(random_state)

    if n > sample_size:
        indices = rng.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]

    # Normalisation L2
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Matrice des similarités cosinus
    similarities = normalized @ normalized.T

    # Ne conserver que les paires de vecteurs distincts
    upper_indices = np.triu_indices(len(normalized), k=1)
    cos_values = similarities[upper_indices]

    # Statistiques
    mean_cos = np.mean(cos_values)
    mean_abs_cos = np.mean(np.abs(cos_values))
    std_cos = np.std(cos_values)
    max_abs_cos = np.max(np.abs(cos_values))

    # Pourcentage de paires approximativement orthogonales
    orthogonal_001 = np.mean(np.abs(cos_values) < 0.01) * 100
    orthogonal_005 = np.mean(np.abs(cos_values) < 0.05) * 100
    orthogonal_010 = np.mean(np.abs(cos_values) < 0.10) * 100

    print(f"\n{'='*50}")
    print(f"Analyse : {name}")
    print(f"{'='*50}")
    print(f"Nombre de vecteurs analysés : {len(normalized)}")
    print(f"Dimension : {embeddings.shape[1]}")
    print(f"Similarité cosinus moyenne : {mean_cos:.6f}")
    print(f"Similarité cosinus absolue moyenne : {mean_abs_cos:.6f}")
    print(f"Écart-type : {std_cos:.6f}")
    print(f"Similarité absolue maximale : {max_abs_cos:.6f}")
    print(f"\nPaires avec |cos| < 0.01 : {orthogonal_001:.2f}%")
    print(f"Paires avec |cos| < 0.05 : {orthogonal_005:.2f}%")
    print(f"Paires avec |cos| < 0.10 : {orthogonal_010:.2f}%")

    return {
        "mean_cosine": mean_cos,
        "mean_absolute_cosine": mean_abs_cos,
        "std_cosine": std_cos,
        "max_absolute_cosine": max_abs_cos,
        "orthogonal_pairs_001": orthogonal_001,
        "orthogonal_pairs_005": orthogonal_005,
        "orthogonal_pairs_010": orthogonal_010
    }

