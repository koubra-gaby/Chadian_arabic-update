import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_embeddings_and_visualisation(path,title, max_words=5000, n_labels=30):
    """
        Loading and visualising embedding
    """
    words = []
    vectors = []

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()

        for line in f:
            if len(words) >= max_words:
                break

            parts = line.rstrip().split()

            if len(parts) < 2:
                continue

            word = parts[0]

            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue

            words.append(word)
            vectors.append(vector)

            pca = PCA(n_components=2)
            embeddings = np.array(vectors)
            reduced = pca.fit_transform(embeddings)

            plt.figure(figsize=(14, 10))

            plt.scatter(
                reduced[:, 0],
                reduced[:, 1],
                s=10,
                alpha=0.5
            )

            # Afficher quelques étiquettes
            for i in range(min(n_labels, len(words))):
                plt.annotate(
                    words[i],
                    (reduced[i, 0], reduced[i, 1]),
                    fontsize=8
                )

            plt.title(title)
            plt.xlabel("Composante principale 1")
            plt.ylabel("Composante principale 2")
            plt.grid(alpha=0.2)
            plt.show()

            print("Variance expliquée :", pca.explained_variance_ratio_.sum())


def Tester_orthogonalite_de_lespace() :
    pass 

assert load_embeddings_and_visualisation("/home/koubra/Documents/Projet_IA/Chadian arabic update/Data/arabic_embedding.vec","Espace d'embeddings arabe - FastText")