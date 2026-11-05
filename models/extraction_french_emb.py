import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def extract_nllb_word_embeddings(
    input_file,
    output_file,
    model_name="facebook/nllb-200-distilled-600M",
    language="fra_Latn"
):
    """
    Extrait les embeddings NLLB des mots français d'un fichier.

    Chaque mot est tokenisé en sous-tokens.
    Son embedding est calculé par la moyenne des embeddings
    de ses sous-tokens.

    Parameters
    ----------
    input_file : str
        Fichier texte contenant un mot français par ligne.
    output_file : str
        Chemin du fichier .vec de sortie.
    model_name : str
        Modèle NLLB à utiliser.
    language : str
        Code de langue NLLB.
    """

    # 1. Charger le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang=language
    )

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    model.eval()

    # 2. Récupérer directement la matrice des embeddings
    embedding_matrix = (
        model.get_input_embeddings()
        .weight.detach()
    )

    embedding_dim = embedding_matrix.shape[1]

    # 3. Lire les mots français
    with open(input_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # Supprimer les doublons en conservant l'ordre
    words = list(dict.fromkeys(words))

    word_embeddings = []

    # 4. Extraire les embeddings
    for word in tqdm(words, desc="Extraction des embeddings"):

        # Tokeniser le mot sans ajouter les tokens spéciaux
        token_ids = tokenizer(
            word,
            add_special_tokens=False
        )["input_ids"]

        if not token_ids:
            continue

        # Récupérer les embeddings des sous-tokens
        subword_embeddings = embedding_matrix[token_ids]

        # Moyenne des embeddings
        word_embedding = subword_embeddings.mean(dim=0)

        word_embeddings.append(
            (word, word_embedding.numpy())
        )

    # 5. Exporter au format Word2Vec / MUSE
    with open(output_file, "w", encoding="utf-8") as f:

        f.write(f"{len(word_embeddings)} {embedding_dim}\n")

        for word, vector in word_embeddings:
            vector_str = " ".join(
                f"{value:.8f}" for value in vector
            )

            f.write(f"{word} {vector_str}\n")

    print(f"Extraction terminée : {len(word_embeddings)} mots")
    print(f"Dimension des embeddings : {embedding_dim}")
    print(f"Fichier enregistré : {output_file}")

    return word_embeddings