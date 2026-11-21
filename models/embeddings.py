import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import io
import re
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from gensim.test.utils import common_texts


def preparer_sentences_entrainer_fastext(fichier1, fichier2, file_to_save, lang):
    sentences = []
    if lang == "fr":
      patern = r"[^\w\s]"
      print("french")
      #patern = r"[^\W\d_]+"
    else :
      patern = r"[^\w\s']"
    for fichier in [fichier1, fichier2]:
        with open(fichier, "r", encoding="utf-8-sig") as f:
            for ligne in f:
                ligne = ligne.strip()

                if ligne:
                    # Remplacer les ponctuations par des espaces,
                    # sauf les apostrophes et les tirets
                    ligne = re.sub(
                        patern,
                        " ",
                        ligne,
                        flags=re.UNICODE
                    )
                    print(ligne)
                    # Tokenisation mot par mot
                    tokens = word_tokenize(
                        ligne,
                        preserve_line=True
                    )

                    if tokens:
                        sentences.append(tokens)

      ## training fastext
    model = FastText(
      sentences,
      vector_size=1024,
      window=5,
      min_count=1,
      epochs=10,
      workers=4,
      sg=1,
  )
    model.wv.save_word2vec_format(file_to_save)

    print("Model trained and savec successfully")

    return model,sentences


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






def add_muse_words_to_nllb(
    muse_file,
    tokenizer,
    model,
    language="shu_Latn"
):
    """
    Ajoute les mots arabes absents du tokenizer NLLB
    et initialise leurs embeddings avec les vecteurs MUSE,
    sans aucune transformation des vecteurs.

    Parameters
    ----------
    muse_file : str
        Fichier .vec contenant les mots et leurs vecteurs MUSE.
    tokenizer : AutoTokenizer
        Tokenizer NLLB.
    model : AutoModel ou AutoModelForSeq2SeqLM
        Modèle NLLB.
    language : str
        Code de langue utilisé pour le tokenizer.

    Returns
    -------
    tokenizer, model, report
    """

    # 1. Lire les vecteurs MUSE
    muse_vectors = {}

    with open(muse_file, "r", encoding="utf-8") as f:

        header = f.readline().strip().split()
        expected_vocab_size = int(header[0])
        muse_dim = int(header[1])

        for line in tqdm(f, desc="Lecture des vecteurs MUSE"):

            parts = line.rstrip().split()

            if len(parts) != muse_dim + 1:
                continue

            word = parts[0]
            vector = np.asarray(
                parts[1:],
                dtype=np.float32
            )

            if word not in muse_vectors:
                muse_vectors[word] = vector

    print("Mots MUSE chargés :", len(muse_vectors))
    print("Dimension MUSE :", muse_dim)

    # 2. Vérifier les dimensions
    embedding_layer = model.get_input_embeddings()
    embedding_dim = embedding_layer.weight.shape[1]

    if muse_dim != embedding_dim:
        raise ValueError(
            f"Dimensions incompatibles : MUSE={muse_dim}, "
            f"NLLB={embedding_dim}"
        )

    # 3. Identifier les mots déjà présents
    existing_tokens = set(tokenizer.get_vocab().keys())

    new_words = [
        word for word in muse_vectors
        if word not in existing_tokens
    ]

    existing_words = [
        word for word in muse_vectors
        if word in existing_tokens
    ]

    print("Mots déjà présents :", len(existing_words))
    print("Nouveaux mots à ajouter :", len(new_words))

    if not new_words:
        return tokenizer, model, {
            "added": 0,
            "already_present": len(existing_words)
        }

    # 4. Ajouter les mots au tokenizer
    old_vocab_size = len(tokenizer)

    added_count = tokenizer.add_tokens(new_words)

    if added_count != len(new_words):
        raise RuntimeError(
            "Le nombre de tokens ajoutés ne correspond pas "
            "au nombre de mots absents."
        )

    # 5. Agrandir la matrice d'embeddings
    model.resize_token_embeddings(len(tokenizer))

    # 6. Copier les vecteurs MUSE sans modification
    embedding_layer = model.get_input_embeddings()

    with torch.no_grad():

        for word in tqdm(new_words, desc="Initialisation NLLB"):

            token_id = tokenizer.convert_tokens_to_ids(word)

            muse_vector = torch.from_numpy(
                muse_vectors[word]
            ).to(
                device=embedding_layer.weight.device,
                dtype=embedding_layer.weight.dtype
            )

            embedding_layer.weight[token_id].copy_(
                muse_vector
            )

    # 7. Vérification
    print("\nTerminé !")
    print("Ancienne taille :", old_vocab_size)
    print("Nouvelle taille :", len(tokenizer))
    print("Nombre de mots ajoutés :", added_count)

    report = {
        "added": added_count,
        "already_present": len(existing_words),
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": len(tokenizer),
        "embedding_dimension": embedding_dim
    }

    return tokenizer, model, report


def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))



def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def filter_mutual_nn(
    arabic_path,
    french_path,
    output_path,
    K=5,
    nmax=50000
):
    # Charger les embeddings
    arabic_emb, arabic_id2word, arabic_word2id = load_vec(
        arabic_path, nmax
    )

    french_emb, french_id2word, french_word2id = load_vec(
        french_path, nmax
    )

    # Normaliser les embeddings pour calculer les similarités cosinus
    arabic_norm = (
        arabic_emb / np.linalg.norm(arabic_emb, 2, 1)[:, None]
    )

    french_norm = (
        french_emb / np.linalg.norm(french_emb, 2, 1)[:, None]
    )

    retained_words = []

    # Parcourir chaque mot arabe
    for arabic_idx, arabic_word in arabic_id2word.items():

        arabic_vector = arabic_norm[arabic_idx]

        # K plus proches voisins français du mot arabe
        scores_ar_fr = french_norm.dot(arabic_vector)
        french_neighbors = scores_ar_fr.argsort()[-K:][::-1]

        # Vérifier la réciprocité
        for french_idx in french_neighbors:

            french_word = french_id2word[french_idx]
            french_vector = french_norm[french_idx]

            # K plus proches voisins arabes du mot français
            scores_fr_ar = arabic_norm.dot(french_vector)
            arabic_neighbors = scores_fr_ar.argsort()[-K:][::-1]

            # Le mot arabe initial est-il parmi ces voisins ?
            if arabic_idx in arabic_neighbors:
                retained_words.append(arabic_word)
                break

    # Écrire les mots arabes conservés avec leurs embeddings originaux
    with io.open(output_path, 'w', encoding='utf-8', newline='\n') as f:

        f.write(f"{len(retained_words)} {arabic_emb.shape[1]}\n")

        for word in retained_words:
            idx = arabic_word2id[word]
            vector = arabic_emb[idx]

            f.write(
                word + " " +
                " ".join(map(str, vector)) + "\n"
            )

    print("Vocabulaire arabe initial :", len(arabic_emb))
    print("Mots arabes conservés :", len(retained_words))
    print("Mots arabes supprimés :", len(arabic_emb) - len(retained_words))
    print("Fichier généré :", output_path)

    return retained_words
