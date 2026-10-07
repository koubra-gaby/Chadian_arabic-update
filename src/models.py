# Les importations
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from datasets import Dataset, DatasetDict, load_dataset
from comet import download_model, load_from_checkpoint
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    get_linear_schedule_with_warmup,
)

SOURCE_LANG_CODE_NLLB =  "fra_Latn" 
TARGET_LANG_CODE_NLLB =  "arb_Latn" 
max_input_length = 128
max_target_length = 128



class LossTrackerCallback(TrainerCallback):
    def __init__(self):
        self.history = {
            "step": [],
            "train_loss": [],
            "eval_loss": []
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        if "loss" in logs:
            self.history["step"].append(state.global_step)
            self.history["train_loss"].append(logs["loss"])
            self.history["eval_loss"].append(None)

        if "eval_loss" in logs:
            self.history["step"].append(state.global_step)
            self.history["train_loss"].append(None)
            self.history["eval_loss"].append(logs["eval_loss"])


def training (modele_name, train_test_val_dataset) : 
    
    loss_tracker = LossTrackerCallback()
    model = AutoModelForSeq2SeqLM.from_pretrained(modele_name)
    tokenizer = AutoTokenizer.from_pretrained(modele_name)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]  # ROUGE/METEOR attend des listes imbriquées
        return preds, labels

    def compute_metrics(eval_preds, dataset_inputs=None):
        """
        Calcule les métriques de traduction incluant BLEU, ROUGE, METEOR, chrF++ et COMET.
        
        :param eval_preds: Tuple (preds, labels) fourni par le Trainer.
        :param dataset_inputs: (Optionnel) Liste des textes sources d'origine. 
                            Si non fourni, COMET sera ignoré ou une simulation sera faite.
        """

        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_model_path)
        # Charger les métriques standards via Hugging Face evaluate
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load("meteor")
        chrf_metric = evaluate.load("chrf") # Le module 'chrf' de evaluate supporte chrF++ via ses options

        preds, labels = eval_preds

        # Si les prédictions sont des logis/probabilités, on prend l'argmax
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = np.argmax(preds, axis=-1) if preds.ndim == 3 else preds

        # Remplacer les -100 (utilisés pour ignorer le calcul de la loss sur le padding) par le token de padding
        decoded_labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Décoder les tokens en chaînes de caractères (Remplacez 'tokenizer' par le nom de votre variable de tokenizer)
        decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(decoded_labels, skip_special_tokens=True)

        # Nettoyage rapide (enlever les espaces superflus)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels] # sacrebleu attend une liste de listes pour les refs

        # 1. Calcul des métriques classiques
        bleu_results = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_results = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # 2. Calcul de chrF++ 
        # Pour activer chrF++, on passe word_order=2 dans les options de configuration
        chrf_results = chrf_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            word_order=2
        )

        # Regrouper les résultats de base
        results = {
            "bleu": bleu_results["score"],
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "meteor": meteor_results["meteor"],
            "chrf++": chrf_results["score"]
        }

        # 3. Calcul de COMET (Nécessite la source)
        if dataset_inputs is not None:
            # Format attendu par COMET : [{'src': ..., 'mt': ..., 'ref': ...}, ...]
            # Comme decoded_labels est une liste de listes (ex: [['ref1'], ['ref2']]), on prend le premier élément [0]
            comet_data = [
                {"src": src, "mt": mt, "ref": ref[0]}
                for src, mt, ref in zip(dataset_inputs, decoded_preds, decoded_labels)
            ]
            
            # Calcul des prédictions (gpus=0 utilise le CPU, passez à 1 ou laissez gpus=None si vous utilisez CUDA)
            comet_output = comet_model.predict(comet_data, batch_size=16, gpus=0, progress_bar=False)
            results["comet"] = comet_output.system_score

        return results


    def extract_langage(examples) :
        # On va extraire le langage arabe et le langage français
        inputs = [fr['fr'] for fr in examples["translation"]]
        target = [arb['arb'] for arb in examples["translation"]]
        return {'inputs' : inputs, 'target' : target}



    def preprocess_function(examples):
        tokenizer.src_lang = SOURCE_LANG_CODE_NLLB
        model_inputs = tokenizer(
            examples["inputs"],
            max_length=max_input_length,
            truncation=True
        )

        # Nettoyer les cibles : enlever None ou chaînes vides
        clean_targets = []
        for t in examples["target"]:
            if isinstance(t, str) and t.strip():
                clean_targets.append(t)
            else:
                clean_targets.append(" ")

        tokenizer.tgt_lang = TARGET_LANG_CODE_NLLB
        labels = tokenizer(
            clean_targets,
            max_length=max_target_length,
            truncation=True,
            text_target=clean_targets  # Pour compatibilité future
        )

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    
    def train_and_valid_loss() :

        loss_dict = loss_tracker.history
        steps = loss_dict["step"]

        train_loss = [
            l for l in loss_dict["train_loss"] if l is not None]
        train_steps = [
            s for s, l in zip(steps, loss_dict["train_loss"]) if l is not None]

        eval_loss = [
            l for l in loss_dict["eval_loss"] if l is not None
        ]
        eval_steps = [
            s for s, l in zip(steps, loss_dict["eval_loss"]) if l is not None
        ]

        plt.plot(train_steps, train_loss, label="Train Loss")
        plt.plot(eval_steps, eval_loss, label="Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()


    # On va appliquer la fonction sur notre dataset
    train_dataset = train_test_val_dataset.map(extract_langage, batched=True, remove_columns=["translation"])
    # On va appliquer la fonction sur notre dataset
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["inputs", "target"])
    # Dataloader
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


    early =  EarlyStoppingCallback(early_stopping_patience = 5)
    #loss_tracker = LossTrackerCallback()


    training_args = Seq2SeqTrainingArguments(
        output_dir=f"../src/{modele_name}",
        report_to="none",
        #hub_model_id= os.environ.get('HUB_ID_2'),  # ✅ Ajoute ton Repo ID
        eval_strategy="steps",
        learning_rate=2e-5,
        #learning_rate=1e-5,
        gradient_checkpointing=True,  # ✅ Active la réduction mémoire
        #use_cache = False,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=50,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        fp16=True, 
        #push_to_hub=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_strategy="steps",
        #save_steps = 500,
        #resume_from_checkpoint=True,
        gradient_accumulation_steps=4,

    )




    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early,loss_tracker ], #CheckpointCleanupCallback()
    )
    torch.cuda.empty_cache()  # Libérer la mémoire GPU inutilisée

    return trainer

