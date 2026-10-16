# Les importations
import evaluate
import os
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import torch
from datasets import Dataset, DatasetDict, load_dataset
#from comet import download_model, load_from_checkpoint
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

SOURCE_LANG_CODE_NLLB =  "fra_Latn" 
TARGET_LANG_CODE_NLLB =  "arb_Latn" 
max_input_length = 128
max_target_length = 128


class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()  # Libérer la mémoire GPU inutilisée

def plot_loss_curves(trainer, model_name) :
    log_history = trainer.state.log_history

    # Extract training / validation loss
    train_losses = [log["loss"] for log in log_history if "loss" in log]
    epoch_train = [log["step"] for log in log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epoch_eval = [log["step"] for log in log_history if "eval_loss" in log]

    # Plot the training loss
    plt.plot(epoch_train, train_losses, label="Training Loss")
    plt.plot(epoch_eval, eval_losses, label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss({model_name})")
    plt.legend()
    plt.grid(True)
    plt.show()


def training (modele_name, train_test_val_dataset) : 
    
    model = AutoModelForSeq2SeqLM.from_pretrained(modele_name)
    tokenizer = AutoTokenizer.from_pretrained(modele_name)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]  # ROUGE/METEOR attend des listes imbriquées
        return preds, labels


    # Charger les métriques une seule fois, en dehors de la fonction
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    chrf_metric = evaluate.load("chrf")
    #comet_metric = evaluate.load("comet")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        # Remplacer les valeurs ignorées par le token PAD
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Décodage des prédictions et des références
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # Nettoyage du texte
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels
        )

        # Calcul des métriques
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        rouge_result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        meteor_result = meteor_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        chrf_result = chrf_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            word_order=2  # chrF++ : caractères + n-grammes de mots
        )

        # COMET nécessite les sources en plus des traductions et références.
        # Voir l'explication ci-dessous pour récupérer les sources.
        # comet_result = comet_metric.compute(
        #     predictions=decoded_preds,
        #     references=decoded_labels
        # )

        result = {
            "bleu": bleu_result["score"],
            "rouge": rouge_result["rougeL"],
            "meteor": meteor_result["meteor"],
            "chrf++": chrf_result["score"],
            #"comet": comet_result["mean_score"],
        }

        return {k: round(float(v), 4) for k, v in result.items()}



    def comet_value(train_test_val_dataset, tokenized_dataset):

        comet_metric = evaluate.load("comet")

        # Dataset brut : textes originaux
        test_raw = train_test_val_dataset["test"]["translation"]

        sources = test_raw["fr"]
        references = test_raw["arb"]

        # Dataset tokenisé : entrées du modèle
        test_tokenized = tokenized_dataset["test"]

        pred_output = trainer.predict(test_tokenized)
        pred_ids = pred_output.predictions

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Décoder les prédictions
        pred_ids = np.where(
            pred_ids == -100,
            tokenizer.pad_token_id,
            pred_ids
        )

        predictions = tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True
        )

        # Vérifier l'alignement
        assert len(sources) == len(predictions) == len(references), (
            "Les sources, prédictions et références n'ont pas la même longueur."
        )

        comet_result = comet_metric.compute(
            sources=sources,
            predictions=predictions,
            references=references
        )

        print("Score COMET sur le test set :", comet_result["mean_score"])

        return comet_result["mean_score"]




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



    # On va appliquer la fonction sur notre dataset
    train_dataset = train_test_val_dataset.map(extract_langage, batched=True, remove_columns=["translation"])
    # On va appliquer la fonction sur notre dataset
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["inputs", "target"])
    # Dataloader
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)


    early =  EarlyStoppingCallback(early_stopping_patience = 3)


    training_args = Seq2SeqTrainingArguments(
        output_dir=f"../src/{modele_name}",
        report_to="none",
        #hub_model_id= os.environ.get('HUB_ID_2'),  
        eval_strategy="steps",
        eval_steps=500,              
        logging_steps=500, 
        learning_rate=2e-5,
        #learning_rate=1e-5,
        gradient_checkpointing=True,  
        #use_cache = False,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=5,
        num_train_epochs=50,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        fp16=True, 
        #push_to_hub=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_strategy="steps",
        #save_steps = 50,
        resume_from_checkpoint=True,
        #gradient_accumulation_steps=4,

    )




    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early, MemoryCleanupCallback], #CheckpointCleanupCallback()
    )
    torch.cuda.empty_cache()  # Libérer la mémoire GPU inutilisée

    trainer.train()
    print(train_test_val_dataset["test"].column_names)
    results = trainer.evaluate(train_dataset["test"]) 

    comet_result = comet_value(train_test_val_dataset,train_dataset )

    return trainer,results, comet_result


