# Les importations
from transformers import get_linear_schedule_with_warmup, AutoTokenizer,TrainerCallback,AutoModelForCausalLM, DataCollatorForSeq2Seq,AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer,EarlyStoppingCallback
from datasets import load_dataset,Dataset, DatasetDict
import evaluate
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

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


def training (model, tokenizer, train_test_val_dataset) : 
    
    loss_tracker = LossTrackerCallback()

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]  # ROUGE/METEOR attend des listes imbriquées
        return preds, labels

    def compute_metrics(eval_preds):

        # Charger les métriques
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load("meteor")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Décodage des prédictions et labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Nettoyage du texte
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # 🏆 Calcul des métriques
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # 📊 Récupération des scores
        result = {
            "bleu": bleu_result["score"],
            "rouge": rouge_result["rougeL"],  # Utilisation du score ROUGE-L
            "meteor": meteor_result["meteor"],
        }

        # Arrondir les scores
        result = {k: round(v, 4) for k, v in result.items()}
        return result


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
        output_dir="/my_train_model_facebook_nllb_200",
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

