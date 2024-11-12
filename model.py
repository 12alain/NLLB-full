# -*- coding: utf-8 -*-
"""Script de Fine-Tuning pour NLLB-200-distilled-600M"""

import os
import transformers
import torch
import evaluate
import numpy as np
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback)
from datasets import load_dataset

# Configurer les variables d'environnement pour Weights and Biases
os.environ["WANDB_PROJECT"] = "NLLB-200-distille-Experiments"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Charger le modèle et le tokenizer
model_checkpoint = '/home/mdrame/alain/nllb-200-distilled-600M'
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Fixer une graine aléatoire pour la reproductibilité
transformers.set_seed(7)
print(f"Version de transformers : {transformers.__version__}")

# Charger le dataset
path_data_dir = "/home/mdrame/alain/data/unidirection/fr_wo"
data = load_dataset(path_data_dir)

# Initialiser la métrique
metric = evaluate.load("sacrebleu")

# Préparer les données
max_input_length = 128
max_target_length = 128
source_lang = "src"
target_lang = "tgt"

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    target = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)
    labels = tokenizer(target, max_length=max_target_length, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Appliquer le prétraitement au dataset
tokenized_dataset = data.map(preprocess_function, batched=True, batch_size=100, remove_columns=data["train"].column_names)

# Arguments pour l'entraînement
batch_size = 16
source_lang = 'fr'
target_lang = 'wo'
model_checkpoint = "model_best/{}-finetuned-{}-to-{}".format(model_checkpoint.split("/")[-1], source_lang, target_lang)

args = Seq2SeqTrainingArguments(
    output_dir=model_checkpoint,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=120,
    predict_with_generate=True,
    report_to='all',
    remove_unused_columns=False,
    dataloader_num_workers=4,  # Ajustement des workers pour éviter les problèmes de mémoire
    load_best_model_at_end=True
)

# Préparer le collateur de données
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Fonction pour le calcul des métriques
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

# Initialiser le Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Lancer l'entraînement
trainer.train()

print("Entraînement terminé.")
print("oka")