#! /usr/bin/python3

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv

from typing import List
from utils import read_jsonl

STR_MODEL_NAME = "microsoft/codebert-base"

def read_dataset(str_fix_jsonl: str, str_nonfix_jsonl: str) -> list:
    list_dict_fix = read_jsonl(str_fix_jsonl)
    list_dict_nonfix = read_jsonl(str_nonfix_jsonl)

    list_str_text, list_n_label = [], []

    # load the fix samples (labeled as 1)
    for dict_fix in list_dict_fix:
        str_commit_message = dict_fix["commit_message"]
        str_code_diff = dict_fix["all_code"]
        list_str_text.append(f"{str_commit_message}. {str_code_diff}")
        list_n_label.append(1)

    # load the non-fix samples (labeled as 0)
    for dict_nonfix in list_dict_nonfix:
        str_commit_message = dict_nonfix["commit_message"]
        str_code_diff = dict_nonfix["all_code"]
        list_str_text.append(f"{str_commit_message}. {str_code_diff}")
        list_n_label.append(0)

    return list_str_text, list_n_label

def model_initialization(from_trained=False, model_path="./final_model"):
    if from_trained:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(STR_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(STR_MODEL_NAME, num_labels=2)
    return model, tokenizer

def to_dataset(texts: List[str], labels: List[int]):
    return Dataset.from_list([{"text": t, "label": l} for t, l in zip(texts, labels)])

def compute_metrics(eval_pred):
    preds = torch.argmax(torch.tensor(eval_pred.predictions), axis=1)
    labels = torch.tensor(eval_pred.label_ids)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def read_single_dataset(str_jsonl: str):
    list_dict_sample = read_jsonl(str_jsonl)

    list_str_text, list_n_label = [], []

    for dict_sample in list_dict_sample:
        str_commit_message = dict_sample["commit_message"]
        str_code_diff = dict_sample["all_code"]
        n_label = dict_sample["label"]

        list_str_text.append(f"{str_commit_message}. {str_code_diff}")
        # list_str_text.append(str_commit_message)
        list_n_label.append(n_label)

    return list_str_text, list_n_label


if __name__ == "__main__":
    bool_train = False  # True: train model, False: test model only

    list_str_train_text, list_n_train_label = read_single_dataset("./dataset/train_dataset.jsonl")

    list_str_train_dbms_text, list_n_train_dbms_label = read_single_dataset("./dataset/train_dbms_dataset.jsonl")
    list_str_test_text, list_n_test_label = read_single_dataset("./dataset/test_dbms_dataset.jsonl")

    train_dataset_raw = to_dataset(list_str_train_text, list_n_train_label)
    train_dbms_dataset_raw = to_dataset(list_str_train_dbms_text, list_n_train_dbms_label)
    test_dataset_raw = to_dataset(list_str_test_text, list_n_test_label)

    # model and tokenizer initialization
    model, tokenizer = model_initialization(from_trained=not bool_train, model_path="./trained_models/nighthawk")

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    train_dataset = train_dataset_raw.map(tokenize_fn)
    train_dbms_dataset = train_dbms_dataset_raw.map(tokenize_fn)
    test_dataset = test_dataset_raw.map(tokenize_fn)


    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if bool_train else None,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )


    if bool_train:
        trainer.train()
        trainer.save_model("./trained_models/general_trained")
        tokenizer.save_pretrained("./trained_models/general_trained")

        # DBMS-specific fine-tuning

        model, tokenizer = model_initialization(True, "./trained_models/general_trained")
        trainer_dbms = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dbms_dataset if bool_train else None,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )


        trainer_dbms.train()
        trainer_dbms.save_model(".../final_model_save_path")
        tokenizer.save_pretrained(".../final_model_save_path")

    else:
        print("Evaluating model on test set...")
        metrics = trainer.evaluate()
        print("Evaluation results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        print("\nCalculating detected and confirmed fix predictions...")
        predictions = trainer.predict(test_dataset)
        pred_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1).tolist()
        true_labels = predictions.label_ids.tolist()

        # read the commit urls
        list_dict_commit = read_jsonl("./dataset/test_dbms_dataset.jsonl")

        detected = sum(1 for p in pred_labels if p == 1)
        confirmed = sum(1 for p, t in zip(pred_labels, true_labels) if p == 1 and t == 1)