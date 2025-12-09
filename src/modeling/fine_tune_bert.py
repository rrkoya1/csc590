# train_transformer.py
# Generic transformer trainer for text classification (Top-8/12 cancer-type, TNM T-stage)
# Compat-safe TrainingArguments: only enable best-checkpoint logic if supported by local transformers

import argparse
import json
import os
import random
import inspect
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick(keys: List[str], obj: Dict):
    """Return obj[k] for the first k in keys that exists, else raise KeyError."""
    for k in keys:
        if k in obj:
            return obj[k]
    raise KeyError(f"None of the keys {keys} found in split json. Keys present: {list(obj.keys())}")

def load_split(filtered_csv: str, split_json: str, text_col: str, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataframe and indices (supports multiple key name variants)."""
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_idx = pick(["train", "train_ids", "train_idx"], split)
    val_idx   = pick(["val", "valid", "val_ids", "valid_ids", "val_idx", "valid_idx"], split)
    test_idx  = pick(["test", "test_ids", "test_idx"], split)

    df = pd.read_csv(filtered_csv)

    for col in (text_col, label_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not in {filtered_csv}. Available: {df.columns.tolist()}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df

def build_label_maps(series: pd.Series):
    classes = sorted(series.unique().tolist())
    cls2id = {c: i for i, c in enumerate(classes)}
    id2cls = {i: c for c, i in cls2id.items()}
    return cls2id, id2cls

def to_hf_dataset(df: pd.DataFrame, text_col: str, label_col: str, cls2id: Dict[str, int]) -> Dataset:
    return Dataset.from_pandas(
        pd.DataFrame({
            "text": df[text_col].astype(str).tolist(),
            "label": df[label_col].map(cls2id).astype(int).tolist(),
        }),
        preserve_index=False
    )

def tokenize_function(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

def compute_metrics_builder(num_labels: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        macro = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro}
    return compute_metrics

def filter_kwargs_for_signature(kwargs: Dict, cls) -> Dict:
    """Keep only kwargs supported by cls.__init__ (compat helper)."""
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--filtered_csv", required=True, type=str)
    parser.add_argument("--split_json", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="cancer_type")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load data/splits
    train_df, val_df, test_df = load_split(
        args.filtered_csv, args.split_json, args.text_col, args.label_col
    )

    # 2) Labels
    cls2id, id2cls = build_label_maps(train_df[args.label_col])

    # 3) Tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    hf_train = to_hf_dataset(train_df, args.text_col, args.label_col, cls2id)
    hf_val   = to_hf_dataset(val_df,   args.text_col, args.label_col, cls2id)
    hf_test  = to_hf_dataset(test_df,  args.text_col, args.label_col, cls2id)

    hf_train = hf_train.map(lambda b: tokenize_function(b, tokenizer, args.max_length), batched=True, remove_columns=["text"])
    hf_val   = hf_val.map(  lambda b: tokenize_function(b, tokenizer, args.max_length), batched=True, remove_columns=["text"])
    hf_test  = hf_test.map( lambda b: tokenize_function(b, tokenizer, args.max_length), batched=True, remove_columns=["text"])

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(cls2id),
        id2label={i: c for i, c in id2cls.items()},
        label2id=cls2id,
    )

    # 5) TrainingArguments (compat-safe “best checkpoint” logic)
    base_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.0,
        logging_steps=50,
        save_total_limit=1,
        seed=args.seed,
        report_to=[],  # no external loggers by default
        fp16=torch.cuda.is_available(),
    )

    sig = inspect.signature(TrainingArguments.__init__)
    has_eval   = "evaluation_strategy" in sig.parameters
    has_save   = "save_strategy" in sig.parameters
    has_load   = "load_best_model_at_end" in sig.parameters
    has_metric = "metric_for_best_model" in sig.parameters

    extra = {}
    # Only turn these on if BOTH eval/save exist; only turn load_best on if eval/save exist too.
    if has_eval and has_save:
        extra["evaluation_strategy"] = "epoch"
        extra["save_strategy"] = "epoch"
        if has_load:
            extra["load_best_model_at_end"] = True
            if has_metric:
                extra["metric_for_best_model"] = "macro_f1"

    ta_kwargs = filter_kwargs_for_signature({**base_kwargs, **extra}, TrainingArguments)
    train_args = TrainingArguments(**ta_kwargs)

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        tokenizer=tokenizer,  # fine if it warns about future deprecation
        compute_metrics=compute_metrics_builder(len(cls2id)),
    )

    # 7) Train
    trainer.train()

    # 8) Validation metrics
    val_metrics = trainer.evaluate(hf_val)
    print("== Validation ==")
    print(val_metrics)

    # 9) Test metrics
    test_pred = trainer.predict(hf_test)
    test_logits = test_pred.predictions
    test_labels = test_pred.label_ids
    test_preds  = np.argmax(test_logits, axis=1)
    test_macro  = f1_score(test_labels, test_preds, average="macro")
    print("== Test ==")
    print({"macro_f1": test_macro})

    # 10) Save artifacts
    y_true = [id2cls[i] for i in test_labels]
    y_pred = [id2cls[i] for i in test_preds]

    cls_report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(cls_report)

    cm = confusion_matrix(y_true, y_pred, labels=list(cls2id.keys()))
    pd.DataFrame(cm, index=list(cls2id.keys()), columns=list(cls2id.keys())).to_csv(
        os.path.join(args.output_dir, "cm.csv"), index=True
    )

    row = {
        "model": args.model_name_or_path,
        "max_len": args.max_length,
        "epochs": args.num_train_epochs,
        "lr": args.learning_rate,
        "batch": args.batch_size,
        "val_macro_f1": float(val_metrics.get("eval_macro_f1", np.nan)),
        "test_macro_f1": float(test_macro),
        "seed": args.seed,
    }
    pd.DataFrame([row]).to_csv(
        os.path.join(args.output_dir, "results_table.csv"), index=False
    )

    print("\nSaved:")
    print(f"  - {os.path.join(args.output_dir, 'classification_report.txt')}")
    print(f"  - {os.path.join(args.output_dir, 'cm.csv')}")
    print(f"  - {os.path.join(args.output_dir, 'results_table.csv')}")

if __name__ == "__main__":
    main()
