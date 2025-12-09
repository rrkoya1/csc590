
import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def load_split(filtered_csv, split_json):
    """Loads train/val/test based on fixed indices."""
    df = pd.read_csv(filtered_csv)
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    
    # Robust key lookups (handles 'train' vs 'train_ids')
    def get_idx(keys):
        for k in keys:
            if k in split: return split[k]
        raise KeyError(f"Could not find indices. Available keys: {list(split.keys())}")

    train_idx = get_idx(["train", "train_ids", "train_idx"])
    val_idx   = get_idx(["val", "val_ids", "val_idx", "validation"])
    test_idx  = get_idx(["test", "test_ids", "test_idx"])

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to filtered.csv")
    parser.add_argument("--split_json", required=True, help="Path to split.json")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="cancer_type")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_n", type=int, default=None, help="(Optional) Compatibility arg")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {args.csv_path} using {args.split_json}...")
    train_df, val_df, test_df = load_split(args.csv_path, args.split_json)
    
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    # 2. Vectorize (TF-IDF)
    print("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    # Fit on Train only
    X_train = tfidf.fit_transform(train_df[args.text_col].astype(str))
    X_val   = tfidf.transform(val_df[args.text_col].astype(str))
    X_test  = tfidf.transform(test_df[args.text_col].astype(str))

    y_train = train_df[args.label_col]
    y_val   = val_df[args.label_col]
    y_test  = test_df[args.label_col]

    # 3. Train & Eval
    results = {}
    
    # We test both LogReg and SVM
    models = [
        ("logreg", LogisticRegression(max_iter=1000, C=1.0)),
        ("svm",    LinearSVC(dual="auto", C=1.0, max_iter=1000))
    ]

    for name, clf in models:
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        
        # Inference
        test_pred = clf.predict(X_test)
        
        # Metrics
        test_f1 = f1_score(y_test, test_pred, average="macro")
        acc = (test_pred == y_test).mean()
        
        print(f"  -> {name} Test Macro-F1: {test_f1:.4f}")

        # Save classification report
        report = classification_report(y_test, test_pred, digits=4)
        with open(os.path.join(args.output_dir, f"classification_report_{name}.txt"), "w") as f:
            f.write(report)
            f.write(f"\nTest Macro F1: {test_f1:.4f}\n")

        # Save Confusion Matrix
        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, test_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(os.path.join(args.output_dir, f"cm_{name}.csv"))

        results[name] = {"test_macro_f1": test_f1, "accuracy": acc}

    # Save summary
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
