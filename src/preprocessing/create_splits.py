# src/make_fixed_split.py
import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--text_col", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.csv_path}...")
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if len(df) == 0:
        print("Error: Input CSV is empty.")
        return

    # Basic Cleaning
    df = df.dropna(subset=[args.text_col, args.label_col])
    df = df.drop_duplicates(subset=[args.text_col])
    df = df.reset_index(drop=True)

    # Top-N Filtering
    if args.top_n:
        counts = df[args.label_col].value_counts()
        top_classes = counts.head(args.top_n).index.tolist()
        print(f"Filtering to Top {args.top_n} classes: {top_classes}")
        df = df[df[args.label_col].isin(top_classes)].reset_index(drop=True)

    # Splits (80/10/10)
    train_val, test = train_test_split(
        df, test_size=args.test_size, stratify=df[args.label_col], random_state=args.seed
    )
    
    relative_val = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val[args.label_col], random_state=args.seed
    )

    # Save
    df.to_csv(os.path.join(args.out_dir, "filtered.csv"), index=False)
    
    split = {
        "train_ids": train.index.tolist(),
        "val_ids": val.index.tolist(),
        "test_ids": test.index.tolist()
    }
    
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(split, f)

    print(f"Saved split artifacts to {args.out_dir}")
    print("Label Distribution:\n", df[args.label_col].value_counts())

if __name__ == "__main__":
    main()