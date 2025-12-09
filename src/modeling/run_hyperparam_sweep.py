import argparse, itertools, os, subprocess, sys

def run_one(model, filtered, split, out_root, max_len, lr, epochs, batch, text_col, label_col):
    run_dir = os.path.join(out_root, f"lr{lr}_ep{epochs}")
    os.makedirs(run_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join("src", "train_transformer.py"),
        "--model_name_or_path", model,
        "--filtered_csv", filtered,
        "--split_json", split,
        "--output_dir", run_dir,
        "--max_length", str(max_len),
        "--num_train_epochs", str(epochs),
        "--learning_rate", str(lr),
        "--batch_size", str(batch),
        "--text_col", text_col,
        "--label_col", label_col,
    ]

    log_path = os.path.join(run_dir, "train.log")
    print(f"[run_sweep] Starting LR={lr}, EP={epochs}...")
    
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("CMD: " + " ".join(cmd) + "\n\n")
        # Standard subprocess run (Safe, no encoding crashes)
        rc = subprocess.run(cmd, stdout=logf, stderr=logf, text=True).returncode

    status = "ok" if rc == 0 else f"exit {rc}"
    print(f"[run_sweep] Finished LR={lr}, EP={epochs} -> {status} (log: {log_path})")
    return rc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--filtered", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lrs", nargs="+", default=["2e-5","3e-5","5e-5"])
    p.add_argument("--epochs", nargs="+", default=["2","3","4"])
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="cancer_type")
    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    for lr, ep in itertools.product(args.lrs, args.epochs):
        run_one(
            model=args.model,
            filtered=args.filtered,
            split=args.split,
            out_root=args.out_root,
            max_len=args.max_len,
            lr=lr,
            epochs=ep,
            batch=args.batch,
            text_col=args.text_col,
            label_col=args.label_col,
        )

    print("\nAll sweep runs completed.")

if __name__ == "__main__":
    main()
