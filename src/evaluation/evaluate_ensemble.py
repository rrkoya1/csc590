
import argparse
import pandas as pd
import os
import glob
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import classification_report

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        return self.tokenizer(self.texts[i], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")

def find_best_checkpoint(base_path):
    """
    Automatically finds the model weights.
    1. Checks the base path.
    2. If not found, looks for the 'checkpoint-X' subfolder with the highest number.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Directory not found: {base_path}")

    # Check if model exists in root
    if os.path.exists(os.path.join(base_path, "model.safetensors")) or \
       os.path.exists(os.path.join(base_path, "pytorch_model.bin")):
        return base_path

    # Search for checkpoints
    checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No model weights or checkpoints found in {base_path}")

    # Sort by step number (checkpoint-100, checkpoint-2000)
    def get_step(path):
        try:
            return int(path.split("-")[-1])
        except:
            return 0
            
    best_checkpoint = max(checkpoints, key=get_step)
    print(f"  -> Found checkpoint: {os.path.basename(best_checkpoint)}")
    return best_checkpoint

def predict(base_model_path, texts, device="cuda"):
    # 1. Resolve actual path (handle checkpoints)
    model_path = find_best_checkpoint(base_model_path)
    
    print(f"Loading from: {model_path}...")
    
    # 2. Load Tokenizer (Safe Fallback)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"  -> Warning: Local tokenizer failed. Downloading base 'emilyalsentzer/Bio_ClinicalBERT'...")
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # 3. Load Model
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    # 4. Inference
    ds = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=32)
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            mask = batch['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids, attention_mask=mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            
    # Map IDs back to Labels
    id2label = model.config.id2label
    return [id2label[p] for p in preds]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--model_t", required=True)
    parser.add_argument("--model_n", required=True)
    parser.add_argument("--model_m", required=True)
    parser.add_argument("--out_csv", default="outputs/predictions_bert_ensemble.csv")
    args = parser.parse_args()

    print(f"Loading Data: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    texts = df['text'].astype(str).tolist()
    
    # Run Inference
    print("\n=== Running T-Stage Model ===")
    pred_t = predict(args.model_t, texts)
    
    print("\n=== Running N-Stage Model ===")
    pred_n = predict(args.model_n, texts)
    
    print("\n=== Running M-Stage Model ===")
    pred_m = predict(args.model_m, texts)
    
    # Save Results
    out = df.copy()
    out['pred_T'] = pred_t
    out['pred_N'] = pred_n
    out['pred_M'] = pred_m
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    
    print("\n" + "="*30)
    print("FINAL ENSEMBLE RESULTS (Exact Match Check)")
    print("="*30)
    print("T-Stage Report:\n", classification_report(out['t_label'], out['pred_T'], digits=4))
    print("N-Stage Report:\n", classification_report(out['n_label'], out['pred_N'], digits=4))
    print("M-Stage Report:\n", classification_report(out['m_label'], out['pred_M'], digits=4))

if __name__ == "__main__":
    main()
