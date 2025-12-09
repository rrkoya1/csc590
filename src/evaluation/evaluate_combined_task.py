# src/eval_combined_llm.py
import argparse
import json
import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from llm_utils_local import load_prompt_json, build_prompt, _post_ollama_chat

# -----------------------------
# Mapping & Parsing
# -----------------------------
INT_TO_T = {"1": "T1", "2": "T2", "3": "T3", "4": "T4"}
INT_TO_N = {"0": "N0", "1": "N1", "2": "N2", "3": "N3"}
INT_TO_M = {"0": "M0", "1": "M1"}

def parse_combined_json(llm_output):
    """
    Robust JSON parser for Llama 3 output.
    Returns dict: {"T": "T1", "N": "N0", "M": "M0"} (or "Error")
    """
    preds = {"T": "Error", "N": "Error", "M": "Error"}
    try:
        # 1. Clean Markdown
        clean = llm_output.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        
        # 2. Extract & Map T
        if "T" in data and "label" in data["T"]:
            val = str(data["T"]["label"])
            preds["T"] = INT_TO_T.get(val, f"T{val}") # Keep original if not in map
            
        # 3. Extract & Map N
        if "N" in data and "label" in data["N"]:
            val = str(data["N"]["label"])
            preds["N"] = INT_TO_N.get(val, f"N{val}")

        # 4. Extract & Map M
        if "M" in data and "label" in data["M"]:
            val = str(data["M"]["label"])
            preds["M"] = INT_TO_M.get(val, f"M{val}")
            
    except Exception:
        pass # Return Errors
    return preds

# -----------------------------
# Few-Shot Construction
# -----------------------------
def get_fewshot_examples(train_csv, k=3, seed=42):
    """Selects k random examples from the training set."""
    if not train_csv or not os.path.exists(train_csv):
        return []
    
    df = pd.read_csv(train_csv).dropna(subset=['text', 'tnm_label'])
    # Simple random sample
    sample = df.sample(n=k, random_state=seed)
    
    examples = []
    for _, row in sample.iterrows():
        # Construct the "Gold" JSON response for this example
        # Note: We need to reverse-map labels (T1->1) to match the prompt instructions
        # But for simplicity, let's assume the model learns from the string or we construct valid JSON.
        # Ideally, we parse the tnm_label (e.g. T1N0M0) into components.
        
        # Quick parse of combined string (e.g. T1N0M0)
        t_val = row['t_label'] # e.g. T1
        n_val = row['n_label'] # e.g. N0
        m_val = row['m_label'] # e.g. M0
        
        # Convert back to int for the prompt example
        t_int = next((k for k,v in INT_TO_T.items() if v == t_val), "1")
        n_int = next((k for k,v in INT_TO_N.items() if v == n_val), "0")
        m_int = next((k for k,v in INT_TO_M.items() if v == m_val), "0")
        
        gold_json = json.dumps({
            "T": {"label": int(t_int), "explanation": "Extracted from history."},
            "N": {"label": int(n_int), "explanation": "Extracted from history."},
            "M": {"label": int(m_int), "explanation": "Extracted from history."}
        }, indent=2)
        
        examples.append((row['text'], gold_json))
        
    return examples

def build_fewshot_prompt(prompt_obj, target_text, examples):
    """Injects examples into the prompt history."""
    # Base prompt (System + Instruction)
    base = build_prompt(prompt_obj, "") 
    messages = base['messages']
    
    # Insert Examples before the final User query
    # Format: User(Report) -> Assistant(JSON)
    history = []
    for ex_text, ex_json in examples:
        history.append({"role": "user", "content": f"REPORT:\n{ex_text}"})
        history.append({"role": "assistant", "content": ex_json})
        
    # Combine: [System/Instruct] + [Examples] + [Target Report]
    # We assume messages[0] is system/instruct. We append to it.
    final_messages = messages[:-1] + history + [{"role": "user", "content": f"REPORT:\n{target_text}"}]
    return {"messages": final_messages}

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--train_csv", default=None, help="Path to training set for few-shot examples")
    parser.add_argument("--prompt_json", required=True)
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fewshot_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Data
    test_df = pd.read_csv(args.test_csv)
    print(f"Loaded {len(test_df)} test samples.")

    # 2. Load Prompt
    prompt_obj = load_prompt_json(args.prompt_json)
    
    # 3. Prepare Few-Shot Examples (if k > 0)
    examples = []
    if args.fewshot_k > 0 and args.train_csv:
        print(f"Selecting {args.fewshot_k} few-shot examples from {args.train_csv}...")
        examples = get_fewshot_examples(args.train_csv, k=args.fewshot_k, seed=args.seed)

    # 4. Inference
    results = []
    print(f"Running Combined Inference (k={args.fewshot_k})...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        text = row['text']
        
        # Build Payload
        if examples:
            chat_payload = build_fewshot_prompt(prompt_obj, text[:8000], examples)
        else:
            chat_payload = build_prompt(prompt_obj, text[:8000])
            
        payload = {
            "model": args.model,
            "messages": chat_payload["messages"],
            "stream": False,
            "options": {"temperature": 0.0},
            "format": "json" 
        }
        
        try:
            import requests
            url = f"{args.ollama_host.rstrip('/')}/api/chat"
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            
            llm_output = resp.json().get("message", {}).get("content", "")
            parsed = parse_combined_json(llm_output)
            
            results.append({
                "true_T": row['t_label'], "pred_T": parsed['T'],
                "true_N": row['n_label'], "pred_N": parsed['N'],
                "true_M": row['m_label'], "pred_M": parsed['M'],
                "raw_output": llm_output
            })
            
        except Exception as e:
            results.append({"pred_T": "Error", "pred_N": "Error", "pred_M": "Error"})

    # 5. Save & Score
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(args.out_dir, "predictions_combined.csv"), index=False)
    
    # Filter out errors for scoring
    valid = res_df[res_df["pred_T"] != "Error"]
    print(f"Valid parsed responses: {len(valid)}/{len(res_df)}")
    
    print("\n--- T-Stage ---")
    print(classification_report(valid['true_T'], valid['pred_T'], digits=4))
    print("\n--- N-Stage ---")
    print(classification_report(valid['true_N'], valid['pred_N'], digits=4))
    print("\n--- M-Stage ---")
    print(classification_report(valid['true_M'], valid['pred_M'], digits=4))

if __name__ == "__main__":
    main()