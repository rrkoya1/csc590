# src/eval_zero_shot_llm.py
from __future__ import annotations
import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_utils_local import (
    load_prompt_json,
    build_prompt,
    classify_report_ollama,
    call_ollama_raw,
    parse_label_strict,
    TN_ONLY_LABELS,
    N_ONLY_LABELS,
    M_ONLY_LABELS,
)

# -------------------------------
# Robust split loader
# -------------------------------
def _pick_key(d: Dict[str, Any], candidates: List[str]) -> List[int]:
    for k in candidates:
        if k in d:
            return d[k]
    if "indices" in d and isinstance(d["indices"], dict):
        for k in candidates:
            if k in d["indices"]:
                return d["indices"][k]
    raise KeyError(f"None of the keys {candidates} found in split JSON. Available: {list(d.keys())}")

def load_split(filtered_csv: str, split_json: str, text_col: str, label_col: str):
    df = pd.read_csv(filtered_csv)
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_idx = [int(x) for x in _pick_key(split, ["train", "train_ids", "train_idx", "idx_tr", "train_indices"])]
    val_idx   = [int(x) for x in _pick_key(split, ["val", "val_ids", "val_idx", "idx_va", "validation"])]
    test_idx  = [int(x) for x in _pick_key(split, ["test", "test_ids", "test_idx", "idx_te"])]

    return (
        df.iloc[train_idx][[text_col, label_col]].reset_index(drop=True),
        df.iloc[val_idx][[text_col, label_col]].reset_index(drop=True),
        df.iloc[test_idx][[text_col, label_col]].reset_index(drop=True),
    )

# -------------------------------
# Metrics
# -------------------------------
def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    uniq = sorted(list(set(y_true + y_pred)))
    f1s = []
    for lab in uniq:
        tp = sum((yt == lab) and (yp == lab) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != lab) and (yp == lab) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == lab) and (yp != lab) for yt, yp in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

# -------------------------------
# Few-shot helpers
# -------------------------------
def stratified_fewshot(
    train_df: pd.DataFrame,
    label_col: str,
    k: int,
    seed: int,
) -> List[Tuple[str, str]]:
    """
    Returns k examples total, roughly balanced across classes.
    """
    rng = random.Random(seed)
    groups = {lab: [] for lab in sorted(train_df[label_col].astype(str).unique())}
    for _, row in train_df.sample(frac=1.0, random_state=seed).iterrows():
        lab = str(row[label_col])
        txt = str(row[train_df.columns[0]])  # assume first col is text_col (we pass filtered dfs)
        groups[lab].append((txt, lab))

    # round-robin pick
    labels = list(groups.keys())
    shots: List[Tuple[str, str]] = []
    i = 0
    while len(shots) < k and any(groups[l] for l in labels):
        lab = labels[i % len(labels)]
        if groups[lab]:
            shots.append(groups[lab].pop(0))
        i += 1
    return shots

def is_chat_list(prompt_obj: Any) -> bool:
    return isinstance(prompt_obj, list)

def extract_system_from_chat(prompt_obj: List[Dict[str, str]]) -> str:
    """
    Take the first 'system' content if present, else empty string.
    """
    for m in prompt_obj:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""

def extract_instruction_from_chat(prompt_obj: List[Dict[str, str]]) -> str:
    """
    Take the last 'user' message content as instruction if present, else empty string.
    """
    last_user = ""
    for m in prompt_obj:
        if m.get("role") == "user":
            last_user = m.get("content", "")
    return last_user

def format_labels_instruction(allowed: List[str]) -> str:
    return "Valid labels: " + ", ".join(allowed) + ". Respond with exactly one of these tokens."

def make_fewshot_messages_from_chat_template(
    prompt_obj: List[Dict[str, str]],
    shots: List[Tuple[str, str]],
    test_report: str,
    allowed_labels: List[str],
) -> List[Dict[str, str]]:
    """
    Use a chat-style prompt template (list of messages). We:
      - Keep a system message if present
      - Use the last user message as the general instruction
      - Append k pairs of (user: shot report, assistant: gold label)
      - Append final user with the test report and an explicit 'valid labels' line
    """
    system_base = extract_system_from_chat(prompt_obj)
    instruction = extract_instruction_from_chat(prompt_obj)

    messages: List[Dict[str, str]] = []
    if system_base:
        messages.append({"role": "system", "content": system_base})

    # Instruction block (remind valid labels)
    instr = instruction.replace("{{REPORT}}", "").replace("{report}", "")
    instr = instr.strip()
    if instr:
        instr += "\n\n" + format_labels_instruction(allowed_labels)
    else:
        instr = format_labels_instruction(allowed_labels)
    messages.append({"role": "user", "content": instr})

    # Few-shot exemplars
    for txt, lab in shots:
        # keep exemplars compact
        exemplar_user = f"REPORT:\n{txt}\n\nPlease classify the T-stage."
        messages.append({"role": "user", "content": exemplar_user})
        messages.append({"role": "assistant", "content": lab})

    # Final query
    final_user = f"REPORT:\n{test_report}\n\nClassify the T-stage. {format_labels_instruction(allowed_labels)}"
    messages.append({"role": "user", "content": final_user})
    return messages

def make_fewshot_messages_from_generic_template(
    prompt_obj: Any,
    test_report: str,
    shots: List[Tuple[str, str]],
    allowed_labels: List[str],
) -> List[Dict[str, str]]:
    """
    If the prompt is dict or string, reuse build_prompt() to get a base chat,
    then append few-shot turns + final user turn.
    """
    base = build_prompt(prompt_obj, "")  # inject later per message
    messages: List[Dict[str, str]] = base.get("messages", []).copy()

    # append instruction with valid labels
    messages.append({
        "role": "user",
        "content": format_labels_instruction(allowed_labels)
    })

    # shots
    for txt, lab in shots:
        messages.append({"role": "user", "content": f"REPORT:\n{txt}\n\nPlease classify the T-stage."})
        messages.append({"role": "assistant", "content": lab})

    # final query
    messages.append({"role": "user", "content": f"REPORT:\n{test_report}\n\nClassify the T-stage. {format_labels_instruction(allowed_labels)}"})
    return messages

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered_csv", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--text_col", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--prompt_json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    ap.add_argument("--tn_only", action="store_true")
    ap.add_argument("--n_only", action="store_true")
    ap.add_argument("--m_only", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_chars", type=int, default=8000)
    ap.add_argument("--fewshot_k", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Label space
    if args.tn_only:
        labels = TN_ONLY_LABELS
        task_name = "T-only"
    elif args.n_only:
        labels = N_ONLY_LABELS
        task_name = "N-only"
    elif args.m_only:
        labels = M_ONLY_LABELS
        task_name = "M-only"
    else:
        labels = TN_ONLY_LABELS
        task_name = "T-only (default)"

    print(f"[eval] Task: {task_name}")
    print(f"[eval] Allowed Labels: {labels}")

    # Data
    train_df, val_df, test_df = load_split(args.filtered_csv, args.split_json, args.text_col, args.label_col)
    test_texts = test_df[args.text_col].astype(str).tolist()
    golds = test_df[args.label_col].astype(str).tolist()
    print(f"[eval] Loaded {len(test_texts)} test samples.")

    # Prompt
    prompt_obj = load_prompt_json(args.prompt_json)

    preds: List[str] = []

    # Few-shot path
    if args.fewshot_k > 0:
        print(f"[eval] Running FEW-SHOT k={args.fewshot_k} on {args.model} ...")
        # Prepare K exemplars from train (balanced)
        train_small = train_df[[args.text_col, args.label_col]].rename(columns={args.text_col: "text", args.label_col: "label"})
        shots = stratified_fewshot(train_small, "label", args.fewshot_k, args.seed)

        for t in tqdm(test_texts, desc="LLM Few-shot Inference"):
            t_cut = t[: args.max_chars] if len(t) > args.max_chars else t

            if is_chat_list(prompt_obj):
                messages = make_fewshot_messages_from_chat_template(prompt_obj, shots, t_cut, labels)
            else:
                messages = make_fewshot_messages_from_generic_template(prompt_obj, t_cut, shots, labels)

            raw = call_ollama_raw(
                host=args.ollama_host,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            pred = parse_label_strict(raw, labels)
            preds.append(pred)

    # Zero-shot path
    else:
        print(f"[eval] Running ZERO-SHOT on {args.model} ...")
        for t in tqdm(test_texts, desc="LLM Inference"):
            pred = classify_report_ollama(
                host=args.ollama_host,
                model=args.model,
                prompt_obj=prompt_obj,
                report_text=t,
                allowed_labels=labels,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                max_chars=args.max_chars,
            )
            preds.append(pred)

    # Save predictions
    out_csv = os.path.join(args.out_dir, "pred_test.csv")
    pd.DataFrame({"text": test_texts, "gold": golds, "pred": preds}).to_csv(out_csv, index=False)
    print(f"[eval] Saved predictions to {out_csv}")

    # Metric
    mf1 = macro_f1(golds, preds)
    print(f"\n[eval] TEST MACRO-F1: {mf1:.4f}")

    # Summary
    out_table = os.path.join(args.out_dir, "results_table.csv")
    pd.DataFrame([{
        "model": args.model,
        "task": task_name,
        "fewshot_k": args.fewshot_k,
        "test_macro_f1": round(mf1, 4),
        "n_samples": len(golds),
    }]).to_csv(out_table, index=False)
    print(f"[eval] Saved summary to {out_table}")

if __name__ == "__main__":
    main()
