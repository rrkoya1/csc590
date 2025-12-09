# src/llm_utils_local.py
from __future__ import annotations
import json
import re
import requests
from typing import Any, Dict, List, Optional

# -----------------------------
# Label Definitions
# -----------------------------
TN_ONLY_LABELS = ["T1", "T2", "T3", "T4"]
N_ONLY_LABELS  = ["N0", "N1", "N2", "N3"]
M_ONLY_LABELS  = ["M0", "M1"]
TNM_ALL_LABELS = TN_ONLY_LABELS + N_ONLY_LABELS + M_ONLY_LABELS

# -----------------------------
# Prompt Handling
# -----------------------------
def load_prompt_json(path: str) -> Any:
    """Load a JSON prompt file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _inject(template: str, text: str) -> str:
    if not template:
        return ""
    # Replace common placeholders, case-insensitive variants
    return (
        template
        .replace("{{REPORT}}", text)
        .replace("{{report}}", text)
        .replace("{REPORT}", text)
        .replace("{report}", text)
    )

def build_prompt(prompt_obj: Any, report_text: str) -> Dict[str, Any]:
    """
    Build an Ollama /api/chat payload 'messages' list from:
      - list of {role, content} dicts, OR
      - dict with 'system'/'user', OR
      - raw string
    Injects the report text into {{REPORT}} / {report} if present;
    otherwise appends a user message with the report.
    """
    messages: List[Dict[str, str]] = []

    # Case 1: Already a chat-style list
    if isinstance(prompt_obj, list):
        for m in prompt_obj:
            role = m.get("role", "user")
            content = _inject(m.get("content", ""), report_text)
            messages.append({"role": role, "content": content})

        # If no placeholder was used, make sure the report is present
        has_report = any("{{REPORT}}" in m.get("content", "") or "{report}" in m.get("content", "") for m in prompt_obj)
        if not has_report:
            messages.append({"role": "user", "content": f"\n\nREPORT:\n{report_text}"})
        return {"messages": messages}

    # Case 2: Dict with 'system'/'user'
    if isinstance(prompt_obj, dict):
        sys_content = prompt_obj.get("system", "")
        usr_content = prompt_obj.get("user") or prompt_obj.get("instruction") or ""
        usr_content = _inject(usr_content, report_text)

        if sys_content:
            messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": usr_content if usr_content else f"REPORT:\n{report_text}"})
        return {"messages": messages}

    # Case 3: Plain string
    s = str(prompt_obj)
    if "{{REPORT}}" in s or "{report}" in s:
        s = _inject(s, report_text)
    else:
        s = s + "\n\nREPORT:\n" + report_text
    return {"messages": [{"role": "user", "content": s}]}

# -----------------------------
# Output Parsing
# -----------------------------
# Strict patterns like T1..T4, N0..N3, M0..M1 (avoid matching words like "is")
STRICT_LABEL_RE = re.compile(r"\b([TMN][0-4])\b", re.IGNORECASE)

def normalize_label(s: str) -> Optional[str]:
    """Extract a canonical TNM label like 'T3' from free text."""
    if not s:
        return None
    s = s.upper().replace("STAGE", "").replace(" ", "").strip()
    m = STRICT_LABEL_RE.search(s)
    return m.group(1).upper() if m else None

def parse_label_from_text(text: str, allowed_labels: List[str]) -> str:
    """
    Heuristic parser (tolerant). Tries exact match, regex, then contains,
    then falls back to rough numeric hint, else first label.
    """
    if not text:
        return allowed_labels[0]
    clean = text.strip().upper()

    # Exact
    if clean in allowed_labels:
        return clean

    # Regex strict
    found = normalize_label(clean)
    if found and found in allowed_labels:
        return found

    # Contains
    for lab in allowed_labels:
        if lab in clean:
            return lab

    # Numeric hint (works fine for T-only)
    for lab in ["T1", "T2", "T3", "T4", "N0", "N1", "N2", "N3", "M0", "M1"]:
        if lab in allowed_labels and lab[1] in clean:
            return lab

    return allowed_labels[0]

def parse_label_strict(text: str, allowed_labels: List[str]) -> str:
    """
    Stricter parser for few-shot: requires a strict TNM token present.
    Falls back to heuristic if needed.
    """
    if not text:
        return allowed_labels[0]
    clean = text.strip().upper()
    m = STRICT_LABEL_RE.search(clean)
    if m:
        cand = m.group(1).upper()
        if cand in allowed_labels:
            return cand
    # fallback to tolerant
    return parse_label_from_text(text, allowed_labels)

# -----------------------------
# Ollama API Client
# -----------------------------
def _post_ollama_chat(host: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{host.rstrip('/')}/api/chat"
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()

def call_ollama_raw(
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Low-level call for few-shot where we construct messages ourselves.
    Returns the raw assistant content.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    if options:
        payload["options"].update(options)

    try:
        result = _post_ollama_chat(host, payload)
        return result.get("message", {}).get("content", "") or ""
    except requests.RequestException as e:
        print(f"[ollama] request failed: {e}")
        return ""

def classify_report_ollama(
    host: str,
    model: str,
    prompt_obj: Any,
    report_text: str,
    allowed_labels: List[str],
    temperature: float = 0.0,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    max_chars: int = 8000,
) -> str:
    """
    Zero-shot helper using a prompt template (prompt_obj).
    """
    # Truncate to avoid context issues
    if len(report_text) > max_chars:
        report_text = report_text[:max_chars]

    chat_payload = build_prompt(prompt_obj, report_text)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": chat_payload["messages"],
        "stream": False,
        "options": {"temperature": temperature, "top_p": top_p},
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    try:
        result = _post_ollama_chat(host, payload)
        llm_output = result.get("message", {}).get("content", "") or ""
        return parse_label_from_text(llm_output, allowed_labels)
    except requests.RequestException as e:
        print(f"[ollama] request failed: {e}")
        return allowed_labels[0]
