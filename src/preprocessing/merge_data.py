import os
import re
import argparse
import pandas as pd

def norm_barcode(x: str) -> str:
    """Return the canonical 12-char TCGA patient barcode (e.g., 'TCGA-AB-1234')."""
    if not isinstance(x, str):
        return None
    x = x.strip().upper()
    return x[:12]  # TCGA-XX-XXXX

def clean_t(v):
    """Map T2a/T3b → T2/T3; keep only T1..T4."""
    if not isinstance(v, str):
        return None
    v = v.strip().upper()
    m = re.match(r"^T([0-4])", v)
    return f"T{m.group(1)}" if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_csv", required=True, help="Path to TCGA_Reports.csv")
    ap.add_argument("--tnm_t_csv", required=True, help="Path to T-stage patient file")
    ap.add_argument("--out_dir", default="./data/tnm_T")
    ap.add_argument("--out_csv", default="filtered_tnm_T.csv")

    # NEW: explicit column overrides (recommended)
    ap.add_argument("--reports_pid_col", default=None, help="Patient ID col in reports CSV")
    ap.add_argument("--reports_text_col", default=None, help="Text col in reports CSV")
    ap.add_argument("--tnm_pid_col", default=None, help="Patient ID col in TNM CSV")
    ap.add_argument("--tnm_t_col", default=None, help="T-stage col in TNM CSV")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    rep = pd.read_csv(args.reports_csv)
    tdf = pd.read_csv(args.tnm_t_csv)

    # If user didn’t pass names, try sensible fallbacks
    def pick(df, user, candidates, what):
        if user and user in df.columns:
            return user
        for c in candidates:
            if c in df.columns: return c
        raise ValueError(f"Could not find {what}. Tried: { [user] + candidates } ; Available: {list(df.columns)}")

    rep_pid = pick(
        rep,
        args.reports_pid_col,
        ["bcr_patient_barcode","case_submitter_id","submitter_id","patient","patient_id","case_id","tcga_id","case_barcode"],
        "reports patient-id column"
    )
    rep_txt = pick(
        rep,
        args.reports_text_col,
        ["text","report_text","note","content","report","report_content","pathology_text"],
        "reports text column"
    )
    tnm_pid = pick(
        tdf,
        args.tnm_pid_col,
        ["bcr_patient_barcode","case_submitter_id","submitter_id","patient","patient_id","case_id","tcga_id","case_barcode"],
        "TNM patient-id column"
    )
    tnm_t = pick(
        tdf,
        args.tnm_t_col,
        ["T","t_stage","T_stage","pathologic_T","pathologic_t","ajcc_pathologic_t","ajcc_t","ajcc_t_pathologic"],
        "TNM T-stage column"
    )

    # Normalize IDs
    rep["pid12"] = rep[rep_pid].map(norm_barcode)
    tdf["pid12"] = tdf[tnm_pid].map(norm_barcode)

    # Clean T labels to T1..T4
    tdf["t_stage_clean"] = tdf[tnm_t].map(clean_t)

    # Merge and filter
    merged = rep.merge(tdf[["pid12","t_stage_clean"]], on="pid12", how="left")
    out = merged.dropna(subset=["t_stage_clean"]).copy()
    out = out.rename(columns={rep_txt: "text", "t_stage_clean": "t_stage"})
    out = out[["text","t_stage", rep_pid]].dropna(subset=["text"])

    out_path = os.path.join(args.out_dir, args.out_csv)
    out.to_csv(out_path, index=False)

    print(f"Saved merged TNM T-stage file: {out_path}")
    print("Label distribution:\n", out["t_stage"].value_counts())

if __name__ == "__main__":
    main()
