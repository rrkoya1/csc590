# CSC 590 Master's Project: Comparative Analysis of Transformers and LLMs for Automated Cancer Staging

## 🔬 Project Overview
This repository contains the source code and final results for the **CSC 590 Master's Project** at California State University, Dominguez Hills.

The project systematically benchmarks three modeling paradigms for extracting **TNM staging** (Tumor, Node, Metastasis) from unstructured clinical pathology reports:
1.  **Classical Baseline:** TF-IDF + Linear SVM
2.  **Specialist Encoder:** Fine-Tuned ClinicalBERT (Bio_ClinicalBERT)
3.  **Generalist LLM:** Llama 3 (Zero-Shot & Few-Shot Prompting)

### Key Contributions & Findings
* **Specialist Performance:** The fine-tuned ClinicalBERT achieved the highest Macro-F1 scores (up to **0.75**) on complex T-Stage classification.
* **Efficiency Bottleneck:** LLM inference (Llama 3) was found to be **273x slower** than the specialized encoder, highlighting a major barrier to generative AI in high-throughput clinical use.
* **Robustness:** Modular ClinicalBERT ensembles proved significantly more robust than monolithic LLM prompts, which suffered from severe **Task Interference** (F1 collapse to ~0.20 on combined tasks).

## 🛠️ Repository Structure & Usage
The project adheres to best practices by isolating source code, data, and results.

| Folder/File | Purpose | Key Script |
| :--- | :--- | :--- |
| `src/modeling/` | Core training and fine-tuning scripts. | `fine_tune_bert.py`, `train_baseline_svm.py` |
| `src/evaluation/` | Scripts for running inference on LLMs and calculating final metrics. | `evaluate_llm.py`, `evaluate_ensemble.py` |
| `src/preprocessing/` | Scripts for data cleaning, merging labels, and creating fixed splits. | `merge_data.py`, `create_splits.py` |
| `results/figures/` | **Final Visualizations** (CMs, Comparison Bar Charts, Calibration Plots). | `final_thesis_comparison.png`, `calibration_clinicalbert_T.png` |
| `results/` | Final metric tables used for the report (Source of Truth). | `final_thesis_results.csv`, `final_runtime_analysis.csv` |

Installation:
git clone https://github.com/rrkoya1/csc590.git

##  Getting Started

### 1. Data Acquisition & Setup
**Note:** Due to patient data privacy (HIPAA compliance), **raw TCGA data files are excluded.**
* Acquire the raw **TCGA Pathology Reports** and staging labels (e.g., `TCGA_M01_patients.csv`) from the official TCGA database.
* Place the acquired files (see `.gitignore` for list) inside the `data/datasets/` folder.

### 2. Example Execution (ClinicalBERT Training)
The following command runs the core specialist training task (T-Stage classification):
```bash
python src/modeling/fine_tune_bert.py \
    --model_name_or_path "emilyalsentzer/Bio_ClinicalBERT" \
    --label_col t_stage \
    --output_dir outputs/clinicalbert_t_stage_final \
    --num_train_epochs 4