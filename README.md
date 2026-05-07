# DuoShao-Reproducibility-NeurIPS26

This repository contains the anonymous code artifact for the NeurIPS 2026 submission on Reviving Shapley Values for LLM Interpretability.
The code is organized around three experimental settings:

1. **Mechanistic Circuit Discovery**  
   Attribution over transformer components such as attention heads and MLPs for identifying task-relevant circuits.

2. **Training Data Quality Tracing**  
   In-run data Shapley estimation for identifying influential or harmful training examples during model training.

3. **Long-Context Localization**  
   Hierarchical paragraph- and sentence-level attribution for long-context reasoning tasks.

This repository includes code only. Large datasets, model checkpoints, cached activations, and experiment outputs are not included.

---

## Repository Structure

```text
.
├── data_quality_tracing/
│   ├── main_script/
│   ├── evaluation/
│   └── plotting/
│
├── long_context/
│   ├── main_script/
│   ├── evaluation/
│   └── plotting/
│
├── mechanistic_circuit_discovery/
│   ├── main_script/
│   └── plotting/
│
├── .gitignore
└── README.md
