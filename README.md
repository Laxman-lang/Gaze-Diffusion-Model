# Diffusion Model for Eye-Gaze Trajectory Generation

Master's thesis implementation — OsloMet, ACIT 2026

## Author
Laxman Basnet

## Description
Unconditional diffusion model (DDPM/DDIM) for synthetic eye-gaze trajectory generation, 
evaluated on the Where's Waldo eye-tracking dataset.

## Notebook
`Basnet_Laxman_394089_ACIT5900.ipynb` — full pipeline including:
- Data preprocessing
- Model architecture (U-Net1D with FiLM conditioning)
- Training (500 epochs, EMA)
- Evaluation (KS, JS, Wasserstein, FGD)
- Ablation studies
- Baseline comparisons
- TSTR downstream evaluation

## Requirements
- Python 3.10+
- PyTorch 2.5+
- scipy, numpy, matplotlib

## Note
Notebook outputs are from the original training run and match thesis results.
Dataset not included due to privacy restrictions.
