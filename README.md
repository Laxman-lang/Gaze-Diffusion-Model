# Diffusion Model for Eye-Gaze Trajectory Generation

Master's thesis implementation — OsloMet, ACIT 2026

## Author
Laxman Basnet

## Description
Unconditional diffusion model for synthetic eye-gaze trajectory generation,
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

## Pipeline

The notebook follows a sequential pipeline. Run cells from top to bottom in order:

| Step | Description |
|------|-------------|
| Step 1 | Imports, configuration, and seed setting |
| Step 2 | Dataset loading and preprocessing (downsampling, normalisation, segmentation, augmentation) |
| Step 3 | Diffusion schedule definition (DDPM noise schedule) |
| Step 4 | Model architecture (U-Net1D with FiLM conditioning) |
| Step 5 | Training (500 epochs, EMA, mixed precision) |
| Step 6 | Generation via DDIM sampling |
| Step 7 | Evaluation metrics (KS, JS, Wasserstein, FGD) |
| Step 8 | Fixation and saccade analysis |
| Step 9 | Ablation studies (ε vs x₀, EMA, DDIM steps, η) |
| Step 10 | Baseline comparisons (Gaussian, Random, Markov) |
| Step 11 | Spatial coverage evaluation |
| Step 12 | TSTR downstream evaluation |

## Compute Environment
This notebook was developed and executed on the **Simula Research Laboratory HPC cluster (ex3)**
using an **NVIDIA A40 GPU**.

The notebook is provided for transparency and reproducibility of the thesis results.
To run locally, a CUDA-capable GPU is recommended with the following dependencies:
- Python 3.10+
- PyTorch 2.5+
- numpy
- scipy
- matplotlib
- tqdm

## Note on Outputs
Outputs shown in the notebook are from the original training run on ex3. The eye-tracking dataset is not included as it belongs to Mathema et al. and is not 
available for redistribution. Please refer to the original paper for dataset access.
