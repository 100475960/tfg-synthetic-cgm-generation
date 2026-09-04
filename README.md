# Synthetic CGM Data Generation for Type 1 Diabetes

Bachelor's Thesis — Universidad Carlos III de Madrid, 2026  
*Synthetic Data Generation and Validation Using Deep Generative Models for Diabetes Datasets*

## Overview

This repository contains the code developed for the experimental work of the thesis. The project studies how specific architectural choices within TimeGAN affect the statistical fidelity of synthetic continuous glucose monitoring (CGM) time series, applied to the OhioT1DM dataset (12 patients with type 1 diabetes).

The central finding is that replacing the deterministic autoencoder with a VAE encoder resolves mode collapse structurally, by regularising the latent space toward N(0, I) via KL divergence.

## Dataset

The OhioT1DM dataset is required to run the preprocessing pipeline. It is not included in this repository, it can be accessed via [OhioT1DM](https://www.kaggle.com/datasets/ryanmouton/ohiot1dm/data).

## Repository structure

| File | Description |
|---|---|
| `xml_to_csv.ipynb` | Converts OhioT1DM XML files to CSV format |
| `preprocess_dataset.ipynb` | Builds normalised multivariate windows from CSV files |
| `timegan_baseline.ipynb` | Baseline TimeGAN with deterministic autoencoder and BCE loss |
| `timegan_vae_reference.ipynb` | Reference VAE-TimeGAN with WGAN-GP and Wiener noise |
| `timegan_multipatient.ipynb` | Parametrisable notebook for multi-patient validation (used by the orchestrator) |
| `run_all_patients.py` | Orchestrator, runs `timegan_multipatient.ipynb` across all 12 patients |
| `build_summary_table.py` | Aggregates per-run metrics into a single CSV table |
| `multipatient_validation_results.csv` | Pre-computed validation results for all 12 patients |

## Requirements

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib nbformat nbclient jupyter ipykernel
```

## Pipeline

```
xml_to_csv.ipynb → preprocess_dataset.ipynb → timegan_multipatient.ipynb (via run_all_patients.py) → build_summary_table.py
```
