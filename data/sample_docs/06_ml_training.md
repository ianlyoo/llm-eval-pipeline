# ML Model Training Guide (Internal)

## Environment
- Python 3.11, PyTorch 2.4, CUDA 12.1, Docker image `ml-train:2026a`.
- Experiment tracking: MLflow at mlflow.acme.example.

## Data
- Training data stored in S3 `s3://acme-ml/datasets/v3/`, versioned via DVC.
- Train/val/test split 80/10/10, stratified by label.

## Training Recipe
- Optimizer: AdamW (lr 3e-4, weight_decay 0.01), scheduler cosine with warmup 500 steps.
- Batch size 32, epochs 10, early stopping patience 2 (monitor val_loss).
- Mixed precision (bf16) enabled.

## Evaluation
- Metrics: accuracy, F1 macro, calibration ECE.
- Before/after comparison via `lm-eval` harness on held-out tasks.

## Reproducibility
- Seed 42, deterministic cuDNN, log git commit + data hash.

Keywords: PyTorch, MLflow, DVC, AdamW, cosine scheduler, early stopping, lm-eval, ECE.
