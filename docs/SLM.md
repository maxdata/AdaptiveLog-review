# Small Model (SLM) with SentenceTransformer

This document explains how the repo uses SentenceTransformer as a supervised small model (SLM) to classify logs, why it’s chosen over small GPTs for these tasks, and how to train/evaluate it.

## What the SLM Does

- Acts as a fast, local classifier/regressor for log analysis tasks:
  - LDSM (binary): log ↔ description match.
  - MC (multi‑class): masked module prediction.
  - LP (binary): masked severity level (Error/Info).
  - LPCR (regression/ranking): relevance scores for five causes.
  - AD (binary): sequence anomaly detection (variant).
- Produces calibrated probabilities (via softmax) that drive uncertainty routing.

## Why SentenceTransformer

- Encoder‑based and bidirectional (e.g., BERT), well‑suited for short, structured log text.
- Efficient fine‑tuning for supervised objectives; easy to deploy on CPU/GPU.
- Deterministic logits and straightforward uncertainty estimation (MC‑Dropout).
- Low per‑sample cost; can run fully on‑prem.

Compared to small GPTs (decoder‑only): classification often lags encoder baselines on these inputs; generative outputs are slower, costlier, and harder to calibrate.

## Key Files

- Training: `ldsm_small_model_train.py`
- Inference: `ldsm_small_model_pred.py`
- Core library (extended): `sentence_transformers/SentenceTransformer.py`
- Evaluator (accuracy/F1): `sentence_transformers/evaluation/LabelAccuracyEvaluator.py`

## Training

Example (LDSM):

```
python ldsm_small_model_train.py \
  --train_data ./datasets/adaptive_ldsm/ldsm_hwswitch_train.json \
  --dev_data   ./datasets/adaptive_ldsm/ldsm_hwswitch_dev.json \
  --pretrain_model bert-base-uncased \
  --epoch 3 \
  --batch_size 16 \
  --outfolder ldsm_slm.pt
```

Notes:
- The repo extends `SentenceTransformer.fit` to optionally save the trained classification head for later reuse in uncertainty estimation (see `sentence_transformers/SentenceTransformer.py:805` and `sentence_transformers/SentenceTransformer.py:991`).
- Tasks use `SoftmaxLoss` for classification; LPCR may use regression/ranking variants.

## Evaluation

Example (LDSM):

```
python ldsm_small_model_pred.py \
  --test_data ./datasets/adaptive_ldsm/ldsm_hwswitch_test.json \
  --pretrain_model ldsm_slm.pt
```

`LabelAccuracyEvaluator` computes Accuracy and F1 using the saved classification head (`sentence_transformers/evaluation/LabelAccuracyEvaluator.py:1`).

## Uncertainty (Preview)

The SLM supports MC‑Dropout for uncertainty estimation by running multiple stochastic forward passes and measuring variance, used by the router to decide when to call the LLM. See `docs/UNCERTAINTY_ROUTING.md`.

## Not for Clustering (Here)

Although SentenceTransformer can produce embeddings for clustering, this repo uses it as a supervised predictor. Clustering is not part of this codepath.

## Practical Tips

- Start with smaller encoders (DistilBERT/MPNet small) for better throughput.
- Calibrate classification thresholds on your dev set.
- Keep tokenization consistent with training; logs here already replace values with parameter types (e.g., `[IPADDR]`).
- Re‑fine‑tune periodically as log patterns drift.

