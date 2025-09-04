# AdaptiveLog Architecture

This document explains the overall architecture, workflow, and repository layout of AdaptiveLog. It complements the task‑specific docs in `docs/LDSM.md`, `docs/MC.md`, `docs/LP.md`, `docs/LPCR.md`, and `docs/AD.md`.

## Goals

- Maximize accuracy on log analysis tasks while minimizing cost and latency.
- Combine a fast, supervised Small Language Model (SLM) with a powerful Large Language Model (LLM).
- Route only uncertain cases to the LLM; keep “easy” ones on the SLM.

## High‑Level Workflow

1. Train SLM (SentenceTransformer with a classification/regression head) on labeled data.
2. Run SLM inference on incoming logs and estimate uncertainty with MC‑Dropout.
3. Route: If uncertain → send to the LLM with Error‑prone Case Retrieval (ECR) context; else keep SLM result.
4. Integrate final decisions and report metrics (accuracy, cost, latency).

```
Logs → SLM → {certain → accept; uncertain → LLM(+ECR)} → Final prediction
```

The figure in `fig/adaptive_framework.png` provides an overview.

## Why Hybrid (SLM + LLM)

- Cost/Latency: The SLM handles most traffic cheaply and fast; LLM is used sparingly.
- Calibration: SLM provides stable probabilities for routing; LLMs lack reliable calibrated confidences.
- Data Privacy: Only a fraction of logs are sent to an external API (or on‑prem LLM).
- Accuracy: LLM + ECR improves hard cases by leveraging domain reasoning and examples.

## Repository Layout (select)

- `ldsm_small_model_train.py`: Train SLM for LDSM.
- `ldsm_small_model_pred.py`: Evaluate a trained SLM.
- `ldsm_uncertain_pred.py`, `ldsm_uncertain_prob.py`: Uncertainty estimation and sample splitting.
- `query_ChatGPT.py`: Query the LLM with ECR prompts.
- `prompt_template/ecr/*.txt`: Prompt templates for ECR.
- `sentence_transformers/`: Customized SentenceTransformers with training extensions.
- `docs/*.md`: Task docs and this architecture series.

## Data Flow Per Task

1. Prepare labeled dataset for the task (e.g., LDSM pairs, MC module labels, LP severity labels).
2. Fine‑tune the SLM on training data; validate on dev set.
3. Run SLM on test/production logs; compute uncertainty.
4. Route high‑uncertainty samples to the LLM with relevant error‑prone cases (ECR) in the prompt.
5. Combine predictions and measure metrics incl. Accuracy/F1 and LLM call rate.

## Metrics

- Task metrics: Accuracy, F1, Precision/Recall, ranking metrics (for LPCR).
- Adaptive metrics: Fraction routed to LLM, incremental accuracy gain, cost per 1k logs, end‑to‑end latency.

## Implementation Notes

- `sentence_transformers/SentenceTransformer.py` adds `save_task` and `small_model_path` to persist the trained classification head for uncertainty estimation.
- Evaluators such as `sentence_transformers/evaluation/LabelAccuracyEvaluator.py` compute task metrics from the trained head.
- Scripts may require light path/argument adjustments for your environment (see `docs/SLM.md` and `docs/UNCERTAINTY_ROUTING.md`).

