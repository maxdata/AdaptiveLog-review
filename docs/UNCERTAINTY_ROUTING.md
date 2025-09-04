# Uncertainty Routing with MC‑Dropout

This document describes how AdaptiveLog estimates uncertainty in SLM predictions and routes only hard cases to the LLM.

## Motivation

LLM calls are accurate but expensive. The SLM handles most logs; we detect when it’s “unsure” and only then query the LLM.

## Method: MC‑Dropout

1. Enable dropout at inference (set the model to train mode for the head only).
2. Run N stochastic forward passes per sample (N≈10).
3. For each pass, record softmax probabilities.
4. Compute variance or dispersion of the predicted probability for the true/argmax class.
5. If variance > threshold → route to LLM; else accept SLM prediction.

## Implementation in Repo

- Estimation helpers: `ldsm_uncertain_pred.py`, `ldsm_uncertain_prob.py`.
- Pattern:
  - Batch inputs with `smart_batching_collate`.
  - For i in 1..N: forward pass, store `softmax(prediction)`.
  - Aggregate to get variance per sample, then split to `simple` and `hard` sets.

Key references:
- `ldsm_uncertain_pred.py:1`
- `ldsm_uncertain_prob.py:1`

Note: The scripts capture the intended algorithm; adapt paths/arguments for your environment and verify CLI flags (there are some mismatches in the sample code that are easy to fix locally).

## Threshold Tuning

1. On the dev set, sweep thresholds to trade off cost vs. accuracy.
2. Plot: fraction routed → vs. accuracy uplift.
3. Choose the threshold that minimizes total cost at the target accuracy.

## Outputs

- Simple samples (stay with SLM) and hard samples (sent to LLM) can be written to JSON for downstream processing.

## Best Practices

- Use `temperature=0` for LLM calls to stabilize integration.
- Keep N small (e.g., 5–10) for performance; more passes reduce variance noise but increase latency.
- Consider additional indicators (confidence gap between top‑2 classes, novelty heuristics) if needed.

