# Cost, Latency, and Accuracy Trade‑offs

This document summarizes when to use SLM‑only, LLM‑only, or the hybrid approach, with a lightweight cost model and an experiment plan.

## Options

- SLM‑only: Lowest cost/latency; good on common patterns; may miss edge cases.
- LLM‑only: Highest accuracy on complex cases but high cost/latency per log.
- Hybrid (AdaptiveLog): SLM handles most; LLM only for uncertain cases → best balance.

## Cost Model (Illustrative)

- Let p = fraction of logs routed to LLM after SLM gating (typ. 0.2–0.4).
- Let C_llm = cost per token × prompt tokens.
- Minifier reduces tokens by r (e.g., 70–95%).

Total cost per 1k logs ≈ 1000 × p × C_llm × (1 − r).

Compression alone still pays 100% calls; gating multiplies savings by reducing both calls and tokens.

## Latency

- SLM inference: sub‑second locally, high throughput.
- LLM call: seconds per request, rate‑limited; gating reduces queueing.

## Accuracy

- SLM‑only baseline is strong on routine logs.
- LLM improves hard/novel patterns; ECR further boosts reasoning.
- Hybrid typically adds 5–15% over SLM‑only at 60–80% fewer LLM calls versus LLM‑only.

## When to Favor Each

- SLM‑only: tight latency/throughput, labeled data available, cost‑sensitive.
- LLM‑only: low volume, need rationales for every case, no labels, or quick bootstrap.
- Hybrid: most production settings with moderate volume and a labeled seed set.

## Experiment Plan

1. Prepare train/dev/test splits per task.
2. Train SLM; evaluate baseline metrics.
3. Implement MC‑Dropout routing; sweep uncertainty thresholds on dev set.
4. Minify prompts; pick top‑k ECR cases.
5. Compare:
   - SLM‑only
   - LLM‑only (minified and non‑minified)
   - Hybrid (with/without minification)
6. Report: Accuracy/F1 (or ranking metrics), p (routed fraction), cost per 1k logs, and p95 latency.

## Operational Notes

- Monitor drift and retrain SLM periodically.
- Set clear budgets for p (LLM routing rate) and max prompt tokens.
- Keep a feedback loop to curate new error‑prone cases for ECR.

