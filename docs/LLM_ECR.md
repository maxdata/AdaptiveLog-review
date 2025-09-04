# LLM Integration and Error‑prone Case Retrieval (ECR)

This document describes how AdaptiveLog integrates an LLM (e.g., ChatGPT) and uses ECR to improve reasoning on hard cases.

## Purpose

- Focus expensive LLM reasoning only on cases where the SLM is uncertain.
- Provide the LLM with concise, relevant prior “error‑prone” cases to steer reasoning and avoid common mistakes.

## Prompting with ECR

- Templates live under `prompt_template/ecr/*.txt`.
- For LDSM, see `prompt_template/ecr/ldsm.txt` — the template instructs the model to reason and then output a definitive label.
- The pipeline injects a small set of similar, error‑prone examples into `{error_cases}` before the test input.

## Running the LLM

Example (LDSM hard cases):

```
python query_ChatGPT.py \
  --data ./datasets/adaptive_ldsm/ldsm_hwswitch_test.json \
  --model gpt-3.5-turbo-16k-0613
```

Implementation entry: `query_ChatGPT.py`.

Suggested settings:
- Temperature: `0` for deterministic classification behavior.
- Max tokens: small (e.g., `32–128`) since output is a short label.
- Strict output format: ask the model to output `Label: 0|1` (or a structured score list).

## Practical Guidance

- Case selection: choose top‑k most similar error‑prone cases (k≈2–3) to limit tokens.
- Trimming: include only the essential log text and the label/outcome from each case; omit long parameters.
- Reasoning then label: prompt asks for brief “Reason:” followed by “Label: …” to ensure interpretability.

## Data Privacy

- Keep PII out of prompts; prefer placeholders (`[IPADDR]`, `[INTEGER]`).
- Consider on‑prem LLMs for strict environments.

## Known Rough Edges to Review

- `query_ChatGPT.py` header uses `"Bearer sb-$OPENAI_API_KEY"`; standard practice is using `OPENAI_API_KEY` directly.
- There is a minor CLI argument syntax to double‑check (`--url` argument default needs a comma separator). Adjust locally if needed.

