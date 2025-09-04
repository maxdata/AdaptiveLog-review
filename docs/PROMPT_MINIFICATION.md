# Prompt and Token Minification Strategies

This document outlines safe token‑reduction techniques to cut LLM cost without harming accuracy for AdaptiveLog tasks.

## Why Minify

- LLM cost scales with tokens. Even after routing, reducing prompt size on the hard subset multiplies savings.

## Principles

- Preserve discriminative structure (module, severity slot, message ID, summary).
- Replace values with typed placeholders (already common in datasets): `[IPADDR]`, `[INTEGER]`, `[STRING]`.
- Keep only top‑k ECR examples and trim each to essentials.

## Minifier Spec (Per Task)

### LDSM (log ↔ description)
- Keep: full log template (with placeholders), 1–2 sentence description (trimmed), and label format.
- Drop: long parameter lists inside parentheses; keep only names/types or a count.

### MC (module classification)
- Keep: masked log line `"[MASK]/<SEV>/<ID>:<summary>"`.
- Preserve module‑indicative keywords (e.g., OSPF, LSA, area) if present in text.
- Provide module list or constrain the output to one label.

### LP (severity)
- Keep: module name, masked severity slot `[MASK]`, summary, short description.
- Add brief definition: Error vs Info (one line) to align criteria.

### LPCR (5 causes)
- Keep: log summary + 5 cause statements.
- Trim each cause to a single clear sentence; avoid extraneous details.

## ECR Trimming

- Select top‑k (k≈2–3) similar error‑prone cases.
- For each case include: short log, short rationale phrase, final label.
- Exclude long parameter blocks and narrative.

## Prompting Tips

- Fixed output schema: e.g., `Label: 0|1` or `Output: A: xx, B: xx, C: xx, D: xx, E: xx`.
- `temperature=0`, `max_tokens` small (32–128), no streaming.
- Be explicit about “reason first, then label” in one short sentence.

## Example Skeleton (LDSM)

```
You are a network logs analyst. Decide if the description matches the log.
Error‑prone cases:
{top_k_compact_cases}

Test Input:
Log: {LOG_MODULE}/{MASK}/{MSG_ID}:{SUMMARY}
Description: {DESC_1_2_SENTENCES}

Output format:
Reason: <short>
Label: 0 or 1
```

## Validation

- A/B test minifier versions on dev set; ensure no significant accuracy drop versus unminified prompts.
- Inspect LLM rationales for signs of missing context.

