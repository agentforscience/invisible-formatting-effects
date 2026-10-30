# Planning: Invisible Formatting Effects

## Motivation & Novelty Assessment

### Why This Research Matters
Text reaching an LLM in production is routinely altered in ways a human cannot see: Windows CRLF line endings,
trailing spaces from copy-paste, word-processor "smart" quotes, en/em dashes, non-breaking spaces. If such
meaning-preserving changes flip a classifier's label or change generated code, then evaluation numbers are
partly noise, A/B tests of prompts are confounded, and deployed pipelines behave inconsistently across
operating systems and editors. Practitioners (eval builders, people deploying small on-device models) need
to know how big this effect is relative to the irreducible noise floor of inference.

### Gap in Existing Work
(The resource-finder stage was skipped for this workspace — no papers/ datasets/ code/ were provided — so this
is based on background knowledge.) Prior work on prompt sensitivity (Sclar et al. 2023 "Quantifying LMs'
sensitivity to spurious features in prompt design / FormatSpread"; Lu et al. 2022 on example order;
Zhuo et al. 2024 ProSA; Mizrahi et al. 2024 multi-prompt evaluation) mostly perturbs *visible* template
structure (separators, casing, field names, spacing between fields). Character-level robustness work
(TextFlint, adversarial typos, homoglyph attacks) studies visible typos or adversarial search. What is
missing: a controlled, typed comparison of *invisible / near-invisible* formatting changes (whitespace vs.
punctuation-glyph variants) across tasks (classification vs. code), across model sizes in one family,
**measured against a nondeterminism noise floor** (byte-identical re-runs) and with a check of whether the
perturbation even survives the chat template/tokenizer (e.g., Llama templates trim trailing whitespace).

### Our Novel Contribution
1. A taxonomy of 10 invisible perturbations (5 whitespace, 5 punctuation) applied to the full user message.
2. Flip / divergence rates for 7–8 open models from 0.5B to 8B (two families, Qwen2.5 and Llama-3.x, plus Qwen3-8B)
   on 3 classification tasks and 2 code-generation tasks, with the byte-identical re-run as control.
3. A mediator analysis: does the perturbation change the token sequence the model sees, and how many tokens?

### Experiment Justification
- Exp 1 (classification flips: SST-2, AG News, BoolQ): directly tests "formatting flips labels 5–15%" and
  whitespace (3–8%) vs punctuation (8–20%).
- Exp 2 (code generation: HumanEval, MBPP): tests "code is most affected (15–25%)"; we measure both raw output
  change and functional (pass/fail) change because raw-text change is inflated by output length.
- Exp 3 (model-size comparison): Qwen2.5 0.5/1.5/3/7B + Llama-3.2 1/3B + Llama-3.1-8B + Qwen3-8B tests
  "<3B models are 2–3× more sensitive".
- Control (identical prompt re-run in a different batch) — without it, vLLM batch nondeterminism would be
  mis-attributed to formatting.

## Research Question
Do meaning-preserving, (near-)invisible formatting changes — whitespace and punctuation-glyph variants — change
LLM outputs, by how much, for which perturbation types, tasks and model sizes?

## Hypothesis Decomposition
- H1: Any formatting perturbation flips classification labels in 5–15% of items (averaged over perturbations/models).
- H2: Whitespace variants → 3–8% divergence; punctuation variants → 8–20%; punctuation > whitespace.
- H3: Models <3B have 2–3× higher flip rates than ≥3B models.
- H4: Code generation output changes 15–25% and more than classification.

## Proposed Methodology
### Approach
Greedy decoding (temperature 0) with local open-weight models in vLLM (no API keys available on this host),
plus the host's Qwen3-8B vLLM server. Each item is run under the canonical prompt, an identical-prompt
re-run (control), and 10 perturbations. Divergence = output differs from the canonical output.

### Perturbations (applied to the whole user message: instruction + input)
Whitespace: `trailing_space` (space at end of every line), `crlf` (\n→\r\n), `double_space` (every single
space between words → two), `trailing_newlines` (append "\n\n"), `nbsp` (spaces → U+00A0).
For code prompts additionally `tabs` (4-space indentation → tab).
Punctuation: `curly_quotes` (" ' → “ ” ‘ ’), `em_dash` (hyphen/“--” between words → —, and ranges → –),
`ellipsis` ("..." → …, also appended-free), `fullwidth_punct`? (rejected: visible), `no_final_period`?
(rejected: semantic). Final punctuation set: curly_quotes, curly_apostrophes, dashes (- / -- → – / —),
ellipsis_and_primes, unicode_punct_all (all at once). Instructions are written to contain quotes, an apostrophe,
a dash and an ellipsis-free colon so every prompt is affected by each punctuation variant.
For code we only change punctuation inside docstrings/comments so the code stays valid.

### Baselines / controls
Canonical prompt; identical re-run (noise floor); accuracy of each condition vs gold.

### Evaluation Metrics
- Classification: flip rate = P(label_pert ≠ label_canonical); accuracy change.
- Code: text-change rate (normalized code differs), functional flip rate (pass/fail differs), pass@1 change.
- Effective-perturbation rate: fraction of prompts whose token ids actually differ after chat template.

### Statistical Analysis Plan
Item-level paired data. Wilson / item-cluster bootstrap 95% CIs (resample items). Perturbation vs control:
McNemar-style exact test on flips (flip in pert vs flip in control). Whitespace vs punctuation and small vs
large: cluster bootstrap of ratio/difference, plus mixed logistic regression (statsmodels GLM with
item-clustered SEs) of flip ~ type + size + task. α=0.05, Holm correction across perturbations.

## Expected Outcomes
Support: punct > ws, small/large ratio 2–3, code > classification with rates in the stated ranges.
Refute: rates near the control floor, ws ≥ punct, ratio ≈1, or code ≤ classification.

## Timeline
Setup+data 30 min; prompt building 30 min; generation 2–3 h (shared GPUs); scoring/code execution 30 min;
analysis+report 1 h.

## Potential Challenges
Shared GPUs with ~4–6 GB free → AWQ checkpoints, in-process memory holder, small models first; Qwen3-8B via
server. vLLM batch nondeterminism → control condition. Heavy CPU load (load avg ~230) → slow code execution;
use per-test timeouts and a process pool.

## Success Criteria
All conditions run for ≥5 models on ≥1,000 classification items and ≥400 code problems; CIs reported;
each sub-hypothesis gets a clear supported/refuted verdict.

## Deviations from the plan (recorded after execution)
- No API keys on this host, so all models are open weights. Final set: 11 models, Qwen2.5 0.5–14B AWQ, Llama-3.2-1B/3B,
  Llama-3.1-8B-AWQ, Qwen3-0.6B/4B-AWQ, and Qwen3-8B via the local vLLM server. Qwen3-0.6B/4B were added mid-run to get a
  third same-family size series.
- Final perturbations: whitespace = trailing_space, crlf, double_space, trailing_newlines, nbsp (classification) or
  tabs (code); punctuation = curly_quotes, curly_apostrophe, unicode_dash, ellipsis_char, punct_all.
  `ellipsis_char` applies to few items. Unchanged prompts are excluded.
- Added a replication run (Qwen2.5-0.5B in a separate engine): all outputs were byte-identical.
- Mixed models: GEE logistic (exchangeable, clustered by item) instead of a mixed-effects GLM.
