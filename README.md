# Invisible Formatting Effects

Do meaning-preserving, (near-)invisible formatting changes such as trailing spaces, CRLF line endings, double spaces,
no-break spaces, curly quotes, en/em dashes and ellipsis characters change what an LLM outputs? We ran 11 open models
(0.5B–14.8B; Qwen2.5, Qwen3 and Llama-3.x) on 3 classification tasks (SST-2, AG News, BoolQ; 1,200 items) and 2
code-generation tasks (HumanEval, MBPP; 464 problems). Each item ran under a canonical prompt, a byte-identical re-run
(the noise floor) and 10 formatting perturbations, all with greedy decoding: about 220k generations in total.

**Full write-up: [REPORT.md](REPORT.md)**

## Key findings
- One invisible formatting change flips **4.9%** of classification labels on average (95% CI 4.6–5.1%), against
  **0.02%** for a byte-identical re-run. The edit is significant in 80 of 110 model × perturbation cells (Holm-corrected).
- **Whitespace hurts more than punctuation**, the reverse of the hypothesis. Whitespace flips 7.1% of labels and
  punctuation 2.2%, and whitespace is worse in every model. Part of the reason is that double spaces and NBSPs re-tokenise
  every word, but whitespace stays worse at a matched number of changed tokens.
- Models under 3B parameters flip **2.2×** as often as ≥3B models on classification (CI 2.0–2.4; 2.9× within
  Qwen2.5). On code, pass/fail flips show no size effect.
- Code text changes for **36%** of problems (identical re-run: 6%), but pass/fail changes for only **5.1%**, about the
  same as classification label flips.
- Net accuracy barely moves (mostly within ±1 pp): flips go both ways and land mostly on items the model gets wrong.
  Placebo check: Llama chat templates strip trailing newlines, so the token sequence is unchanged and the flip rate is exactly 0.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync      # vllm 0.32, pandas, statsmodels, ...
export NUMPY_MADVISE_HUGEPAGE=0 HF_HOME=/data/transformers
python src/build_prompts.py                                         # data/items.jsonl, data/prompts.jsonl (19,968 prompts)
CUDA_VISIBLE_DEVICES=0 python src/worker.py logs/queue.txt 16       # local models (one key per line in the queue file)
python src/gen_server.py 48                                         # Qwen3-8B via an OpenAI-compatible vLLM server on :8011
python src/tok_effect.py                                            # does each perturbation change the token sequence?
python src/score.py                                                 # label parsing + sandboxed unit-test execution
python src/analyze.py                                               # tables (results/tables), figures (figures/), stats.json
```

## Layout
```
planning.md             research plan (motivation, hypotheses, analysis plan)
src/perturb.py          the 10 perturbations (+ code-aware application: prose-only edits touch docstrings/comments only)
src/build_prompts.py    datasets -> prompts (canonical, control, perturbations)
src/worker.py           in-process vLLM worker for shared GPUs (greedy decoding, flock per model)
src/gen_server.py       generation through the local Qwen3-8B server
src/tok_effect.py       chat template + tokenizer: token edit size per perturbation
src/score.py            label parsing, code extraction, test execution (cached by program hash)
src/analyze.py          divergence metrics, bootstrap CIs, McNemar tests, GEE models, figures
results/gens/*.jsonl    raw generations, one file per model
results/tables/         all CSV tables;  results/stats.json  summary statistics
figures/                heatmaps, per-condition bars, size trends
```
