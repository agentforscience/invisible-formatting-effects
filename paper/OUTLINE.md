# Outline: Invisible formatting effects

1. Abstract: problem (invisible edits in production text) -> gap (no controlled ws vs punct comparison w/ noise floor) -> 11 models, 5 tasks, 10 edits, 239,616 generations -> 4.9% flips vs 0.02% control; ws 7.1% > punct 2.2%; <3B 2.2x; code text 36% but pass/fail 5.1% -> normalise whitespace.
2. Intro: hook (CRLF, NBSP, smart quotes); importance (evals, A/B); gap (visible-template sensitivity work, Sclar/Mizrahi/ProSA); approach (Fig 1 per-condition); preview numbers; 4 contributions.
3. Related work: prompt-format sensitivity; character-level robustness/typos; tokenization; nondeterminism of inference.
4. Method: problem formulation (divergence wrt canonical), perturbations table, tasks, models, metrics, controls, stats.
5. Results: overall table vs noise floor; per perturbation (Fig per_condition + table); per model (heatmap + table); size (Fig size_trend + table); task; mediators (token survival, edit size, uncertainty, accuracy).
6. Discussion: hypotheses verdicts, mechanism, implications, limitations.
7. Conclusion.
