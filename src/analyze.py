"""Analysis: divergence of perturbed outputs from canonical outputs, vs the identical-prompt control.

Input: results/scored.parquet (src/score.py). Outputs: results/tables/*.csv, results/stats.json, figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
TAB, FIG = ROOT / "results/tables", ROOT / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
import sys; sys.path.insert(0, str(ROOT / "src"))
from models import MODELS
from perturb import WS, WS_CODE, PUNCT

B = 2000
RNG = np.random.default_rng(42)
REPL = {k: k.replace("rerun-", "") for k, v in MODELS.items() if v.get("replicate")}
MAIN = [k for k in MODELS if k not in REPL]
SIZE = {k: v["params"] for k, v in MODELS.items()}
ORDER = sorted(MAIN, key=lambda k: (MODELS[k]["family"], SIZE[k]))
COND_ORDER = ["control"] + WS + ["tabs"] + PUNCT


def paired(df):
    """One row per (model, item, cond != canonical) with divergence indicators relative to the canonical output."""
    can = df[df.cond == "canonical"].set_index(["model", "item_id"])
    p = df[df.cond != "canonical"].join(can[["text", "label", "code", "correct", "ids_hash"]], on=["model", "item_id"],
                                        rsuffix="_can")
    p = p[p.text_can.notna()].copy()
    # a perturbation that did not alter the prompt string (e.g. ellipsis_char on text without "...") is not a trial
    p = p[(p.family == "control") | p.prompt_changed].copy()
    p["text_diff"] = p.text != p.text_can
    is_cls = p.kind == "cls"
    p["flip"] = np.where(is_cls, p.label != p.label_can, p.correct != p.correct_can)  # label flip / pass-fail flip
    p["code_diff"] = np.where(is_cls, np.nan, (p.code != p.code_can).astype(float))
    te = pd.read_parquet(ROOT / "results/tok_effect.parquet")
    p = p.merge(te[["model", "pid", "tok_changed", "tok_edit", "n_tok_can"]], on=["model", "pid"], how="left")
    p["tok_diff"] = p.tok_changed.astype(float)
    p["tok_edit_frac"] = p.tok_edit / p.n_tok_can
    p["acc_delta"] = p.correct.astype(int) - p.correct_can.astype(int)
    p["small"] = p.model.map(SIZE) < 3.0
    p["params"] = p.model.map(SIZE)
    return p


def boot_mean(x_by_item, b=B):
    """Item-level bootstrap CI for the mean of per-item values (array: items x ...)."""
    x = np.asarray(x_by_item, float)
    idx = RNG.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def rate_table(p, metric, kind):
    """model x cond table of mean(metric) with bootstrap CIs (items resampled)."""
    rows = []
    for (m, c), g in p[p.kind == kind].groupby(["model", "cond"]):
        v = g[metric].astype(float).values
        lo, hi = boot_mean(v, 500)
        rows.append(dict(model=m, cond=c, n=len(v), rate=v.mean(), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def family_contrast(p, metric, kind, groupcols=("model",)):
    """Per group: mean over items of (avg over whitespace conds), (avg over punct conds), control; with CIs and
    bootstrap CI for the punct - whitespace difference and the excess over control."""
    out = []
    q = p[p.kind == kind]
    for key, g in q.groupby(list(groupcols)):
        w = g.pivot_table(index="item_id", columns="family", values=metric, aggfunc="mean").dropna()
        ws, pu, ct = w["whitespace"].values, w["punctuation"].values, w["control"].values
        idx = RNG.integers(0, len(w), (B, len(w)))
        d = pu[idx].mean(1) - ws[idx].mean(1)
        out.append(dict(**dict(zip(groupcols, key if isinstance(key, tuple) else (key,))), n_items=len(w),
                        control=ct.mean(), whitespace=ws.mean(), punctuation=pu.mean(),
                        ws_excess=ws.mean() - ct.mean(), punct_excess=pu.mean() - ct.mean(),
                        punct_minus_ws=d.mean(), pmw_lo=np.percentile(d, 2.5), pmw_hi=np.percentile(d, 97.5)))
    return pd.DataFrame(out)


def size_ratio(p, metric, kind, excess=False):
    """Ratio of mean divergence (small <3B models) / (>=3B models), item-bootstrap CI (same items resampled for all)."""
    q = p[(p.kind == kind) & (p.family != "control")]
    w = q.pivot_table(index="item_id", columns="model", values=metric, aggfunc="mean")
    if excess:
        c = p[(p.kind == kind) & (p.family == "control")].pivot_table(index="item_id", columns="model", values=metric)
        w = w - c[w.columns]
    w = w.loc[:, w.notna().mean() > 0.95].dropna()      # models with incomplete coverage of this task kind are left out
    sm_cols = [m for m in w.columns if SIZE[m] < 3]; lg_cols = [m for m in w.columns if SIZE[m] >= 3]
    a, b = w[sm_cols].values.mean(1), w[lg_cols].values.mean(1)
    idx = RNG.integers(0, len(w), (B, len(w)))
    r = a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-9)
    return dict(metric=metric, kind=kind, excess=excess, small_models=sm_cols, large_models=lg_cols,
                small=a.mean(), large=b.mean(), ratio=a.mean() / max(b.mean(), 1e-9),
                ratio_lo=np.percentile(r, 2.5), ratio_hi=np.percentile(r, 97.5))


def excess_tests(p, metric, kind):
    """Per model x perturbation: exact binomial (McNemar) test of flips(pert) vs flips(control) on the same items."""
    rows = []
    q = p[p.kind == kind]
    for m, g in q.groupby("model"):
        w = g.pivot_table(index="item_id", columns="cond", values=metric, aggfunc="first").astype(float)
        for c in w.columns:
            if c == "control":
                continue
            x, y = w[c], w["control"]
            b10, b01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
            pv = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
            rows.append(dict(model=m, cond=c, pert=x.mean(), control=y.mean(), b10=b10, b01=b01, p=pv))
    t = pd.DataFrame(rows)
    # Holm correction across all tests in the table
    order = np.argsort(t.p.values); k = len(t); adj = np.empty(k); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * t.p.values[i])); adj[i] = run
    t["p_holm"] = adj
    t["sig"] = t.p_holm < 0.05
    return t


def gee(p, metric, kind, extra=""):
    q = p[(p.kind == kind) & (p.family.isin(["whitespace", "punctuation"]))].copy()
    q["log_edit"] = np.log1p(q.tok_edit)
    q["y"] = q[metric].astype(int)
    q["punct"] = (q.family == "punctuation").astype(int)
    q["small"] = q.small.astype(int)
    q["grp"] = q.item_id
    mod = smf.gee("y ~ punct + small + C(task)" + extra, groups="grp", data=q, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable())
    r = mod.fit()
    return pd.DataFrame(dict(coef=r.params, se=r.bse, p=r.pvalues, OR=np.exp(r.params)))


def heat(t, title, path, vmax=None):
    pv = t.pivot(index="model", columns="cond", values="rate")
    pv = pv.reindex(index=[m for m in ORDER if m in pv.index], columns=[c for c in COND_ORDER if c in pv.columns])
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * pv.shape[1], 0.9 + 0.45 * pv.shape[0]))
    im = ax.imshow(100 * pv.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = 100 * pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if vmax and v > 0.6 * vmax else "black")
    ax.set_xticks(range(pv.shape[1]), pv.columns, rotation=40, ha="right")
    ax.set_yticks(range(pv.shape[0]), pv.index)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="% of items")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    df = pd.read_parquet(ROOT / "results/scored.parquet")
    p_all = paired(df[df.model.isin(MODELS)])
    # replication: same checkpoint re-run in a separate engine; compare per-condition rates and item-level agreement
    rep = []
    for r, orig in REPL.items():
        for kind, metric in [("cls", "flip"), ("code", "code_diff"), ("code", "flip")]:
            a = p_all[(p_all.model == orig) & (p_all.kind == kind)].groupby("cond")[metric].mean()
            b = p_all[(p_all.model == r) & (p_all.kind == kind)].groupby("cond")[metric].mean()
            if len(b):
                rep.append(pd.DataFrame(dict(model=orig, kind=kind, metric=metric, original=a, rerun=b)))
        o = df[(df.model == orig)].set_index("pid"); n = df[(df.model == r)].set_index("pid")
        common = o.index.intersection(n.index)
        if len(common):
            same = (o.loc[common, "text"] == n.loc[common, "text"])
            print(f"replication {orig}: identical outputs {same.mean():.4f} over {len(common)} prompts; "
                  f"by kind: {same.groupby(o.loc[common, 'kind']).mean().round(4).to_dict()}")
    if rep:
        rep = pd.concat(rep).reset_index(); rep.to_csv(TAB / "replication.csv", index=False)
        print(rep.round(4).to_string())
        print("replication corr per metric:", rep.groupby(["kind", "metric"]).apply(
            lambda g: np.corrcoef(g.original, g.rerun)[0, 1]).round(3).to_dict())
    df = df[df.model.isin(MAIN)]
    p = p_all[p_all.model.isin(MAIN)].copy()
    p.to_parquet(ROOT / "results/paired.parquet")
    stats = {}
    # coverage
    cov = df.groupby(["model", "task"]).size().unstack()
    cov.to_csv(TAB / "coverage.csv"); print(cov)

    # accuracy per model x task (canonical) and per condition
    acc = df.groupby(["model", "task", "cond"]).correct.mean().unstack("cond")
    acc.to_csv(TAB / "accuracy_by_cond.csv")
    print((100 * df[df.cond == "canonical"].groupby(["model", "task"]).correct.mean().unstack()).round(1))

    # main rate tables
    for kind, metric, title, vmax in [("cls", "flip", "Classification label flip rate (%) vs canonical", 30),
                                      ("cls", "text_diff", "Classification raw-output change (%)", 60),
                                      ("code", "code_diff", "Code: generated code changed (%)", 100),
                                      ("code", "flip", "Code: pass/fail flip rate (%)", 30),
                                      ("cls", "tok_diff", "Prompt token sequence changed (%) - cls", 100),
                                      ("code", "tok_diff", "Prompt token sequence changed (%) - code", 100)]:
        t = rate_table(p, metric, kind)
        t.to_csv(TAB / f"rate_{kind}_{metric}.csv", index=False)
        heat(t, title, FIG / f"heat_{kind}_{metric}.png", vmax)

    # family contrasts (whitespace vs punctuation) per model, per task, overall
    fam = {}
    for kind, metric in [("cls", "flip"), ("cls", "text_diff"), ("code", "code_diff"), ("code", "flip")]:
        a = family_contrast(p, metric, kind, ("model",)); a.to_csv(TAB / f"family_{kind}_{metric}_by_model.csv", index=False)
        b = family_contrast(p, metric, kind, ("task",)); b.to_csv(TAB / f"family_{kind}_{metric}_by_task.csv", index=False)
        q = p.assign(all="all")
        c = family_contrast(q, metric, kind, ("all",))
        fam[f"{kind}_{metric}"] = c.iloc[0].to_dict()
        print(kind, metric); print(a.round(4).to_string())
    stats["family_overall"] = fam

    # size effect
    stats["size"] = [size_ratio(p, m, k, e) for k, m in [("cls", "flip"), ("cls", "text_diff"), ("code", "code_diff"),
                                                       ("code", "flip")] for e in (False, True)]
    # same-family size comparison (Qwen2.5 0.5/1.5 vs 3/7/14)
    qw = p[p.model.str.startswith("qwen2.5")]
    if qw.model.nunique() > 2:
        stats["size_qwen_only"] = [size_ratio(qw, m, k, e) for k, m in [("cls", "flip"), ("code", "code_diff"),
                                                                     ("code", "flip")] for e in (False, True)]
    print(json.dumps(stats["size"], indent=1, default=str))

    # task comparison: code vs classification (overall mean over non-control perts)
    tk = p[p.family.isin(["whitespace", "punctuation"])].groupby(["model", "kind"]).agg(
        flip=("flip", "mean"), text_diff=("text_diff", "mean"), code_diff=("code_diff", "mean")).unstack("kind")
    tk.to_csv(TAB / "task_kind_by_model.csv")
    ctl = p[p.family == "control"].groupby(["model", "kind"]).agg(flip=("flip", "mean"), text_diff=("text_diff", "mean"),
                                                                    code_diff=("code_diff", "mean")).unstack("kind")
    ctl.to_csv(TAB / "control_by_model.csv")
    pert_task = p[p.family.isin(["whitespace", "punctuation"])].groupby(["task"]).agg(
        flip=("flip", "mean"), text_diff=("text_diff", "mean"), code_diff=("code_diff", "mean"), n=("flip", "size"))
    pert_task.to_csv(TAB / "by_task.csv"); print(pert_task)

    # excess-over-control tests
    for kind, metric in [("cls", "flip"), ("code", "flip"), ("code", "code_diff")]:
        t = excess_tests(p, metric, kind); t.to_csv(TAB / f"tests_{kind}_{metric}.csv", index=False)
        stats[f"n_sig_{kind}_{metric}"] = f"{int(t.sig.sum())}/{len(t)}"

    # GEE logistic models
    for kind, metric in [("cls", "flip"), ("code", "code_diff"), ("code", "flip")]:
        try:
            g = gee(p, metric, kind); g.to_csv(TAB / f"gee_{kind}_{metric}.csv"); print(kind, metric); print(g.round(4))
            g = gee(p, metric, kind, " + log_edit"); g.to_csv(TAB / f"gee_{kind}_{metric}_tokedit.csv")
            print(kind, metric, "+log_edit"); print(g.round(4))
        except Exception as e:
            print("GEE failed", kind, metric, e)

    # mediator: does a perturbation that leaves the token sequence unchanged ever change the output?
    med = p[p.tok_diff.notna()].groupby(["kind", "tok_diff"]).agg(text_diff=("text_diff", "mean"), flip=("flip", "mean"),
                                                                   n=("flip", "size"))
    med.to_csv(TAB / "mediator_tokdiff.csv"); print(med)

    # per-condition summary across models: token edit size and divergence
    pc = p.groupby(["kind", "cond"]).agg(family=("family", "first"), tok_changed=("tok_diff", "mean"),
                                         tok_edit=("tok_edit", "mean"), tok_edit_frac=("tok_edit_frac", "mean"),
                                         flip=("flip", "mean"), text_diff=("text_diff", "mean"),
                                         code_diff=("code_diff", "mean"), n=("flip", "size"))
    pc.to_csv(TAB / "per_cond_summary.csv"); print(pc.round(4).to_string())
    # tokenizer/template survival per model x cond
    tc = p.pivot_table(index=["kind", "cond"], columns="model", values="tok_diff", aggfunc="mean")
    tc.to_csv(TAB / "tok_changed_by_model_cond.csv")

    # direction of flips: accuracy change per condition
    accd = p[p.family != "control"].groupby(["kind", "cond"]).acc_delta.mean()
    accd.to_csv(TAB / "acc_delta_by_cond.csv")
    # are flips concentrated on items the model gets wrong (i.e. uncertain items)?
    fc = p[p.family.isin(["whitespace", "punctuation"])].groupby(["kind", "model", "correct_can"]).flip.mean().unstack()
    fc.to_csv(TAB / "flip_by_canonical_correct.csv"); print(fc.round(3))
    # accuracy per family (mean over items) vs canonical, per model
    af = p[p.family.isin(["whitespace", "punctuation"])].groupby(["kind", "model", "family"]).acc_delta.mean().unstack()
    af.to_csv(TAB / "acc_delta_by_family.csv"); print((100 * af).round(2))

    # figure: ws vs punct vs control per model (classification flips and code changes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (kind, metric, ttl) in zip(axes, [("cls", "flip", "Classification: label flip"),
                                              ("code", "code_diff", "Code: generated code changed"),
                                              ("code", "flip", "Code: pass/fail flip")]):
        a = family_contrast(p, metric, kind, ("model",)).set_index("model")
        a = a.reindex([m for m in ORDER if m in a.index])
        x = np.arange(len(a))
        for k, (col, colr) in enumerate([("control", "#9e9e9e"), ("whitespace", "#4c78a8"), ("punctuation", "#f58518")]):
            ax.bar(x + (k - 1) * 0.27, 100 * a[col], 0.27, label=col, color=colr)
        ax.set_xticks(x, [f"{m}\n({SIZE[m]}B)" for m in a.index], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% of items"); ax.set_title(ttl)
    axes[0].legend(); fig.tight_layout(); fig.savefig(FIG / "family_by_model.png", dpi=150); plt.close(fig)

    # figure: divergence vs model size
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (kind, metric, ttl) in zip(axes, [("cls", "flip", "Classification flip (excess over control)"),
                                              ("code", "code_diff", "Code changed (excess over control)")]):
        a = family_contrast(p, metric, kind, ("model",)).set_index("model")
        for fam_, mk in [("ws_excess", "o"), ("punct_excess", "s")]:
            for fmly, colr in [("qwen2.5", "#4c78a8"), ("llama3", "#e45756"), ("qwen3", "#54a24b")]:
                ms = sorted([m for m in a.index if MODELS[m]["family"] == fmly], key=SIZE.get)
                if ms:
                    ax.plot([SIZE[m] for m in ms], [100 * a.loc[m, fam_] for m in ms], marker=mk, color=colr,
                            ls="-" if fam_ == "ws_excess" else "--", label=f"{fmly} {fam_.split('_')[0]}")
        ax.set_xscale("log"); ax.set_xlabel("parameters (B)"); ax.set_ylabel("% items (minus control)")
        ax.axvline(3, color="k", lw=0.5, ls=":"); ax.set_title(ttl)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "size_trend.png", dpi=150); plt.close(fig)

    # figure: per-perturbation divergence pooled over models, with token edit size annotated
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (kind, metric, ttl) in zip(axes, [("cls", "flip", "Classification: label flip"),
                                              ("code", "code_diff", "Code: generated code changed"),
                                              ("code", "flip", "Code: pass/fail flip")]):
        q = p[p.kind == kind]
        conds = [c for c in COND_ORDER if c in set(q.cond)]
        vals, los, his, edits = [], [], [], []
        for c in conds:
            v = q[q.cond == c].groupby("item_id")[metric].mean().values
            lo, hi = boot_mean(v, 500); vals.append(v.mean()); los.append(lo); his.append(hi)
            edits.append(q[q.cond == c].tok_edit.mean())
        cols = ["#9e9e9e" if c == "control" else "#4c78a8" if c in WS + ["tabs"] else "#f58518" for c in conds]
        y = 100 * np.array(vals)
        ax.bar(range(len(conds)), y, color=cols,
               yerr=[y - 100 * np.array(los), 100 * np.array(his) - y], capsize=2)
        for i, e in enumerate(edits):
            ax.text(i, y[i] + 0.5, f"Δtok\n{e:.0f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(conds)), conds, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% (all models pooled)"); ax.set_title(ttl)
    fig.tight_layout(); fig.savefig(FIG / "per_condition.png", dpi=150); plt.close(fig)

    json.dump(stats, open(ROOT / "results/stats.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
