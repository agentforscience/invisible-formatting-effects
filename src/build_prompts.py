"""Build all prompts: 3 classification tasks + 2 code tasks, each item x (canonical, control, 5 whitespace, 5 punct).

Writes data/items.jsonl (one row per item, with gold info) and data/prompts.jsonl (one row per prompt).
"""
import glob
import json
import random
from pathlib import Path

import pandas as pd

from perturb import FUNCS, WS, WS_CODE, PUNCT, FAMILY, apply_to_code

ROOT = Path(__file__).resolve().parent.parent
HUB = "/data/transformers/hub/"
SEED, N_CLS, N_MBPP = 42, 400, 300

TEMPLATES = {
    "sst2": ('Classify the sentiment of the movie review below as "positive" or "negative".\n'
             "Don't explain - answer with one word only.\n\nReview: {text}\nSentiment:"),
    "agnews": ('Classify the news article below into one of four topics: "World", "Sports", "Business" or "Sci/Tech".\n'
               "Don't explain - answer with the topic name only.\n\nArticle: {text}\nTopic:"),
    "boolq": ('Read the passage and answer the question with "yes" or "no".\n'
              "Don't explain - answer with one word only.\n\nPassage: {passage}\nQuestion: {question}?\nAnswer:"),
}
CODE_INSTR = {
    "humaneval": ("Complete the following Python function. Don't change its signature - return the complete function "
                  '(with any imports it needs) in a single ```python code block, without "explanations" or tests.'),
    "mbpp": ("Write a Python function for the task below. Don't write anything else - return the code in a single "
             '```python code block, without "explanations".'),
}
AG = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}


def pq(pattern):
    return pd.read_parquet(glob.glob(HUB + pattern)[0])


def load_items():
    rng = random.Random(SEED)
    items = []
    sst = pq("datasets--stanfordnlp--sst2/snapshots/*/data/validation*")
    for i in sorted(rng.sample(range(len(sst)), N_CLS)):
        r = sst.iloc[i]
        items.append(dict(item_id=f"sst2/{r['idx']}", task="sst2", kind="cls", gold=["negative", "positive"][r["label"]],
                          fields=dict(text=r["sentence"].strip())))
    ag = pq("datasets--fancyzhx--ag_news/snapshots/*/data/test*")
    for i in sorted(rng.sample(range(len(ag)), N_CLS)):
        r = ag.iloc[i]
        text = r["text"].replace("#39;", "'").replace("\\", " ").strip()   # undo AG News escaping artefacts
        items.append(dict(item_id=f"agnews/{i}", task="agnews", kind="cls", gold=AG[r["label"]], fields=dict(text=text)))
    bq = pq("datasets--google--boolq/snapshots/*/data/validation*")
    bq = bq[bq["passage"].str.split().str.len() <= 180].reset_index()   # keep prompts short
    for i in sorted(rng.sample(range(len(bq)), N_CLS)):
        r = bq.iloc[i]
        items.append(dict(item_id=f"boolq/{r['index']}", task="boolq", kind="cls", gold="yes" if r["answer"] else "no",
                          fields=dict(passage=r["passage"].strip(), question=r["question"].strip())))
    he = pq("datasets--openai--openai_humaneval/snapshots/*/openai_humaneval/test*")
    for _, r in he.iterrows():
        items.append(dict(item_id=r["task_id"], task="humaneval", kind="code",
                          fields=dict(code=r["prompt"]), test=r["test"], entry_point=r["entry_point"]))
    mb = pq("datasets--google-research-datasets--mbpp/snapshots/*/full/test*")
    for i in sorted(rng.sample(range(len(mb)), N_MBPP)):
        r = mb.iloc[i]
        tests = "\n".join(r["test_list"])
        items.append(dict(item_id=f"mbpp/{r['task_id']}", task="mbpp", kind="code",
                          fields=dict(text=r["text"].strip(), code=tests + "\n"),
                          test=tests, setup=r["test_setup_code"], entry_point=None))
    return items


def render(item, cond):
    """User message for `item` under condition `cond` (canonical/control = unperturbed)."""
    t, f = item["task"], item["fields"]
    p = None if cond in ("canonical", "control") else cond
    if item["kind"] == "cls":
        msg = TEMPLATES[t].format(**f)
        return FUNCS[p](msg) if p else msg
    # code: prose part perturbed fully, code part via apply_to_code (prose-only perturbations touch docstrings only)
    prose = CODE_INSTR[t] + ("\n\nTask: " + f["text"] + "\nYour code should pass these tests:" if t == "mbpp" else "")
    code = f["code"]
    if p:
        if p in ("crlf", "trailing_space"):        # whole-message layout perturbations
            return FUNCS[p](prose + "\n\n```python\n" + code + "```")
        if p == "trailing_newlines":
            return prose + "\n\n```python\n" + code + "```" + "\n\n"
        prose = FUNCS[p](prose) if p != "tabs" else prose
        code = apply_to_code(p, code)
    return prose + "\n\n```python\n" + code + "```"


def main():
    items = load_items()
    (ROOT / "data").mkdir(exist_ok=True)
    with open(ROOT / "data/items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n = 0
    with open(ROOT / "data/prompts.jsonl", "w") as f:
        for it in items:
            conds = ["canonical", "control"] + (WS if it["kind"] == "cls" else WS_CODE) + PUNCT
            base = render(it, "canonical")
            for c in conds:
                msg = render(it, c)
                f.write(json.dumps(dict(pid=f"{it['item_id']}|{c}", item_id=it["item_id"], task=it["task"],
                                        kind=it["kind"], cond=c, family=FAMILY[c], changed=msg != base,
                                        prompt=msg), ensure_ascii=False) + "\n")
                n += 1
    print(len(items), "items", n, "prompts")


if __name__ == "__main__":
    main()
