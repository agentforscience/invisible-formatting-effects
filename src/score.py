"""Score generations: parse classification labels; extract + execute generated code against unit tests.

Writes results/scored.parquet with one row per (model, pid). Code execution results are cached by program hash in
results/exec_cache.jsonl (identical programs across conditions/models are executed once).
"""
import hashlib, json, os, re, resource, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GENS = ROOT / "results/gens"
CACHE = ROOT / "results/exec_cache.jsonl"
TIMEOUT = 20

LABELS = {"sst2": ["positive", "negative"], "boolq": ["yes", "no"]}


def parse_label(task, text):
    t = text.strip().lower().replace("’", "'")
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    if task == "agnews":
        cands = {"world": "world", "sports": "sports", "sport": "sports", "business": "business",
                 "sci/tech": "sci/tech", "sci-tech": "sci/tech", "science": "sci/tech", "technology": "sci/tech",
                 "tech": "sci/tech"}
    else:
        cands = {l: l for l in LABELS[task]}
    best = None
    for k, v in cands.items():
        m = re.search(r"\b" + re.escape(k) + r"\b", t)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), v)
    return best[1] if best else "other"


def extract_code(text):
    """First ```python block that defines something (else first block, else the raw text)."""
    text = text.replace("\r\n", "\n")
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)", text, flags=re.S)
    for b in blocks:
        if "def " in b:
            return b
    return blocks[0] if blocks else text


def norm_code(code):
    return "\n".join(l.rstrip() for l in code.replace("\r\n", "\n").strip().split("\n"))


def program(item, code):
    if item["task"] == "humaneval":
        imports = "\n".join(l for l in item["fields"]["code"].split("\n") if re.match(r"^(import|from)\s", l))
        return f"{imports}\n{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    return f"{item.get('setup') or ''}\n{code}\n\n{item['test']}\n"


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    os.setsid()


def execute(prog):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "prog.py"
        p.write_text(prog)
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=d, capture_output=True, timeout=TIMEOUT,
                               preexec_fn=_limits, env={"PATH": "/usr/bin:/bin", "NUMPY_MADVISE_HUGEPAGE": "0"})
            return r.returncode == 0, (r.stderr.decode(errors="replace").strip().split("\n") or [""])[-1][:200]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def main():
    items = {it["item_id"]: it for it in map(json.loads, open(ROOT / "data/items.jsonl"))}
    prompts = {p["pid"]: p for p in map(json.loads, open(ROOT / "data/prompts.jsonl"))}
    rows = []
    for f in sorted(GENS.glob("*.jsonl")):
        model = f.stem
        for l in open(f):
            g = json.loads(l)
            p = prompts[g["pid"]]
            rows.append(dict(model=model, pid=g["pid"], item_id=p["item_id"], task=p["task"], kind=p["kind"],
                             cond=p["cond"], family=p["family"], prompt_changed=p["changed"], text=g["text"],
                             finish=g["finish"], n_prompt=g["n_prompt"], n_out=g["n_out"], ids_hash=g["ids_hash"]))
    df = pd.DataFrame(rows).drop_duplicates(["model", "pid"], keep="last")
    cls = df.kind == "cls"
    df.loc[cls, "label"] = [parse_label(t, x) for t, x in zip(df.task[cls], df.text[cls])]
    df.loc[cls, "correct"] = [l == items[i]["gold"] for l, i in zip(df.label[cls], df.item_id[cls])]
    code = ~cls
    df.loc[code, "code"] = [norm_code(extract_code(x)) for x in df.text[code]]
    progs = {}
    for i, c in zip(df.item_id[code], df.code[code]):
        pr = program(items[i], c)
        progs[hashlib.sha1(pr.encode()).hexdigest()] = pr
    cache = {}
    if CACHE.exists():
        for l in open(CACHE):
            r = json.loads(l); cache[r["h"]] = (r["ok"], r["err"])
    todo = [h for h in progs if h not in cache]
    print(f"{len(progs)} unique programs, {len(todo)} to execute", flush=True)
    with open(CACHE, "a") as fc, ThreadPoolExecutor(int(os.environ.get("EXEC_THREADS", 24))) as ex:
        for k, (h, res) in enumerate(zip(todo, ex.map(lambda h: execute(progs[h]), todo))):
            cache[h] = res
            fc.write(json.dumps(dict(h=h, ok=res[0], err=res[1])) + "\n")
            if k % 500 == 0:
                fc.flush(); print(k, flush=True)
    hs = [hashlib.sha1(program(items[i], c).encode()).hexdigest() for i, c in zip(df.item_id[code], df.code[code])]
    df.loc[code, "correct"] = [cache[h][0] for h in hs]
    df.loc[code, "exec_err"] = [cache[h][1] for h in hs]
    df["correct"] = df["correct"].astype(bool)
    df.to_parquet(ROOT / "results/scored.parquet")
    print(df.groupby(["model", "task"]).correct.mean().unstack().round(3))


if __name__ == "__main__":
    main()
