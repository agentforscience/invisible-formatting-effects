"""For every model and prompt: render the chat template, tokenize, and record (n_tokens, hash, token edit size vs the
canonical prompt). Answers "does the perturbation survive the chat template + tokenizer, and how many tokens change?"
Output: results/tok_effect.parquet
"""
import difflib, hashlib, json, os, sys
os.environ.setdefault("HF_HOME", "/data/transformers"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS


def main():
    prompts = [json.loads(l) for l in open(ROOT / "data/prompts.jsonl")]
    rows = []
    seen_tok = {}
    for key, cfg in MODELS.items():
        tok = AutoTokenizer.from_pretrained(cfg["hf"])
        kw = {"enable_thinking": False} if cfg["family"] == "qwen3" else {}
        ids = {}
        for p in prompts:
            s = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                        add_generation_prompt=True, **kw)
            ids[p["pid"]] = tok(s, add_special_tokens=False)["input_ids"]
        for p in prompts:
            a, b = ids[p["pid"]], ids[f"{p['item_id']}|canonical"]
            if a == b:
                ed = 0
            else:   # number of token positions inserted/deleted/replaced
                sm = difflib.SequenceMatcher(None, b, a, autojunk=False)
                ed = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
            rows.append(dict(model=key, pid=p["pid"], n_tok=len(a), n_tok_can=len(b), tok_changed=a != b, tok_edit=ed))
        print(key, "done", flush=True)
    pd.DataFrame(rows).to_parquet(ROOT / "results/tok_effect.parquet")


if __name__ == "__main__":
    main()
