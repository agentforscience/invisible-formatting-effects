"""Generate with the host's long-lived vLLM server (Qwen3-8B, OpenAI-compatible, :8011), greedy, thinking off.

Usage: python src/gen_server.py [N_THREADS]     -> results/gens/qwen3-8b.jsonl (resumable, append-only)
Uses only the standard library.
"""
import json, os, random, sys, threading, time, urllib.request, fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8011/v1/chat/completions"
KEY, MODEL = "qwen3-8b", "Qwen3-8B"
OUT = ROOT / "results/gens" / f"{KEY}.jsonl"
SEED = 42


def call(p):
    body = dict(model=MODEL, messages=[{"role": "user", "content": p["prompt"]}], temperature=0.0, seed=SEED,
                max_tokens=12 if p["kind"] == "cls" else 512, chat_template_kwargs={"enable_thinking": False})
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=600))
            c = r["choices"][0]
            return dict(pid=p["pid"], text=c["message"]["content"], finish=c["finish_reason"],
                        n_prompt=r["usage"]["prompt_tokens"], n_out=r["usage"]["completion_tokens"], ids_hash=None)
        except Exception as e:
            time.sleep(2 ** attempt)
            err = e
    print("FAILED", p["pid"], err, flush=True)
    return None


def main():
    n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lock = open(OUT.with_suffix(".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prompts = [json.loads(l) for l in open(ROOT / "data/prompts.jsonl")]
    done = {json.loads(l)["pid"] for l in open(OUT)} if OUT.exists() else set()
    todo = [p for p in prompts if p["pid"] not in done]
    random.Random(SEED).shuffle(todo)
    todo.sort(key=lambda p: p["kind"] != "cls")       # short classification prompts first
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    wl, t0, n = threading.Lock(), time.time(), 0
    with open(OUT, "a") as f, ThreadPoolExecutor(n_threads) as ex:
        for res in ex.map(call, todo):
            if res is None:
                continue
            with wl:
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if n % 500 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
