"""Long-lived in-process GPU worker for a shared GPU.

Holds GPU memory as 512 MB placeholder tensors (grown opportunistically while other tenants free memory), and runs
models from a queue file one after another IN-PROCESS. Placeholders are handed to vLLM through PyTorch's caching
allocator (empty_cache disabled), so memory never goes back to the driver between models -> no startup races.

Usage: CUDA_VISIBLE_DEVICES=<g> python src/worker.py QUEUE_FILE MAX_GB
Queue file: one model key per line (re-read after each model). Each model's output file is protected by flock, so
several workers (on different GPUs) can share one queue.
"""
import os, sys, json, time, gc, fcntl, random, hashlib
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HOME", "/data/transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS

SEED = 42
CHUNK = 512 << 20
LOGS, GENS = ROOT / "logs", ROOT / "results" / "gens"
GENS.mkdir(parents=True, exist_ok=True)
PLACE = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


if os.environ.get("TAKEOVER_PID"):       # hand-over: wait (imports done) until the previous worker exits
    open(LOGS / "takeover_ready", "w").write(str(os.getpid()))
    while os.path.exists(f"/proc/{os.environ['TAKEOVER_PID']}"):
        time.sleep(0.002)
while True:                              # CUDA context needs ~0.5 GB; retry while the GPU is full
    try:
        torch.zeros(1, device="cuda"); break
    except Exception:
        time.sleep(0.5)


def held():
    return len(PLACE) * CHUNK / 2**30


def grab(gb, timeout):
    """Try to hold `gb` GB in total, polling for up to `timeout` s."""
    t = time.time()
    while held() < gb and time.time() - t < timeout:
        try:
            PLACE.append(torch.empty(CHUNK, dtype=torch.uint8, device="cuda"))
        except Exception:
            time.sleep(0.02)
    return held() >= gb


if os.environ.get("TAKEOVER_PID"):
    grab(float(os.environ.get("TAKEOVER_GB", "4.5")), timeout=20)
    log(f"took over {held():.1f} GB")

# ---- hand-off to vLLM without returning memory to the driver
torch.cuda.empty_cache = lambda: None
try:
    torch.accelerator.empty_cache = lambda: None
except Exception:
    pass
import vllm.v1.worker.gpu_worker as gw
RELEASE = {"gb": 0.0}


class _Snap(gw.MemorySnapshot):
    def measure(self):
        while RELEASE["gb"] > 0 and PLACE:
            PLACE.pop(); RELEASE["gb"] -= CHUNK / 2**30
        RELEASE["gb"] = 0.0
        super().measure()
        self.free_memory += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


gw.MemorySnapshot = _Snap
from vllm import LLM, SamplingParams


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def run_model(key):
    cfg = MODELS[key]
    out = GENS / f"{key}.jsonl"
    prompts = read_jsonl(ROOT / "data/prompts.jsonl")
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    todo = [p for p in prompts if p["pid"] not in done]
    random.Random(SEED).shuffle(todo)            # canonical and control copies land in different batch positions
    log(f"{key}: {len(done)} done, {len(todo)} to do, holding {held():.1f} GB")
    if not todo:
        return
    total = held()
    kv = cfg["kv"] + max(0.0, total - cfg["need"] - 0.5)
    RELEASE["gb"] = total
    llm = LLM(model=cfg["hf"], dtype="auto", max_model_len=2048, gpu_memory_utilization=min(0.95, (total - 0.2) / 47.9),
              kv_cache_memory_bytes=int(kv * 2**30), enforce_eager=True, max_num_seqs=128,
              max_num_batched_tokens=2048, seed=SEED, trust_remote_code=False, enable_prefix_caching=False)
    sps = {"cls": SamplingParams(temperature=0.0, max_tokens=12, seed=SEED),
           "code": SamplingParams(temperature=0.0, max_tokens=512, seed=SEED)}
    t0 = time.time()
    with open(out, "a") as f:
        for kind in ("cls", "code"):
            sub = [p for p in todo if p["kind"] == kind]
            for i in range(0, len(sub), 1500):
                ch = sub[i:i + 1500]
                outs = llm.chat([[{"role": "user", "content": p["prompt"]}] for p in ch], sps[kind], use_tqdm=False,
                                chat_template_kwargs={"enable_thinking": False} if cfg.get("thinking_off") else None)
                for p, o in zip(ch, outs):
                    ids = list(o.prompt_token_ids)
                    f.write(json.dumps(dict(pid=p["pid"], text=o.outputs[0].text, finish=o.outputs[0].finish_reason,
                                            n_prompt=len(ids), n_out=len(o.outputs[0].token_ids),
                                            ids_hash=hashlib.sha1(str(ids).encode()).hexdigest()[:16]),
                                       ensure_ascii=False) + "\n")
                f.flush()
                log(f"{key} {kind} {i + len(ch)}/{len(sub)} {time.time() - t0:.0f}s")
    del llm
    gc.collect()


def main():
    qfile, max_gb = sys.argv[1], float(sys.argv[2])
    finished, fails = set(), {}
    while not (LOGS / "STOP_workers").exists():
        queue = [l.strip() for l in open(qfile) if l.strip() and not l.startswith("#")]
        ran = False
        for key in queue:
            if key in finished or MODELS[key]["need"] > max_gb:
                continue
            lock = open(GENS / f"{key}.lock", "w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock.close(); continue                     # another worker has it
            if not grab(MODELS[key]["need"], timeout=30):
                lock.close(); continue
            grab(min(max_gb, MODELS[key]["need"] + 2), timeout=5)   # a bit of extra KV if it's free
            before = held()
            try:
                run_model(key)
                finished.add(key)
            except Exception as e:
                import traceback; traceback.print_exc(); log(f"FAILED {key}: {e}")
                fails[key] = fails.get(key, 0) + 1
                if fails[key] >= 3:
                    finished.add(key)
            lock.close()
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
                destroy_model_parallel(); destroy_distributed_environment()
            except Exception as e:
                log("destroy failed", e)
            gc.collect()
            grab(before, timeout=10)                       # re-take the engine's memory from torch's cache
            log(f"done {key}; holding {held():.1f} GB")
            ran = True
            break
        if not ran:
            remaining = [k for k in queue if k not in finished and MODELS[k]["need"] <= max_gb]
            if not remaining:
                log("queue empty"); time.sleep(30); continue
            grab(min(MODELS[k]["need"] for k in remaining), timeout=20)
    log("STOP seen; exiting")


if __name__ == "__main__":
    main()
