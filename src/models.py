"""Model registry. need = GB of GPU memory held for the engine (weights + KV + overhead); kv = KV-cache GB."""
MODELS = {
    "qwen2.5-0.5b": dict(hf="Qwen/Qwen2.5-0.5B-Instruct-AWQ", params=0.5, family="qwen2.5", need=3, kv=1.2),
    "qwen2.5-1.5b": dict(hf="Qwen/Qwen2.5-1.5B-Instruct-AWQ", params=1.5, family="qwen2.5", need=3.5, kv=1.2),
    "qwen2.5-3b":   dict(hf="Qwen/Qwen2.5-3B-Instruct-AWQ", params=3.1, family="qwen2.5", need=4.2, kv=1.0),
    "qwen2.5-7b":   dict(hf="Qwen/Qwen2.5-7B-Instruct-AWQ", params=7.6, family="qwen2.5", need=8, kv=1.4),
    "qwen2.5-14b":  dict(hf="Qwen/Qwen2.5-14B-Instruct-AWQ", params=14.8, family="qwen2.5", need=13, kv=1.5),
    "llama3.2-1b":  dict(hf="unsloth/Llama-3.2-1B-Instruct", params=1.2, family="llama3", need=4.2, kv=1.0),
    "llama3.2-3b":  dict(hf="AMead10/Llama-3.2-3B-Instruct-AWQ", params=3.2, family="llama3", need=4.4, kv=1.0),
    "llama3.1-8b":  dict(hf="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", params=8.0, family="llama3",
                         need=8.5, kv=1.4),
    "qwen3-0.6b":   dict(hf="Qwen/Qwen3-0.6B", params=0.6, family="qwen3", need=3.5, kv=1.0, thinking_off=True),
    "qwen3-4b":     dict(hf="Qwen/Qwen3-4B-AWQ", params=4.0, family="qwen3", need=4.6, kv=0.9, thinking_off=True),
    # replication run (same checkpoint, separate engine/GPU/batching) - excluded from the main analysis
    "rerun-qwen2.5-0.5b": dict(hf="Qwen/Qwen2.5-0.5B-Instruct-AWQ", params=0.5, family="qwen2.5", need=3, kv=1.2,
                               replicate=True),
    # served by the host's long-lived vLLM server (bf16), see src/gen_server.py
    "qwen3-8b":     dict(hf="Qwen/Qwen3-8B", params=8.2, family="qwen3", server=True),
}
