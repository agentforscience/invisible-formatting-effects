#!/bin/bash
# when qwen2.5-14b has finished classification, hand GPU0's memory to a new worker (7B, 8B first; 14B code resumes last)
cd /data/summereunann/neurico-workspaces/invisible_formatting_effects_20261029_120001_qr78st90
until grep -q "qwen2.5-14b cls 14400/14400" logs/worker_g0.log; do sleep 5; done
rm -f logs/takeover_ready
export NUMPY_MADVISE_HUGEPAGE=0
CUDA_VISIBLE_DEVICES=0 TAKEOVER_PID=2188610 TAKEOVER_GB=15 nohup .venv/bin/python src/worker.py logs/queue_g0.txt 16 > logs/worker_g0b.log 2>&1 &
until [ -f logs/takeover_ready ]; do sleep 1; done
kill 2188610
echo "handover done $(date)"
