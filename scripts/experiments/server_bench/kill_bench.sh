#!/bin/bash
# 내 계정의 run_bench 프로세스만 정리(자식 워커 포함)
for P in $(pgrep -u "$(id -u)" -f "[r]un_bench.py.*panel400"); do
  kill -TERM "$P" 2>/dev/null
done
for i in $(seq 1 10); do
  [ -z "$(pgrep -u "$(id -u)" -f "[r]un_bench.py.*panel400")" ] && break
  sleep 2
done
for P in $(pgrep -u "$(id -u)" -f "[r]un_bench.py.*panel400"); do kill -KILL "$P" 2>/dev/null; done
sleep 3
echo "남은 프로세스: $(pgrep -af "[r]un_bench" | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
