#!/bin/bash
# 진행 중인 4모델 순차 실행을 멈추고, GPU2/GPU0 두 갈래로 나눈다(내 프로세스만 종료).
P=$(pgrep -f "[r]un_bench.py.*panel400" | head -1)
if [ -n "$P" ]; then
  echo "기존 실행 종료: PID $P"
  kill -TERM "$P" 2>/dev/null
  for i in $(seq 1 15); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
  kill -0 "$P" 2>/dev/null && kill -KILL "$P" 2>/dev/null
  sleep 3
fi
pgrep -af "[r]un_bench" | head -3
