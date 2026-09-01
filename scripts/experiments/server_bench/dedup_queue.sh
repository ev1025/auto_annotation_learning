#!/bin/bash
# queue_gpu2.sh 가 중복 실행됐다. 가장 오래된 하나만 남긴다(추가 모델이 두 번 돌지 않게).
PIDS=$(pgrep -u "$(id -u)" -f "[q]ueue_gpu2\.sh" | sort -n)
KEEP=$(echo "$PIDS" | head -1)
for p in $PIDS; do
  [ "$p" = "$KEEP" ] && continue
  kill -TERM "$p" 2>/dev/null && echo "중복 종료: $p"
done
sleep 2
echo "남은 대기 스크립트:"
pgrep -af "[q]ueue_gpu[02]\.sh" | sed "s|/workspace/data2/jinwoolee/||" | grep -v "bash -c" | sed "s/^/  /"
