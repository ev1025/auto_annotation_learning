#!/bin/bash
for P in $(pgrep -u "$(id -u)" -f "[c]hain_3x_rest"); do kill -TERM "$P" 2>/dev/null && echo "대기 스크립트 종료: $P"; done
sleep 2
pgrep -af "[c]hain_3x" | head -3
