#!/bin/bash
# HLG(HDR) 평가영상을 SDR 로 톤매핑해 같은 프레임 번호로 뽑는다.
# npl=200: 기본값 100 은 부품 영역의 17% 가 날아갔고(포화), 200 이 대비를 가장 살렸다.
set -u
SRC=/home/mrod/xr_autolearning/data/bell412/_gt_videos
NAIVE=/home/mrod/xr_autolearning/results/gt_frames
OUT=/home/mrod/xr_autolearning/results/gt_frames_tm
TMP=/home/mrod/tm_tmp
mkdir -p "$OUT"
for stem_dir in "$NAIVE"/*/; do
  stem=$(basename "$stem_dir")
  vid=$(ls "$SRC/$stem".* 2>/dev/null | head -1)
  [ -z "$vid" ] && { echo "영상 없음: $stem"; continue; }
  need=$(ls "$stem_dir" | grep -oE '^[0-9]+' | sort -n)
  cnt=$(echo "$need" | wc -l)
  rm -rf "$TMP"; mkdir -p "$TMP" "$OUT/$stem"
  ffmpeg -v error -y -i "$vid" \
    -vf "zscale=t=linear:npl=200,tonemap=hable,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" \
    -vsync 0 -q:v 2 "$TMP/%06d.jpg"
  made=0
  for i in $need; do
    n=$((10#$i + 1))                        # ffmpeg 출력은 1 부터 시작
    src=$(printf '%s/%06d.jpg' "$TMP" "$n")
    [ -f "$src" ] && cp "$src" "$(printf '%s/%s/%05d.jpg' "$OUT" "$stem" "$((10#$i))")" && made=$((made+1))
  done
  echo "$stem  필요 $cnt · 생성 $made"
done
rm -rf "$TMP"
echo "추출 완료"; du -sh "$OUT"
