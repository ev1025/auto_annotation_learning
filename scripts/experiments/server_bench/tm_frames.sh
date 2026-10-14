#!/bin/bash
# 학습 프레임을 HLG(HDR) -> SDR 톤매핑해서 다시 뽑는다. npl=200 은 토르에서 재 본 값이다
# (기본 100 은 부품 영역 17% 가 날아가고, 200 이 대비를 가장 살린다).
#
# 프레임 번호 규칙을 그대로 맞춘다. backend/autolearning/autolabel.py 의 _extract 는
#   stride = max(1, 총프레임 // 200) 로 훑으면서 i % stride == 0 인 프레임만
#   00000.jpg 부터 순번으로 저장한다. 즉 출력 k = 원본 프레임 k*stride.
# 라벨(<stem>_<번호>.txt)이 이 번호를 가리키므로 어긋나면 학습이 엉킨다.
set -u
FF=/home/jinwoolee/bin/ffmpeg
FP=/home/jinwoolee/bin/ffprobe
W=/workspace/data2/jinwoolee/xr_dash_260901
TMV="zscale=t=linear:npl=200,tonemap=hable,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
n_vid=0

for vp in $(find "$W/data/bell412" -name "*.mp4" | sort); do
  part=$(basename "$(dirname "$(dirname "$vp")")")
  stem=$(basename "$vp" .mp4)
  dest="$W/results/autolabels/$part/images/$stem"
  [ -d "$dest" ] || { echo "건너뜀(프레임 폴더 없음) $part/$stem"; continue; }
  have=$(ls "$dest" | wc -l)

  total=$($FP -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames \
          -of default=nw=1:nk=1 "$vp" 2>/dev/null)
  [ -z "$total" ] && total=0
  stride=$(( total / 200 )); [ "$stride" -lt 1 ] && stride=1

  tmp="$W/_tmtmp"; rm -rf "$tmp"; mkdir -p "$tmp"
  $FF -v error -y -i "$vp" -vf "$TMV,select='not(mod(n\,$stride))'" -vsync 0 -q:v 2 \
      -start_number 0 "$tmp/%05d.jpg"
  made=$(ls "$tmp" | wc -l)

  if [ "$made" -lt "$have" ]; then
    echo "!! $part/$stem 기존 $have -> 새로 $made (부족, 원본 유지)"
    rm -rf "$tmp"; continue
  fi
  # 기존 번호만 교체한다(새로 더 나온 뒷번호는 라벨이 없어 쓰지 않는다)
  for f in "$dest"/*.jpg; do
    b=$(basename "$f")
    [ -f "$tmp/$b" ] && cp -f "$tmp/$b" "$f"
  done
  rm -rf "$tmp"
  n_vid=$((n_vid+1))
  echo "$(date '+%H:%M:%S') $part/$stem  총 $total · stride $stride · 교체 $have장"
done
echo "완료: 영상 $n_vid개"
du -sh "$W/results/autolabels"
