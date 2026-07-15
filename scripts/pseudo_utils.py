"""pseudo_utils.py - 자동 라벨링(pseudo-labeling) 공용 유틸.

운영 스크립트(1_auto_labeling.py)와 실험 스크립트(4_experiment_autolearn.py)가
같은 박스 처리·필터 로직을 쓰므로 한 곳에 모은다(중복 구현 방지).
"""


def iou(a, b):
    """정규화 cxcywh 두 박스의 IoU."""
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def boxes_of(r, flip=False):
    """ultralytics Results -> [(클래스id, (cx,cy,w,h), conf)].

    flip=True 면 좌우반전 이미지의 결과이므로 cx 를 원본 기준으로 복원한다.
    """
    bs = []
    if r.boxes is not None and len(r.boxes) > 0:
        for (cx, cy, w, h), c, cf in zip(r.boxes.xywhn.cpu().numpy(),
                                         r.boxes.cls.cpu().numpy().astype(int),
                                         r.boxes.conf.cpu().numpy()):
            if flip:
                cx = 1.0 - cx
            bs.append((int(c), (float(cx), float(cy), float(w), float(h)), float(cf)))
    return bs


def consistent(cand, others, iou_thr=0.8):
    """TTA 일관성: 후보 박스가 다른 모든 뷰에서 같은 클래스 + IoU>=iou_thr 로 재현되는지."""
    c, box, _ = cand
    return all(any(oc == c and iou(box, ob) >= iou_thr for oc, ob, _ in other)
               for other in others)


def parse_conf_per_class(spec, class_names):
    """'gear=0.45,bolt=0.7' -> {클래스id: 임계값}. class_names = {id: 이름}.

    쉬운 클래스만 pseudo 라벨이 양산되는 '클래스 소외' 완화용.
    """
    if not spec:
        return {}
    name_to_id = {v: k for k, v in class_names.items()}
    out = {}
    for part in spec.split(","):
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in name_to_id:
            raise SystemExit(f"[오류] --conf-per-class 클래스가 정의에 없음: {name} / 보유: {list(name_to_id)}")
        out[name_to_id[name]] = float(val)
    return out
