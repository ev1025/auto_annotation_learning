"""pseudo_utils.py - 자동 라벨링(pseudo-labeling) 공용 유틸.

운영 스크립트(1_auto_labeling.py)와 실험 스크립트(4_experiment_autolearn.py)가
같은 로직을 쓰므로 한 곳에 모은다. 특히 predict_boxes() 는 '박스 선택 규칙'의
단일 구현이다: 실험이 검증하는 규칙과 운영이 쓰는 규칙이 두 벌로 갈라져
몰래 어긋나는 것(drift)을 구조적으로 막는다.
"""
import gc

import torch
from PIL import Image


def free_cuda():
    """무거운 학습/추론 직후 GPU 메모리가 곧바로 안 풀리는 경우가 있어 명시적으로 회수한다.

    (교훈) 대용량 조건에서 학습 직후 다음 단계가 OOM 으로 죽는 사고가 있었다.
    학습 -> 추론/변환으로 넘어가는 경계마다 호출한다.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    if r.boxes is not None:
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


def _tta_variants(im):
    """TTA 3뷰: 원본 / 좌우반전 / 0.8배 축소."""
    return [im,
            im.transpose(Image.FLIP_LEFT_RIGHT),
            im.resize((max(32, int(im.width * 0.8)), max(32, int(im.height * 0.8))))]


def predict_boxes(model, imgs, conf, conf_per_class=None, tta=False, tta_batch=8):
    """자동 라벨링 박스 선택의 단일 구현. 이미지별 (경로, 채택 박스, tta 탈락 수) 를 yield.

    - 채택 박스 = conf(클래스별 임계값 포함) 필터를 이미 통과한 것. 호출부 재필터 불필요.
    - 추론은 가장 낮은 임계값으로 하고 채택 단계에서 클래스별 임계값을 적용한다.
    - (주의) 리스트 source 에서 ultralytics 8.4 의 r.path 는 'image0' 같은 가짜 이름이
      될 수 있다. 결과 순서는 입력 순서를 보존하므로 입력 리스트와 zip/인덱스로 짝짓는다.
    - tta=True: 원본/반전/축소 3뷰 모두에서(같은 클래스, IoU>=0.8) 재현되는 예측만 채택.
      '자신 있게 틀리는' 예측 방어(도메인 갭 구간용). 뷰는 이미지 tta_batch 장씩 묶어
      한 번에 추론해 호출 오버헤드를 줄인다(이미지당 낱개 호출 금지).
    """
    imgs = list(imgs)
    cpc = conf_per_class or {}
    min_conf = min([conf, *cpc.values()]) if cpc else conf

    def ok(b):
        return b[2] >= cpc.get(b[0], conf)

    if not tta:
        results = model.predict(source=[str(p) for p in imgs], conf=min_conf,
                                stream=True, verbose=False)
        for img_path, r in zip(imgs, results):
            yield img_path, [b for b in boxes_of(r) if ok(b)], 0
        return

    for i in range(0, len(imgs), tta_batch):
        chunk = imgs[i:i + tta_batch]
        variants = []
        for p in chunk:
            variants += _tta_variants(Image.open(p).convert("RGB"))
        rs = model.predict(source=variants, conf=min_conf, verbose=False)
        for j, img_path in enumerate(chunk):
            cands = [b for b in boxes_of(rs[3 * j]) if ok(b)]
            others = [boxes_of(rs[3 * j + 1], flip=True), boxes_of(rs[3 * j + 2])]
            kept = [b for b in cands if consistent(b, others)]
            yield img_path, kept, len(cands) - len(kept)
