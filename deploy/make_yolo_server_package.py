# -*- coding: utf-8 -*-
"""현장 추론서버를 남에게 넘길 수 있는 한 덩어리로 묶는다(코드 + 현재 모델 + 코드표 + 문서).

왜 필요한가
  추론서버는 우리 저장소 안에서 돌지만, XR 쪽에서는 그 서버만 따로 받아 자기 장비에서
  최적화(ONNX·TensorRT·배치)해 쓴다. 그때마다 손으로 파일을 모으면 빠지는 것이 생긴다.
  부품 판별은 응답의 detection_class(부품 이름)로 한다. 이름은 가중치 안에 저장돼 있어
  모델이 바뀌어도 그대로다. part_codes.json 은 참고용으로만 같이 넣는다.

담기는 것
  server.py            추론 API (FastAPI). /health · /detect · /reload
  model.pt             현재 서비스 모델(PyTorch)
  model.onnx           같은 모델의 ONNX (있으면)
  part_codes.json      전역 부품 코드표(폴백용)
  requirements.txt     의존성
  실행방법.md          실행·형식전환·API 규격·최적화 메모(자동 생성, 실제 값으로 채움)

사용
  python deploy/make_yolo_server_package.py
  -> dist/yolo_server_<모델id>/ 와 같은 이름의 .tar.gz 생성
"""
import json
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import config   # noqa: E402

ROOT = config.BASE_DIR
SRC_SERVER = ROOT / "backend" / "yolo_server" / "server.py"
SRC_REQ = ROOT / "backend" / "yolo_server" / "requirements.txt"
CODES = config.DATA_DIR / "bell412" / "parts" / "part_codes.json"
OUT_ROOT = ROOT / "dist"

DOC = """# 부품 인식 서버 (XR 연동용)

우리 오토러닝이 학습한 최신 모델과 함께 묶은 추론 서버다. 이 폴더만 있으면 돌아간다.

## 담긴 모델

| 항목 | 값 |
|------|-----|
| 모델 ID | {model_id} |
| 클래스 | {classes} |
| 학습 | {label} |
| 적용 시각 | {applied} |
| 파일 | model.pt {pt_mb}MB · model.onnx {onnx_mb}MB |

## 실행

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8001
```

형식 전환은 환경변수 한 줄이다(기본 model.pt).

```bash
SERVE_MODEL_FILE=model.onnx ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8001
```

## API

`POST /detect` — multipart. `image`(파일) + `camera_id`(문자열, 선택)

```json
{{
  "timestamp": "2026-08-19T10:23:44.773987",
  "camera_id": "TAB-01",
  "model_id": "{model_id}",
  "detection_count": 1,
  "detections": [
    {{"detection_class": "medicine", "detection_code": 1, "part_code": 36,
      "confidence": 0.819, "bbox": {{"x": 97.0, "y": 607.5, "w": 365.3, "h": 529.0}}}}
  ]
}}
```

| 필드 | 뜻 | 주의 |
|------|----|------|
| `detection_class` | **부품 이름. 판별은 이 값으로 한다** | 가중치 안에 저장돼 있어 모델이 바뀌어도 그대로 |
| `detection_code` | 이 모델 안에서의 클래스 인덱스 | **재학습하면 값이 바뀐다.** 판별에 쓰지 말 것 |
| `part_code` | 전역 부품 번호(참고용) | 인식 이력을 DB 에 저장할 때만 의미가 있다. 매핑 파일이 없으면 null |
| `model_id` | 모델 버전 | 값이 바뀌면 모델이 교체된 것이다 |
| `confidence` | 신뢰도 | 0.7 미만은 응답에 포함하지 않음(서버 기본값) |
| `bbox` | 픽셀 좌표(좌상단 x,y + w,h) | 입력 이미지 원본 해상도 기준 |

부품 판별은 `detection_class`(이름) 하나로 끝난다. 번호를 쓰고 싶으면 `GET /health` 의
`classes` 표(인덱스->이름)를 받아두고, 응답의 `model_id` 가 바뀔 때 다시 받으면 된다.

`GET /health` — 모델·클래스·코드표 확인. 기동 직후 이 응답으로 매핑을 검증하면 된다.

```json
{{"ok": true, "model": "model.pt", "model_id": "{model_id}",
  "classes": {classes_json}, "part_codes": {codes_json}}}
```

`POST /reload` — 모델 파일을 덮어쓴 뒤 호출하면 재시작 없이 새 모델을 읽는다.

## 부품 번호표 (part_code)

전체 표는 `part_codes.json` 에 있다. 이 모델이 아는 부품은 아래 뿐이다.

{code_table}

번호는 **추가만** 한다. 부품을 지워도 그 번호는 다시 쓰지 않는다. 새 부품이 생기면 다음 번호를 받는다.

## 성능 최적화 메모 (Jetson Thor 실측, 2026-08-19)

같은 모델·같은 사진으로 단건 지연을 재봤다.

| 방식 | 지연 | 비고 |
|------|------|------|
| `model.pt` + CUDA | **28~42ms** | **현재 채택** |
| `model.engine` (TensorRT fp16) | **20~26ms** | 최속. 변환 427초, 19MB -> 6.7MB |
| `model.onnx` + CPUExecutionProvider | 99~160ms | 3~4배 느림 |
| `model.onnx` + CUDAExecutionProvider | **기동 실패** | `QuickGelu` 노드에서 `cudaErrorNoKernelImageForDevice` (설치된 onnxruntime-gpu 에 Thor sm_110 커널 없음) |

`.pt` 를 택한 이유: 3fps 는 프레임당 333ms 예산인데 `.pt` 가 40ms 다(8배 여유).
TensorRT 로 18ms 를 더 줄일 수 있지만, 모델을 교체할 때마다 7분 변환이 필요하고
`.engine` 은 그 GPU·그 TensorRT 버전에만 쓸 수 있어 다른 장비로 복사할 수 없다.
3fps·1스트림 조건에서는 이득이 비용보다 작다고 판단했다.

TensorRT 로 전환하려면(그 장비에서 직접 변환):

```bash
python -c "from ultralytics import YOLO; YOLO('model.pt').export(format='engine', imgsz=640, half=True, device=0)"
SERVE_MODEL_FILE=model.engine ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8001
```

카메라(스트림)가 늘거나 발열·전력이 문제가 되면 그때 다시 비교하는 게 맞다.
3fps·60초 부하로는 `.pt` 와 `.engine` 의 전력·온도 차이가 측정 한계 아래였다
(GPU 전력 유휴 1969mW · `.pt` 평균 2096mW · 온도 33.4->33.5C).

- Thor 에는 `onnxruntime-gpu 1.29.0` 과 `TensorRT 10.16.2`(trtexec + python 모듈)가 이미 설치돼 있다.
  ONNX 로 GPU 를 쓰려면 **TensorrtExecutionProvider** 를 쓰거나 `trtexec` 로 engine 을 만들어야 한다.
  (`ort.get_available_providers()` 에 CUDA·TensorRT 가 보이지만, CUDA EP 는 위 오류로 막힌다)
- 입력은 640x640 기준으로 학습·변환했다. 다른 크기로 바꾸면 정확도가 달라진다.
- 동시 요청은 서버 내부에서 직렬화한다(3fps 순차 전송 조건에서는 병목이 아니다).
- `/reload` 로 모델을 바꿔도 진행 중 요청과 섞이지 않는다. 모델 교체는 추론 락 안에서 하고,
  `model_id`·부품코드는 로드 시점 스냅샷을 쓴다.

## 모델을 새로 받았을 때

`model.pt`(또는 `model.onnx`)를 덮어쓰고 `POST /reload` 를 호출하면 끝이다.
`.engine` 을 쓰는 경우에는 `.pt` 를 받은 뒤 다시 변환해야 한다(engine 은 재사용 불가).

생성 시각: {now}
"""


def served_meta() -> dict:
    """현재 서비스 모델의 정보. 원천은 results/_served.json 이 가리키는 run 의 meta.json."""
    ptr = ROOT / "results" / "_served.json"
    try:
        runid = json.loads(ptr.read_text(encoding="utf-8"))["run"]
        meta = json.loads((ROOT / "results" / runid / "meta.json").read_text(encoding="utf-8"))
        meta.setdefault("model_id", runid)
        return meta
    except Exception as e:   # noqa: BLE001 - 정보가 없어도 파일은 묶을 수 있게
        print(f"  서비스 모델 정보를 못 읽었다({type(e).__name__}) -> 이름·클래스 표기는 비운다")
        return {}


def main():
    if not config.NEW_MODEL_PT.exists():
        print(f"  현재 서비스 모델이 없다: {config.NEW_MODEL_PT}")
        return 1
    meta = served_meta()
    model_id = meta.get("model_id", "unknown")
    classes = meta.get("classes", [])
    codes = {}

    out = OUT_ROOT / f"yolo_server_{model_id}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy(SRC_SERVER, out / "server.py")
    if SRC_REQ.exists():
        shutil.copy(SRC_REQ, out / "requirements.txt")
    shutil.copy(config.NEW_MODEL_PT, out / "model.pt")
    onnx_mb = 0
    if config.NEW_MODEL_ONNX.exists():
        shutil.copy(config.NEW_MODEL_ONNX, out / "model.onnx")
        onnx_mb = round(config.NEW_MODEL_ONNX.stat().st_size / 1048576, 1)
    if CODES.exists():
        shutil.copy(CODES, out / "part_codes.json")

    table = json.loads(CODES.read_text(encoding="utf-8")) if CODES.exists() else {}
    rows = "\n".join(f"| {c} | {table.get(c, '?')} |" for c in classes)
    (out / "실행방법.md").write_text(DOC.format(
        model_id=model_id, classes=", ".join(classes) or "-", label=meta.get("label", "-"),
        applied=meta.get("applied", "-"),
        pt_mb=round((out / "model.pt").stat().st_size / 1048576, 1), onnx_mb=onnx_mb or "없음",
        classes_json=json.dumps({str(i): c for i, c in enumerate(classes)}, ensure_ascii=False),
        codes_json=json.dumps(codes, ensure_ascii=False),
        code_table="| 부품 | part_code |\n|------|-----------|\n" + rows,
        now=datetime.now().strftime("%Y-%m-%d %H:%M")), encoding="utf-8")

    tgz = out.with_suffix(".tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(out, arcname=out.name)

    print(f"  패키지: {out}")
    for f in sorted(out.iterdir()):
        print(f"    {f.name:<20} {f.stat().st_size / 1048576:>7.1f} MB")
    print(f"  압축: {tgz} ({tgz.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
