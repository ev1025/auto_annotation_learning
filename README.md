# XR 오토러닝 (XR Auto-Learning) - 수리온 부품 비마커 인식 파이프라인

수리온 헬기 정비 부품을 마커 없이(Markerless) 인식하기 위한 YOLO(Ultralytics) 기반 XR 오토러닝 end-to-end 파이프라인.
자동 라벨링 → 학습/ONNX 변환 → REST 추론 서버까지 3단계로 구성했고, 타 부서(XR/프론트엔드)에는 마지막 REST API 형태로 제공한다.

## 1. 오토러닝이 무엇이고 왜 필요한가

오토러닝(self-training)은 다음 루프를 반복하는 방식이다.

- 소량의 손라벨 데이터로 첫 모델을 만든다
- 그 모델이 라벨 없는 새 이미지에 스스로 라벨(바운딩 박스)을 친다
- 신뢰도(confidence) 기준을 넘는 자동 라벨만 골라 손라벨과 합쳐 모델을 재학습한다
- 다음 라운드부터는 더 강해진 모델이 더 많은 이미지를 라벨링 → 학습 데이터가 누적되며 모델이 점점 강해진다

필요한 이유는 다음 세 가지다.

- **라벨링 물량 문제**: 비마커 인식은 마커 인식과 달리 부품의 형태·각도·조명 변화를 전부 학습해야 해서 필요한 라벨 수가 많다. 전량 손라벨은 시간·인건비가 감당되지 않는다.
- **자동 라벨의 신뢰도**: 신뢰도(conf) 임계값으로 걸러진 자동 라벨은 실측 정밀도 0.87~0.90 수준으로, 사람이 손대지 않아도 학습에 쓸 만한 품질이 나온다. 라벨링 노동을 모델이 대신하는 것이 핵심이다.
- **실측으로 증명해야 한다**: "될 것 같다"가 아니라 mAP 수치로 오토러닝이 실제로 성능을 올리는지 확인이 필요했다. 그 결과는 [6. 실험 과정](#6-실험-과정)에 정리했다.

## 2. 어떤 데이터가 필요하고, 어떤 데이터를 썼는가

### 실전 파이프라인이 필요로 하는 데이터

| 데이터 | 역할 | 준비 방법 |
|--------|------|-----------|
| `models/base_model.pt` | 오토러닝의 시작점(첫 라벨 생성기) | 소량(수십~백여 장)의 수리온 부품을 손라벨해 학습 |
| `datasets/unlabeled_images/` | 자동 라벨링 대상 | 라벨 없는 대량의 부품 이미지(다양한 각도·조명) |

### 지금 이 리포에서 쓴 데이터 (실증 실험용)

현재 실제 수리온 부품 데이터가 없어, **오토러닝 방식 자체가 유효한지**를 먼저 공개 데이터로 검증했다.

- 데이터셋: [Mechanical Parts Dataset 2022](https://universe.roboflow.com/mazhar-cakir/mechanical-parts) (Roboflow, 원본 Zenodo)
- 규모: 총 2,250장, 4클래스(bearing / bolt / gear / nut), 어노테이션 10,599건
- 원본 포맷: Roboflow COCO export (`_annotations.coco.json` + 이미지)
- 변환: `0_coco_to_yolo.py` 로 YOLO txt 포맷(중심좌표+크기, 0~1 정규화)으로 변환. Roboflow가 자동 삽입하는 더미 상위 카테고리는 제거
- 위치: `mechanical-parts-coco/`(원본), `mechanical-parts-yolo/`(변환본)

실제 수리온 부품 데이터가 확보되면 **클래스명(`data.yaml`)과 이미지만 교체**하면 동일 파이프라인·스크립트를 그대로 쓸 수 있다.

## 3. 폴더 구조

```
xr_autolearning/
├─ config.py                    # 경로/클래스/임계값 공통 설정
├─ 0_import_roboflow.py         # Roboflow YOLOv8 export → 파이프라인 구조 변환
├─ 0_coco_to_yolo.py            # Roboflow COCO export → YOLO 포맷 변환
├─ 1_auto_labeling.py           # base_model 로 자동 라벨링
├─ 2_train_pipeline.py          # 학습 + ONNX 변환
├─ 3_api_server.py              # FastAPI 추론 서버
├─ 4_experiment_autolearn.py    # 오토러닝 효과 실증 실험(정답 숨기고 자동 채점)
├─ data.yaml                    # 학습 데이터 정의(train/val 경로, 클래스명)
├─ requirements.txt
├─ models/
│   ├─ base_model.pt            # (사용자 준비) 소량 데이터 초기 가중치
│   ├─ new_model.pt             # 2단계 산출
│   └─ new_model.onnx           # 2단계 산출
├─ datasets/
│   ├─ unlabeled_images/        # 1단계 입력
│   ├─ images/                  # 라벨링된 이미지(train/val 분할)
│   └─ labels/                  # YOLO .txt(train/val 분할)
└─ exp_results/                 # 실증 실험 결과(report_*.json, round1_best.pt)
```

## 4. 워크플로우가 어떻게 되는가

첫 모델만 사람이 손라벨로 만들고, 그 뒤로는 "모델이 자동 라벨 → 그 라벨로 재학습"을 반복하며 같은 모델 자리(`new_model`)를 점점 더 센 것으로 갈아끼운다. 손라벨은 첫 라운드 한 번뿐이고, 이후 라벨링은 모델이 대신한다.

```
범례   사람 = 사람이 직접   자동 = 스크립트가 자동   [폴더] 데이터   <모델> 가중치(.pt)

──────────────────────────────────────────────────────────────
 0단계  부트스트랩 (딱 한 번. 사람 손이 들어가는 유일한 구간)
──────────────────────────────────────────────────────────────
   사람) 미라벨 이미지 소량(예: 100장)을 0부터 손라벨
        │
        ▼
   [datasets/images] + [datasets/labels]   (순수 수기 정답)
        │
        ▼  자동) python 2_train_pipeline.py
   <models/new_model.pt> 생성 (첫 모델)
        │
        ▼  사람) 이 모델을 "첫 라벨 생성기"로 승격 (파일 복사)
   <models/base_model.pt>

──────────────────────────────────────────────────────────────
 반복 루프  (오토러닝: 라운드마다 데이터↑, 모델 갱신 / 전 과정 자동)
──────────────────────────────────────────────────────────────
   사람) 새 미라벨 이미지 추가  (매 라운드 '다른' 사진들)
        │
        ▼
   [datasets/unlabeled_images/]
        │
        ▼  자동) python 1_auto_labeling.py --weights <최신 모델>
        │        conf >= 0.6 넘는 것만 자동으로 박스(신뢰도 필터)
        ▼
   [datasets/labels/]  (자동 라벨 .txt)
        │
        ▼  자동) python 2_train_pipeline.py
        │        수기 + 자동 라벨 = '누적' 데이터로 재학습
        ▼
   <models/new_model.pt> (갱신, 더 강함) + <models/new_model.onnx>
        │
        └──► 이 최신 모델을 다음 라운드 --weights 로 투입 ──┐
        ▲                                                    │
        └────────────────────────────────────────────────────┘
            라운드 반복 → 데이터 누적 → 모델 성능 상승

──────────────────────────────────────────────────────────────
 배포  (모델이 충분히 좋아지면)
──────────────────────────────────────────────────────────────
   <models/new_model.pt (또는 .onnx)>
        │
        ▼  자동) python 3_api_server.py
   POST /predict ──JSON(bbox x,y,w,h / class / conf)──► XR·프론트 팀
```

| 구간 | 사람이 하는 일 | 자동(스크립트/모델) |
|------|----------------|---------------------|
| 0단계 | 소량 손라벨 1번 | `2_train_pipeline.py` 첫 모델 학습 |
| 라벨 생성 | (안 함) | `1_auto_labeling.py` 로 최신 모델이 자동 박스(conf 필터) |
| 재학습 | (안 함) | `2_train_pipeline.py` 로 누적 데이터 재학습 |
| 다음 라운드 | 새 이미지 넣기 | 최신 모델을 생성기로 재사용 |
| 배포 | API 호출만 | `3_api_server.py` 가 추론 서빙 |

주의: 쌓이는 것은 '서로 다른 새 이미지'다(데이터 폭을 넓히는 것). 같은 이미지를 박스만 바꿔 반복 축적하는 게 아니며, 이미지 1장당 최종 정답 라벨은 1개다.

## 5. 산출물 구성

| 파일 | 역할 | 핵심 산출물 |
|------|------|-------------|
| `config.py` | 경로/클래스/임계값 공통 설정 | 전 스크립트 공유 상수 |
| `1_auto_labeling.py` | base_model 로 자동 라벨링 | `datasets/labels/*.txt` (YOLO 포맷) |
| `2_train_pipeline.py` | 학습 + ONNX 변환 | `models/new_model.pt`, `models/new_model.onnx` |
| `3_api_server.py` | FastAPI 추론 서버 | `POST /predict` JSON 응답 |
| `4_experiment_autolearn.py` | 오토러닝 효과 실증 | `exp_autolearn/report.json` (mAP 비교) |

## 6. 실험 과정

### 실험 설계 (정답 숨기고 자동 채점)

**정답 라벨을 알고 있는 공개 데이터에서 일부 라벨을 가려 "미라벨인 척"** 만들면, 자동 라벨링 정확도와 재학습 효과를 사람 개입 없이 채점할 수 있다. `4_experiment_autolearn.py` 가 이 실험을 자동화한다.

1. 전체 라벨 데이터를 무작위로 3분할한다: **시드**(손라벨 흉내, 학습에 라벨 사용) / **풀**(라벨 숨김, 이미지만 미라벨 취급) / **테스트**(측정 전용, 어떤 학습에도 안 씀)
2. **라운드0**: 시드만으로 YOLOv8n을 학습 → 테스트 mAP 측정(오토러닝 하기 전 베이스라인)
3. 라운드0 모델로 풀 이미지에 자동 라벨(pseudo label, conf≥0.6) 생성
4. 생성된 자동 라벨을 **숨겨둔 정답과 IoU 매칭**해서 정밀도/재현율 채점
5. **라운드1**: 시드 + 자동 라벨을 합쳐 재학습 → 테스트 mAP 측정(오토러닝 후)
6. 라운드0 대비 라운드1의 mAP 변화가 오토러닝의 실제 효과다

공통 조건: YOLOv8n, imgsz 640, epochs 60, RTX 5090.

이 실험은 conf≥0.6으로 걸러진 자동 라벨을 그대로 재학습에 넣는 전자동 조건이다. 정밀도/재현율은 생성된 자동 라벨을 숨겨둔 정답과 자동 대조해 계산한 값으로, 자동 라벨 자체의 신뢰도를 나타낸다.

### 실험 1. 2클래스(bolt·nut), 시드 15%

- 클래스: bolt, nut
- 분할: 시드 149장 / 풀 647장 / 테스트 199장
- 결과: mAP50 **0.835 → 0.859 (+2.3%p)**, mAP50-95 0.645 → 0.674 (+2.9%p)
- 자동 라벨 품질: 622장에 3,339개 박스 생성, 정밀도 0.902 / 재현율 0.805

### 실험 2. 2클래스(bolt·nut), 시드 10%

- 목적: 초기 손라벨을 더 줄여도(149장→99장) 효과가 유지되는지 확인
- 분할: 시드 99장 / 풀 697장 / 테스트 199장
- 결과: mAP50 **0.819 → 0.839 (+2.0%p)**, mAP50-95 0.623 → 0.663 (+4.0%p)
- 자동 라벨 품질: 651장에 3,448개 박스 생성, 정밀도 0.885 / 재현율 0.765
- 확인된 것: 시드를 34% 줄여도 개선폭이 거의 유지됨 → 초기 손라벨 부담을 낮춰도 방식이 유효

### 실험 3. 3클래스(bolt·nut·gear)

- 목적: 클래스 수를 늘려도(부품 종류 확장) 효과가 재현되는지 확인
- 분할: 시드 236장 / 풀 1,025장 / 테스트 315장
- 결과: mAP50 **0.825 → 0.835 (+1.0%p)**, mAP50-95 0.649 → 0.677 (+2.8%p)
- 자동 라벨 품질: 962장에 4,401개 박스 생성, 정밀도 0.901 / 재현율 0.715

### 실험 4. 4클래스 전체(bolt·nut·gear·bearing)

- 목적: 데이터셋 전체 규모에서도 효과가 나는지 확인
- 분할: 시드 337장 / 풀 1,463장 / 테스트 450장
- 결과: mAP50 **0.827 → 0.863 (+3.6%p, 4개 실험 중 최대 개선)**, mAP50-95 0.649 → 0.680 (+3.1%p)
- 자동 라벨 품질: 1,361장에 5,515개 박스 생성, 정밀도 0.871 / 재현율 0.726
- 진행 중 겪은 문제: 데이터가 가장 큰 조건이라 멀티GPU(DDP) 학습 직후 GPU 메모리가 회수되지 않아 자동 라벨링 단계에서 OOM이 반복 발생. 배치 축소는 효과가 없었고, **단일 GPU 학습으로 전환**해 해결(YOLOv8n은 5090 한 장으로 충분).

### 종합 결과

| 실험 | 시드(손라벨 역할) | mAP50 (라운드0→1) | 변화 | pseudo 정밀도/재현율 |
|------|-------------------|---------------------|------|----------------------|
| 2클래스, 시드 15% | 149장 | 0.835 → 0.859 | **+2.3%p** | 0.902 / 0.805 |
| 2클래스, 시드 10% | 99장 | 0.819 → 0.839 | **+2.0%p** | 0.885 / 0.765 |
| 3클래스 | 236장 | 0.825 → 0.835 | **+1.0%p** | 0.901 / 0.715 |
| 4클래스 전체 | 337장 | 0.827 → 0.863 | **+3.6%p** | 0.871 / 0.726 |

- 4개 실험 전부 라운드1(오토러닝 후)이 라운드0(초기 손라벨만)보다 높음 → 오토러닝 효과가 클래스 수·시드 비율에 걸쳐 일관되게 재현됨
- 자동 라벨 정밀도는 4개 실험 모두 0.87~0.90 수준 유지(사람 개입 0으로도 신뢰 가능한 수준)
- 원본 리포트: `exp_results/report_*.json`, 4클래스 최종 모델: `exp_results/round1_best.pt`

### 반증 실험 (라벨 품질이 왜 중요한지)

실험 중 스크립트 버그로 자동 라벨과 이미지의 짝이 깨진 채 학습된 사례가 있었다. 이때는 mAP50이 **−19.7%p** 폭락했다(부품을 전부 "배경"으로 잘못 학습). 자동 라벨이 이미지와 정확히 매칭되는지(라벨 품질)가 오토러닝 성패를 가르는 핵심임을 같은 조건에서 보여주는 반증 데이터로 남긴다.

### 실험 중 발견한 실무 이슈

- **ultralytics 8.4 함정**: `predict(source=리스트)` 결과의 `r.path` 가 원본 파일명 대신 `image0` 같은 가짜 이름을 반환한다. pseudo 라벨 파일명이 이미지와 안 맞게 되어 위 반증 실험의 원인이 됐다. 입력 리스트와 `zip` 으로 순서 매핑해 해결(`1_auto_labeling.py`, `4_experiment_autolearn.py` 반영).
- **DDP(멀티GPU) 학습 후 메모리 미회수**: 클래스가 많아 데이터가 큰 조건(4클래스)에서 `device=0,1` DDP 학습 후 GPU 메모리가 프로세스에 남아 다음 단계(자동 라벨링)에서 OOM 발생. 배치를 줄이는 미봉책은 재발했고, YOLOv8n은 GPU 한 장으로 충분해 단일 GPU 학습으로 전환해 근본 해결.

## 7. 사전 준비

```bash
pip install -r requirements.txt
# GPU 사용 시 torch 는 CUDA 버전 맞춰 별도 설치 권장
```

- `models/base_model.pt` 를 직접 넣는다.
- `datasets/unlabeled_images/` 에 미라벨 이미지를 넣는다.
- `data.yaml` 의 `names` 를 base_model 클래스와 동일하게 맞춘다.
  - 확인: `python -c "from ultralytics import YOLO; print(YOLO('models/base_model.pt').names)"`

## 8. 실행 순서 (입력 → 출력)

### 1단계. 자동 라벨링
```bash
python 1_auto_labeling.py
```
- 입력: `datasets/unlabeled_images/` + `models/base_model.pt`
- 처리: 추론 후 confidence ≥ 0.6 만 채택
- 출력: `datasets/labels/*.txt` (YOLO 정규화 좌표), `datasets/images/*` 복사
- 옵션: `--conf 0.7`, `--no-copy-images`, `--keep-empty`

### 2단계. 학습 + ONNX 변환
```bash
python 2_train_pipeline.py
```
- 입력: `datasets/images` + `datasets/labels`, `data.yaml`
- 처리: train/val 자동 분할 → 학습 → best.pt 추출 → ONNX export
- 출력: `models/new_model.pt`, `models/new_model.onnx`, `data.generated.yaml`(자동 생성)
- 옵션: `--epochs 50 --batch 8 --device 0`
- 참고: 학습 시 `data.yaml` 의 `path` 를 절대경로로 고정한 `data.generated.yaml` 을 자동 생성해 사용한다. ultralytics 가 상대 `path` 를 전역 datasets_dir 기준으로 해석해 "Dataset not found" 가 나는 문제를 막는다. 사람이 편집하는 `data.yaml`(클래스명)은 그대로 둔다.

### 3단계. 추론 API 서버
```bash
python 3_api_server.py
```
- 서버: `http://0.0.0.0:8000`
- 테스트:
  ```bash
  curl -X POST -F "file=@sample.jpg" "http://localhost:8000/predict?conf=0.3"
  ```

## 9. API 명세

| 항목 | 내용 |
|------|------|
| 엔드포인트 | `POST /predict` |
| 요청 | multipart/form-data, 필드명 `file` |
| 쿼리(옵션) | `conf`(기본 0.25), `iou`(기본 0.45) |
| 상태확인 | `GET /health` |

응답 예시:
```json
{
  "filename": "part.jpg",
  "image": {"width": 1280, "height": 720},
  "count": 1,
  "detections": [
    {"class_id": 0, "class_name": "bolt", "confidence": 0.91,
     "bbox": {"x": 120.5, "y": 80.0, "w": 64.0, "h": 64.0}}
  ]
}
```

- `bbox` 좌표 약속: `x,y` = 박스 좌상단 픽셀, `w,h` = 너비/높이 픽셀. XR/프론트엔드가 그대로 사각형을 그릴 수 있는 형태.

## 10. 설계 메모

- 3개 스크립트가 `config.py` 한 곳의 경로/임계값을 공유한다. 경로 변경은 config 한 곳만 수정.
- 서빙 모델은 `config.SERVE_MODEL` 로 `.pt`/`.onnx` 전환 가능. ONNX 서빙은 `onnxruntime` 필요.
- ONNX export 는 `opset=12`, `dynamic=False`, `simplify=True` 로 비파이썬(Unity/C#) 호환성을 우선했다.
- 학습 시 train/val 을 실제로 분할해 동일 데이터 평가(과적합 착시)를 방지한다.
