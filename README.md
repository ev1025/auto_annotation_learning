# XR 오토러닝 (XR Auto-Learning) - 헬기 부품 비마커 인식 파이프라인

## 1. 프로젝트 개요

헬기 정비 부품을 마커(QR/바코드) 없이 인식하는 YOLO 객체탐지 모델을 **사람 개입 없이(zero-manual) 자동으로 학습·배포하는 MLOps 파이프라인**이다. 3D 렌더링으로 생성된 (이미지 + 어노테이션)을 입력으로 받아, 전처리 → 자동 라벨 확장 → YOLO 학습 → ONNX/API 배포까지 하나의 루프로 돌린다.

**데이터 생성 방식 (요약)**: 학습 데이터는 수작업 라벨링 없이 확보한다. 3D 렌더링 과정에서 객체의 3D 좌표를 이미지 평면에 투영해 정확한 2D 바운딩 박스를 자동 산출하므로, 렌더 1장마다 오차 없는 정답 라벨이 함께 나온다. (3D 렌더·어노테이션 생성은 외부에서 공급받고, 본 파이프라인은 이 데이터를 받아 학습·배포한다.)

**핵심 기능**

- **자동 데이터 파이프라인**: 입력 데이터 → YOLO 포맷 정규화 → train/val 분할까지 무인 처리
- **자동 라벨 확장(self-training)**: 학습된 모델이 실제 미라벨 이미지에 conf ≥ 0.6 라벨을 생성해 데이터셋을 자동 증분·재학습
- **학습·변환 자동화**: YOLOv8 학습 → best.pt 추출 → ONNX export를 단일 스크립트로
- **추론 서빙**: FastAPI `/predict` + ONNX로 박스/클래스/신뢰도 JSON 제공

## 2. 시스템 아키텍처 및 파이프라인

3D 렌더링 합성 데이터를 활용한 YOLO 자동 라벨링·학습 파이프라인이다. 데이터 입수부터 학습·배포까지 사람 손을 타지 않고 단일 루프로 자동화하는 것이 설계 목표다.

데이터 흐름:

```
[3D 렌더 합성 데이터]  (이미지 + 자동 어노테이션, 외부 공급)
   │
   ▼
[전처리]  YOLO 포맷 정규화 / train·val 분할        (무인)
   │
   ▼
[학습]  YOLOv8n (imgsz 640) → best.pt → ONNX      (무인)
   │
   ├─(추론)→ 실제 미라벨 이미지에 자동 라벨(conf≥0.6) ─┐
   │                                                   │ 누적
   ▼                                                   ▼
[자동 라벨 확장 → 재학습]  데이터↑ → 성능↑          (무인)
   │
   ▼
[배포]  FastAPI /predict + ONNX → JSON(bbox/class/conf) → XR·프론트
```

MLOps 자동화 루프 (zero-manual, 첫 데이터 확보 이후 전 과정 무인):

```
범례   입력 = 외부 데이터 투입   자동 = 스크립트   [폴더] 데이터   <모델> 가중치

0단계 부트스트랩(1회)
  입력) 3D 렌더 합성 데이터(이미지+어노테이션) 확보 → [datasets/images]+[datasets/labels]
  자동) 2_train_pipeline.py → <new_model.pt>
  자동) new_model.pt 를 <base_model.pt>(첫 생성기)로 승격

반복 루프(전자동)
  입력) 새 미라벨 이미지 추가 → [datasets/unlabeled_images]
  자동) 1_auto_labeling.py --weights <최신모델>  (conf≥0.6 필터)
        → [datasets/labels] (자동 라벨)
  자동) 2_train_pipeline.py  (누적 데이터 재학습)
        → <new_model.pt> 갱신
  └─► 최신 모델을 다음 라운드 생성기로 재투입 (데이터↑ → 성능↑)

배포
  자동) 3_api_server.py  → <new_model.pt / .onnx> 서빙
```

> 아키텍처 다이어그램 자리: 위 ASCII 흐름을 정식 다이어그램(draw.io/Excalidraw)으로 대체 예정.

컴포넌트 연동:

| 계층 | 구성 |
|------|------|
| 모델 | YOLOv8n (Ultralytics), 단일 스테이지 탐지기 |
| 백엔드 | FastAPI + Uvicorn (`/predict`, `/health`) |
| 배포 포맷 | PyTorch `.pt`(서빙) / ONNX(Unity·C# 등 비파이썬 런타임) |
| 설정 | `config.py` 단일 소스(경로·임계값·하이퍼파라미터) |

## 3. 기술 스택

**Data & AI**

- 언어: Python 3.10+
- 탐지 프레임워크: Ultralytics YOLOv8 (8.4.93), 모델 YOLOv8n
- 딥러닝: PyTorch 2.x (RTX 5090은 CUDA 12.8 빌드, 그 외 GPU는 CUDA 버전에 맞춰 설치)
- 변환·서빙 런타임: ONNX, onnxruntime, onnxslim
- 데이터 처리: NumPy, Pillow, PyYAML

**Infra & Backend**

- API: FastAPI, Uvicorn, python-multipart
- 실행 환경: 단일 GPU(CUDA). 실험 환경 = NVIDIA RTX 5090 1장
- 형상 관리: Git (GitHub: `ev1025/auto_annotation_learning`)
- 데이터셋/모델 가중치는 `.gitignore`로 제외(용량·재생성 가능), 소스·결과 리포트(json)만 추적

## 4. 데이터 전처리 및 모델링

### 4.1 데이터 소스

- **학습 데이터**: 3D 렌더링 합성 데이터(이미지 + 자동 어노테이션). 렌더 시점에 3D 좌표를 투영해 정답 라벨이 함께 생성되므로 수작업 라벨링이 없다.
- **현재 검증 데이터**: 실제 대상 부품의 3D 렌더 데이터 확보 전이라, 학습·자동라벨·평가 파이프라인의 유효성을 공개 라벨 데이터로 대체 검증했다.
  - 데이터셋: Mechanical Parts Dataset 2022 (Roboflow, 원본 Zenodo)
  - 규모: 2,250장, 4클래스(bearing/bolt/gear/nut), 어노테이션 10,599건 (train 1,800 / valid 225 / test 225)
  - 원본 포맷: Roboflow COCO export (`_annotations.coco.json` + 이미지)
- 입력이 3D 렌더든 공개 라벨이든 이후 학습·자동라벨·평가 파이프라인은 동일하다. 검증 대상은 "라벨셋 → YOLO 학습 → 자동라벨 확장 → 성능 향상" 자동화 루프다.

### 4.2 전처리 전략

- **COCO→YOLO 변환** (`0_coco_to_yolo.py`): COCO bbox `[x_min, y_min, w, h]`(픽셀) → YOLO `[x_center, y_center, w, h]`(이미지 크기로 0~1 정규화). 경계 밖 좌표는 `[0,1]`로 클램프.
- **더미 클래스 제거**: Roboflow COCO는 상위 supercategory(id 0)를 더미로 삽입한다. 어노테이션이 실제로 달린 카테고리만 채택해 `0..k-1`로 재매핑.
- **이상 라벨 방어**: 5개 필드 미만 라벨 라인은 스킵. 이미지-라벨 파일명 stem 1:1 매칭 검증.
- **train/val 분할** (`2_train_pipeline.py`): 라벨(.txt)이 실존하는 이미지만 대상. 고정 seed 셔플로 재현 가능, val 비율 기본 0.2, 최소 1장 val 보장. 원본 보존(copy), 재실행 멱등.
- **경로 함정 회피**: ultralytics는 `data.yaml`의 상대 `path`를 전역 `datasets_dir` 기준으로 해석해 "Dataset not found"를 유발한다. 학습 시 절대경로를 박은 `data.generated.yaml`을 생성해 사용(사람이 편집하는 `data.yaml`은 클래스명 정의용으로 보존).

### 4.3 자동 라벨링 로직 (모델 작동 근거)

- **신뢰도 임계값 0.6**: 자동 라벨은 conf ≥ 0.6만 채택. 재현율보다 **정밀도 우선**으로, 오라벨이 학습에 유입되는 것을 억제. 놓친 객체는 다음 라운드의 강해진 모델이 회수.
- **좌표 변환 무손실**: `boxes.xywhn`(정규화 중심좌표)을 그대로 YOLO 라벨로 기록.
- **재학습 전략(콜드 리스타트)**: self-training 비교 실험에서는 매 라운드 동일 사전학습 가중치(`yolov8n.pt`)에서 새로 학습. warm start는 자동 라벨 오류를 가중치에 고착(confirmation bias)시킬 수 있어 배제. 성능 향상의 동력을 "가중치 이어받기"가 아닌 "누적된 데이터"로 격리해 해석.

### 4.4 하이퍼파라미터 및 검증 전략

| 항목 | 값 |
|------|-----|
| 모델 | YOLOv8n (사전학습 `yolov8n.pt`에서 전이학습) |
| 입력 크기 | 640 |
| epochs | 60 (실험), 기본 100 (운영 스크립트) |
| batch | 32 (단일 GPU), 대용량 조건은 24 |
| optimizer | auto (Ultralytics 자동 선택) |
| best 선택 | `model.trainer.best` (val 기준 best.pt) |

- **검증 분할**: 현재는 고정 seed 단일 홀드아웃(시드/풀/테스트). k-fold 교차검증은 미적용(한계점 및 향후 과제 참고).
- **실험용 분할**(`4_experiment_autolearn.py`): 전체 라벨 데이터를 시드(초기 라벨셋 역할)/풀(라벨 숨김)/테스트(측정 전용)로 3분할. 테스트셋은 어떤 라운드 학습에도 미사용.

## 5. 성능 평가 지표

### 5.1 평가 방법

- **탐지 성능**: mAP50, mAP50-95 (Ultralytics `val`, 테스트셋 기준)
- **자동화 루프 효과**: 라운드0(초기 라벨셋만) 대비 라운드1(+자동라벨) mAP 변화량(Δ)
- **자동 라벨 품질**: 생성된 pseudo 라벨을 숨겨둔 정답과 클래스 일치 + IoU ≥ 0.5 그리디 매칭해 정밀도/재현율 산출 (사람 개입 0)
- **재현성**: 데이터 분할·셔플에 고정 seed 사용

### 5.2 자동 라벨 확장 효과 (4개 조건)

| 실험 | 시드(초기 라벨) / 풀 / 테스트 | mAP50 (R0→R1) | ΔmAP50 | mAP50-95 (R0→R1) | 자동라벨 P/R |
|------|---------------------------|----------------|--------|-------------------|--------------|
| 2클래스, 시드15% | 149 / 647 / 199 | 0.835 → 0.859 | **+2.3%p** | 0.645 → 0.674 | 0.902 / 0.805 |
| 2클래스, 시드10% | 99 / 697 / 199 | 0.819 → 0.839 | **+2.0%p** | 0.623 → 0.663 | 0.885 / 0.765 |
| 3클래스 | 236 / 1,025 / 315 | 0.825 → 0.835 | **+1.0%p** | 0.649 → 0.677 | 0.901 / 0.715 |
| 4클래스 전체 | 337 / 1,463 / 450 | 0.827 → 0.863 | **+3.6%p** | 0.649 → 0.680 | 0.871 / 0.726 |

- 4개 조건 전부 라운드1 > 라운드0 → 자동 라벨 확장 효과가 클래스 수·시드 비율에 걸쳐 일관 재현
- 시드 15%→10%(149→99장)로 줄여도 개선폭 유지 → 초기 라벨셋 규모를 줄여도 유효
- 자동 라벨 정밀도 0.87~0.90 유지 → 사람 개입 없이도 학습에 쓸 만한 품질

### 5.3 클래스별 mAP50-95 (4클래스, 라운드1)

| bolt | nut | gear | bearing |
|------|-----|------|---------|
| 0.710 | 0.751 | 0.570 | 0.690 |

- gear가 상대적으로 낮음(형태 다양성·유사 클래스 혼동). 클래스별 데이터 보강 대상.

### 5.4 반증(라벨 품질의 중요성)

- 자동 라벨과 이미지 파일명 매칭이 깨진 채 학습된 경우 mAP50 **−19.7%p** 폭락(전 객체를 배경으로 오학습). 라벨-이미지 정합성이 자동화 루프 성패의 핵심 변수임을 동일 조건에서 확인.
- Confusion Matrix, PR curve 등 상세 플롯은 학습 시 `runs/<name>/`에 자동 생성.

## 6. 설치 및 실행 방법

### 6.1 환경 설정

```bash
pip install -r requirements.txt
# GPU 사용 시 torch 는 CUDA 버전에 맞춰 별도 설치
#   RTX 5090(Blackwell): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   그 외:               pip install torch --index-url https://download.pytorch.org/whl/cu124
```

경로·임계값·하이퍼파라미터는 별도 `.env` 없이 `config.py` 한 곳에서 관리(`BASE_MODEL`, `AUTO_LABEL_CONF`, `IMG_SIZE`, `SERVE_MODEL` 등).

### 6.2 데이터 준비

```bash
# (검증용) Roboflow COCO 데이터를 YOLO 구조로 변환
python 0_coco_to_yolo.py --src ./mechanical-parts-coco --dst ./mechanical-parts-yolo
```

### 6.3 실행 순서 (자동화 루프)

```bash
# 1) 자동 라벨링: 미라벨 이미지 → conf≥0.6 YOLO 라벨
python 1_auto_labeling.py --weights models/base_model.pt

# 2) 학습 + ONNX 변환: train/val 분할 → 학습 → best.pt/onnx
python 2_train_pipeline.py --epochs 100 --device 0

# 3) 추론 API 서버
python 3_api_server.py       # http://0.0.0.0:8000
curl -X POST -F "file=@sample.jpg" "http://localhost:8000/predict?conf=0.3"
```

### 6.4 자동화 효과 재현 실험

```bash
# 정답을 숨기고 자동 채점: 라운드0 vs 라운드1 mAP + 자동라벨 P/R
python 4_experiment_autolearn.py --src ./mechanical-parts-yolo --classes bolt nut --epochs 60
# 결과: exp_autolearn/report.json (exp_results/ 에 조건별 리포트 보관)
```

API 응답 예시:

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

- `bbox` = 좌상단 `x,y` + `w,h`(픽셀). XR/프론트가 그대로 렌더 가능.

## 7. 한계점 및 추후 개선 과제

**한계점**

- **3D 렌더 데이터 연동 미적용**: 학습 데이터 소스인 3D 렌더 합성 데이터는 외부 공급 대상이며, 현재 리포는 라벨셋 확보 이후의 전처리·학습·자동라벨·평가·서빙 루프까지 다룬다. 실제 3D 렌더 데이터 연동은 미완.
- **도메인 갭**: 실제 대상 부품 3D 데이터 부재로 공개 기계부품 데이터로 대체 검증. 실제 부품·조명·배경, 렌더 이미지와 실사 간 sim-to-real 갭은 미검증.
- **검증 방식**: 고정 seed 단일 홀드아웃 분할. k-fold 교차검증 미적용으로 지표의 신뢰구간(분산) 미산출.
- **라운드 깊이**: 라운드1까지만 측정. 다회 라운드 누적 시 수렴·포화 지점 미검증.
- **자동 라벨 재현율**: 0.71~0.81 수준으로 conf 0.6에서 놓치는 객체 존재. 현재 파이프라인은 자동 라벨을 무검수로 학습에 투입.
- **데이터 규모**: 수백~천 장대. 실제 운영 규모(수만 장) 및 클래스 불균형(gear 저조) 미검증.
- **리소스 병목**: 멀티GPU(DDP) 학습 후 메모리 미회수로 대용량 조건 OOM 발생 → 단일 GPU로 우회(다중 GPU 스케일아웃 미확보).

**추후 개선 과제(Next steps)**

- 3D 렌더 합성 데이터 연동(입력 자동 수집 → 전처리 자동 트리거)으로 루프 완전 무인화 및 sim-to-real 갭 보정
- 실제 대상 부품 3D 데이터 확보 후 `data.yaml` 클래스 교체 및 재검증
- k-fold 교차검증 도입, 다회 라운드 누적 실험으로 수렴 곡선 확보
- conf 임계값·시드 비율 스윕으로 정밀도-재현율 트레이드오프 최적화
- 저성능 클래스(gear) 표적 데이터 증강 및 클래스 불균형 보정
- ONNX 서빙의 Unity/HMD 실기기 연동 및 지연·정확도 검증

---

**부록: 스크립트 구성**

| 파일 | 역할 |
|------|------|
| `config.py` | 경로·클래스·임계값·하이퍼파라미터 공통 설정 |
| `0_coco_to_yolo.py` | Roboflow COCO → YOLO 포맷 변환 |
| `0_import_roboflow.py` | Roboflow YOLOv8 export → 파이프라인 구조 정리 |
| `1_auto_labeling.py` | base_model로 자동 라벨링(conf 필터) |
| `2_train_pipeline.py` | train/val 분할 → 학습 → ONNX export |
| `3_api_server.py` | FastAPI 추론 서버 |
| `4_experiment_autolearn.py` | 자동화 효과 실증(정답 숨기고 자동 채점) |
