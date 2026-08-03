# 실험 스크립트 목록 (manifest)

이 폴더는 **실행 스크립트만** 둔다. 결과 데이터는 `results/`(결과 json)·`results/benchmark/`,
증거 이미지는 대시보드 갤러리(`dashboard/previews/`)에 있다.
자세한 경위·판단은 프로젝트 `README.md` 8장(특히 "2026-07-29~30") 참조.

## 유지 (역할별)

| 파일 | 무슨 실험 / 역할 | 산출물 |
|------|-----------------|--------|
| `point_ref_lib.py` | 포인트 참조 라벨링 **공용 함수**(load_img·embed·candidates·nms·write_label + SAM/DINO 상수). 원래 "기어박스 포인트참조+self-training 실험"이었으나 헬퍼가 여기 있어 `verify/autolabel.py`·`sam2_propagate.py` 등이 import함. **삭제 금지(의존)** | - |
| `experiment_autolearn.py` | 로보플로우 기계부품 오토러닝(자동라벨→학습→ONNX) 효과 실증 = 대시보드 **방법 1** | `results/exp_results/report_*.json` |
| `benchmark.py` | YOLO 14조합(모델 7종 × 입력크기 2종) 벤치마크 | `results/benchmark/benchmark.json` |
| `benchmark_followup.py` | 벤치마크 1위 재확인: epoch 연장·n 레시피·**같은 조건 시드 반복** → 1위가 시드(난수) 운 판명 (구 `exp_epochs.py`) | `results/benchmark/exp_epochs.json` |
| `sam2_propagate.py` | **SAM2 영상 전파 검증**: 탭 → 마스크 → 영상 전체 추적. 부품 전체를 일관되게 잡음 = 대시보드 **방법 10**, 현재 진행 방향 (구 `test_sam2.py`) | 검증용 마스크 이미지 |

## 삭제됨 (automask 계열 = 박스 품질 한계로 SAM2(방법 10)로 대체. 결과 json은 `results/`에 보존)

- `exp_variance.py` / `exp_variance_aug.py` / `exp_variance_abc.py` — 재현성(시드 5회)·회전 증강·참조 전략(A=1장4점/B=4장1점/C=4장4점) 비교
- `exp_overnight.py` — 밤샘 오케스트레이터(pseudo-GT mAP 재평가 + TTA + 새 영상). **자동 GT 불량으로 mAP 결과 폐기**
- `exp_selftrain_bell.py` — 위 절차를 BELL412 부품에 재현
- `exp_train_raw.py` — raw 영상 학습(중단)
- `_gen_m8_evidence.py` / `_test_on_video.py` — 일회성 증거·영상 유틸

## 왜 automask 계열을 접었나 (요약)

자동 라벨(SAM automask + DINOv2 매칭)이 복잡한 다중재질 부품을 한 덩어리로 못 잘라서,
학습 라벨도 채점용 pseudo-GT도 프레임마다 "기어만 / 화면 전체"로 들쭉날쭉했다.
presence-recall(프레임당 검출 여부)은 ~90%로 좋아 보였지만 실제 박스 품질은 낮았고 예측이 안 맞았다.
→ **SAM2 영상 전파**(`sam2_propagate.py`)로 탭 마스크를 추적하니 부품 전체를 일관되게 잡음. 이 방향으로 전환.
