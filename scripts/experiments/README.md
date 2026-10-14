# 실험 스크립트 목록 (manifest)

이 폴더는 **지금도 쓰는 스크립트만** 둔다. 지난 실험의 결과는 `docs/서버학습기록_xr_autolearning.md` 에 정리했고,
원본 `summary.txt` 는 서버 B 레포 `results/`(로컬은 git 밖 사본)에 있다. 스크립트 자체는 역할이 끝나면 지운다(2026-08-19 정리).

## 유지

| 파일 | 역할 | 산출물 |
|------|------|--------|
| `point_ref_lib.py` | 포인트 참조 라벨링 **공용 함수**(load_img·embed·candidates·nms·write_label + SAM/DINO 상수). `backend/autolearning/autolabel.py` 가 import 한다. **삭제 금지(의존)** | - |
| `build_multiclass.py` | 클래스 목록 로드·영상→부품 매핑 등 다중클래스 학습 공용 헬퍼. `sam2_autolabel.py` 가 import 한다. **삭제 금지(의존)** | - |
| `train_2class_server.py` | 서버(RTX5090)에서 오토라벨 결과로 학습 → 토르에 넣을 `best.pt` + `meta.json` 생성. 대시보드와 같은 재료·같은 증강을 쓴다 | `results/experiments/train_multi/<시각>/` |
| `compare_models_gt.py` | 여러 가중치를 **같은 GT**로 재서 비교(클래스별 AP·mAP50·mAP50-95). 학습 로그의 mAP 는 train=val 이라 포화되므로 모델 비교는 이 스크립트로만 판단한다 | `results/experiments/compare_gt/<시각>/` |
| `bench_infer.py` | 추론 성능(단건 지연 ms · 연속 FPS)을 PyTorch-GPU/CPU·ONNX 로 측정. 규격의 "FPS 고려" 항목 근거 | `results/bench/<시각>/` |
| `gt_viewer.py` | 사람 GT 70장에 실험별 예측 박스를 겹쳐 본다. `dump` 가 `scripts/gt_preds.js` 를 만들고 화면은 레포 루트 `gt_viewer.html` | `scripts/gt_preds.js` |
| `gt_bootstrap.py` | GT 맞춘 장수 차이에 부트스트랩 95% 신뢰구간을 붙인다 | - |
| `imgsz_probe.py` | 재학습 없이 추론 해상도만 바꿔 GT 70장 영향을 본다 | - |
| `hdr_tonemap/`, `server_bench/` | 톤매핑 비교, 서버에서 돌린 학습 · 데이터셋 스크립트 보관(각 폴더 README) | - |

상위 폴더의 `scripts/eval_gt.py` 도 같은 성격(현재 서비스 모델의 GT mAP 측정)이다.

## 지워진 것 (결과는 `docs/서버학습기록_xr_autolearning.md`, 원본은 서버 `results/`)

| 지운 스크립트 | 무엇을 확인했나 | 남은 근거 |
|---|---|---|
| `ablation_atest.py` · `ablation_synth.py` | 라벨 장수별 성능, 배경 합성 증강 효과 | `results/experiments/ablation_labels/`, `ablation_labels_synth/` |
| `ablation_refshots.py` | 참조샷 개수별 전파 커버리지·mAP | `results/experiments/ablation_refshots/`, `ablation_refshots_synth/` |
| `ablation_2class.py` · `ablation_2class_synth.py` | 2클래스 통합 학습 비교 | `results/ablation/gearbox_atest/`, `results/ablation_synth/gearbox_atest/` |
| `ablation_cuts.py` · `ablation_light.py` | 누끼 개수, 조명·데이터 개수 원인 분리 | 같은 폴더의 summary.txt |
| `synth_aug.py` | 위 ablation 전용 합성 헬퍼(운영은 `sam2_autolabel._synth_augment`) | - |
| `benchmark.py` · `benchmark_followup.py` · `experiment_autolearn.py` · `data_import/` | 공개 데이터셋(`data/robo`) 기반 벤치마크·오토러닝 실증 | git 이력 `ff58895^:results/dashboard/`(서버 B `results/dashboard/` 와 같은 파일) |

## 알아둘 것

- 배경 합성 증강의 **운영 구현은 하나뿐**이다: `backend/autolearning/sam2_autolabel._synth_augment()`.
  실험용 사본(`synth_aug.py`)을 두면 규칙이 두 벌로 갈라져 몰래 어긋난다.
- 모델 비교·제출 수치는 **반드시 GT 기준**으로 낸다. 학습 로그의 mAP 는 `train`·`val` 이 같은 폴더라
  0.99 로 포화된다(2026-08-19 실측: 학습셋 0.995 vs GT 0.885).
