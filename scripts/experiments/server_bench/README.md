# server_bench — 5090 서버에서 돌린 벤치·데이터셋 구축 스크립트 (2026-08)

서버(`203.247.46.194`, RTX 5090) 홈 폴더에 흩어져 있던 것을 레포로 옮긴 것이다.
지금 서비스 중인 모델(`260824_retrain3x/yolo11m`)의 학습셋과 레시피가 이 스크립트들로 만들어졌다.
코드가 레포 밖에 있으면 "이 모델이 무엇으로 어떻게 학습됐나"를 나중에 되짚을 수 없어서 옮겼다.

## 경로 규약 주의

전부 서버 절대경로(`/workspace/data2/jinwoolee/...`)를 그대로 쓴다. 로컬·토르에서는 돌지 않는다.
다시 돌릴 일이 생기면 경로를 먼저 고친다.

## 무엇이 무엇인가

| 파일 | 하는 일 |
|---|---|
| `build_3x.py` | 실촬 대비 3배 합성 학습셋 생성. 서비스 모델이 쓴 레시피 |
| `build_occl8.py` | 판넬 가림을 넣은 학습셋(`benchset_occl8`) 생성. 배포 모델이 실제로 읽은 데이터 |
| `build_panel400.py` | 판넬 가림 400장 버전 |
| `build_gearbox1.py` | 기어박스 단일 부품 실험용 |
| `add_occl_mode.py` | `run_bench.py` 에 `occl` 레시피(mixup 0.2 + erasing 0.5) 모드를 붙인다 |
| `run_bench.py` | 모델·레시피를 인자로 받아 학습을 돌리는 실행기 |
| `bench_all.py`, `bench_cmp.py`, `bench_untrained.py` | 모델 계열별 벤치 실행·비교·미학습 기준선 |
| `chain_*.sh`, `queue_gpu*.sh`, `split_gpus.sh`, `dedup_queue.sh` | 여러 학습을 GPU 별로 줄 세워 돌리던 셸 |
| `kill_bench.sh`, `kill_rest_chain.sh` | 위 줄을 끊는 셸 |
| `gt40.py`, `eval_test4.py` | 사람이 그린 GT 40장으로 재는 평가(구간 3·4 확인) |
| `bg_gather.py`, `bg_inv.py`, `surion_bg.py` | 배경(네거티브) 사진 수집·점검 |
| `move_eval.py`, `move_synth.py`, `synth_share.py`, `split_office.py`, `split_tail.py` | 데이터 이동·분할 |
| `probe_model.py`, `gpu_who.py` | 가중치 내용 확인, GPU 점유자 확인 |
| `pack_run.py` | run 폴더를 압축해 내려받기 |
| `server_stems.txt`, `test4_result.json` | 당시 영상 목록과 test4 평가 결과 |

## 남겨 둔 데이터

`benchset_occl8` 만 서버에 남겼다. 배포 모델의 `args.yaml` 이 그 절대경로를 기록하고 있어
옮기면 추적이 끊긴다. 나머지 benchset 8종과 중간 산출물은 2026-09-01 에 지웠다
(기록: 서버 `_cleanup_260901.txt`).
