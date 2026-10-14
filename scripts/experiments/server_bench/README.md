# server_bench: 5090 서버에서 돌린 벤치 · 데이터셋 · 학습 스크립트 (2026-08 ~ 09)

서버 B(RTX 5090) 홈 폴더 · 임시 학습 폴더에 흩어져 있던 것을 레포로 옮긴 것이다.
이전 서비스 모델(`260824_retrain3x/yolo11m`)과 지금 서비스 모델(`260907_181414`, 2026-09-08 적용)의 학습셋 · 레시피가 이 스크립트들로 만들어졌다.
코드가 레포 밖에 있으면 "이 모델이 무엇으로 어떻게 학습됐나"를 나중에 되짚을 수 없어서 옮겼다.

## 경로 규약 주의

전부 서버 절대경로(`/workspace/data2/jinwoolee/...`)를 그대로 쓴다. 로컬·토르에서는 돌지 않는다.
다시 돌릴 일이 생기면 경로를 먼저 고친다.

## 무엇이 무엇인가

| 파일 | 하는 일 |
|---|---|
| `build_3x.py` | 실촬 대비 3배 합성 학습셋 생성. 이전 서비스 모델(260824_retrain3x)이 쓴 레시피 |
| `build_occl8.py` | 판넬 가림을 넣은 학습셋(`benchset_occl8`) 생성. 이전 서비스 모델이 실제로 읽은 데이터 |
| `build_panel400.py` | 판넬 가림 400장 버전 |
| `build_gearbox1.py` | 기어박스 단일 부품 실험용 |
| `add_occl_mode.py` | `run_bench.py` 에 `occl` 레시피(mixup 0.2 + erasing 0.5) 모드를 붙인다 |
| `run_bench.py` | 모델·레시피를 인자로 받아 학습을 돌리는 실행기 |
| `bench_all.py`, `bench_cmp.py`, `bench_untrained.py` | 모델 계열별 벤치 실행·비교·미학습 기준선 |
| `chain_*.sh`, `queue_gpu*.sh`, `split_gpus.sh`, `dedup_queue.sh` | 여러 학습을 GPU 별로 줄 세워 돌리던 셸 |
| `kill_bench.sh`, `kill_rest_chain.sh` | 위 줄을 끊는 셸 |
| `gt40.py`, `gt40_new.py`, `eval_test4.py` | 사람이 그린 GT 40장으로 재는 평가(구간 3·4 확인). `gt40_new.py` 는 첫 서버 다중 부품 판(260901_181828)을 채점했다(레포 루트 `gt_viewer.html`) |
| `bg_gather.py`, `bg_inv.py`, `surion_bg.py` | 배경(네거티브) 사진 수집·점검 |
| `move_eval.py`, `move_synth.py`, `synth_share.py`, `split_office.py`, `split_tail.py` | 데이터 이동·분할 |
| `probe_model.py`, `gpu_who.py` | 가중치 내용 확인, GPU 점유자 확인 |
| `pack_run.py` | run 폴더를 압축해 내려받기 |
| `server_stems.txt`, `test4_result.json` | 당시 영상 목록과 test4 평가 결과 |

## 남겨 둔 데이터

`benchset_occl8` 만 서버에 남겼다. 이전 서비스 모델의 `args.yaml` 이 그 절대경로를 기록하고 있어
옮기면 추적이 끊긴다. 나머지 benchset 8종과 중간 산출물은 2026-09-01 에 지웠다
(그때 기록 `_cleanup_260901.txt` 는 2026-10-14 서버 정리 때 지워졌다).

## xr_dash_260901 다중 부품 학습 (2026-09-01 ~ 09-08)

서버 B 임시 학습 폴더 `xr_dash_260901` 에서 지금 서비스 모델(260907_181414, 47종)을 만든 스크립트다.
판별 기록 · 계보 · 톤매핑 비교는 `docs/서버학습기록_xr_dash_260901.md` 에 있다. 폴더는 2026-10-14 에 지웠다.

| 파일 | 하는 일 |
|---|---|
| `cut_frames.py` | 라벨이 가리키는 프레임을 서버에서 영상으로 다시 뽑는다 |
| `srv_relabel.py` | 참조샷 기준 SAM2 라벨 재생성. 실행하면 이전 `labels` · `boxs` 를 먼저 지운다 |
| `run_train_parts.py` | 다중 부품 학습 실행기(`XR_AUTOLABELS` · `XR_EPOCHS` · `XR_EXCLUDE` · `XR_EXPECT`) |
| `tm_frames.sh`, `chain_tm.sh` | 톤매핑 학습 프레임 생성, 변환 없는 판 다음에 톤매핑 판을 이어서 학습 |
| `_srv_rename.py`, `_map.json` | 서버 폴더 · part_codes · GT 부품명을 47종 이름으로 개명(09-07), 옛 이름 → 새 이름 표 |

## 2026-08-29 도메인갭 실험

서버 레포(`xr_autolearning`)에 커밋 안 된 채로만 있던 것을 옮겼다. 원래 자리가 `scripts/experiments/` 라서
기본 `BASE = parents[2]` 가 여기서는 한 칸 어긋난다. 돌릴 땐 `XR_BASE=<레포 루트>` 를 준다.
기준셋 `benchset_3x` 는 09-01 에 지워져서 `build_variants.py` 는 그대로는 안 돈다.

| 파일 | 하는 일 |
|---|---|
| `diag_domaingap.py` | 기체 장착 구간(test3 · 4)에서 무너지는 원인이 분류 간섭인지 검출 실패인지 가른다 |
| `diag_scale.py` | 학습셋과 평가 구간의 객체 크기 분포 비교(합성은 화면 폭 18 ~ 42%, test4 gearbox 는 10% 남짓) |
| `eval_gt70.py` | 사람 GT 70장(gearbox 40 + a_test 30) 표준 채점기 |
| `build_variants.py` | 기준셋에서 한 가지만 바꾼 셋 만들기: real(합성 제거) · merged(1클래스 병합) · fewshot(기체 장착 10장 + 리플레이) |
| `run_exp.py`, `queue_260829.sh` | 실험 1건 = 학습 + GT 70장 채점, 순서대로 돌리는 큐(Plan A 퓨샷 · 진단 3 실촬만 · Plan B 1클래스 병합) |
| `eqlv2_loss.py`, `queue_eqlv2.sh` | Plan C: 분류 손실을 EQLv2 로 바꿔 눌린 클래스를 살리는 실험 |
