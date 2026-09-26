# 실험 기록 대시보드 (exp_dashboard)

지금까지 시도한 오토라벨 방법 11가지와 그 결과를 한 화면에서 훑어보는 화면이다.
운영용 대시보드(`frontend/`)와 별개다. 운영용은 "부품 학습 데이터 생성"만 남긴 lean 버전이고,
이쪽은 **실험 기록 · 시도 방법 · 용어**까지 들어 있는 원본이다.

## 왜 따로 있나

2026-08-12 에 운영 화면을 lean 으로 줄이면서 이 프론트를 아카이브로 뺐고,
2026-08-14 커밋 `337a81a` "죽은 코드 정리" 에서 그 백엔드 라우트까지 지웠다.
그래서 한동안 프론트만 남고 켤 수 없는 상태였다. 2026-08-26 에 되살렸다.

**운영 백엔드(`backend/autolearning/dashboard_api.py`)는 건드리지 않았다.**
지운 코드를 다시 섞어 넣는 대신 `dashboard_api_exp.py` 라는 별도 파일로 띄운다.

## 켜는 법

백엔드와 프론트를 따로 띄운다. 프론트(vite)가 `/api` 를 백엔드로 넘긴다.

```bash
# 1) 백엔드 (7862). vite.config.js 의 프록시 대상이 7862 라 포트를 바꾸면 거기도 같이 고칠 것
DASH_PORT=7862 XR_DB_READS=1 ./venv/Scripts/python.exe backend/autolearning/dashboard_api_exp.py

# 2) 프론트 (1234)
cd exp_dashboard
npm install          # 최초 1회
npx vite --port 1234 --host 127.0.0.1
```

접속: `http://127.0.0.1:1234`

DB 컨테이너(`xr_db`)가 떠 있어야 한다. 조회를 전부 DB 로 하기 때문에 없으면 화면이 503 만 뜬다.

```bash
docker compose --env-file deploy/.env -f deploy/docker-compose.dev.yml up -d
```

## 구성 파일

| 파일 | 역할 |
|---|---|
`src/App.jsx` | 화면 전체(부품학습 + 실험 + 시도방법 + 용어) |
`backend/autolearning/dashboard_api_exp.py` | 복원한 백엔드. 실험·용어 라우트 8개가 여기 있다 |
`backend/autolearning/dashboard_core.py` | 실험 결과 집계·비교 로직 |
`backend/autolearning/dashboard_content.yaml` | 시도 방법 11가지 설명·용어 사전 원문 |

## 지금 안 되는 것

| 기능 | 상태 | 이유 |
|---|---|---|
`HTML 리포트 내보내기` (`POST /api/export`) | 500 | `build_report.py` 가 같이 지워졌고 git 이력에도 없다 |
추론 비교 (`GET /api/compare`) | 이미지 없음 | `data/robo/yolo/test/images` 가 비어 있다. bearing·bolt·gear·nut 테스트셋이 로컬에 없다 |

실험 기록 · 시도 방법 · 용어 탭은 정상 동작한다.
