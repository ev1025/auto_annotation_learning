# archive

지금 앱·배포에서 쓰지 않는 옛 코드를 동결 보관한다(2026-09-26 이동). 경로가 바뀌어 그대로는 실행되지 않는다.

| 경로 | 원래 위치 | 무엇 |
|---|---|---|
| `exp_dashboard/` | 레포 루트 | 실험 결과 대시보드(React). 방법 비교·실험 목록 화면 |
| `backend/dashboard_api_exp.py` | `backend/autolearning/` | 위 화면의 API 서버 (`/api/methods` · `/api/experiments` 등) |
| `backend/dashboard_core.py`, `dashboard_content.yaml` | `backend/autolearning/` | 위 서버의 데이터·문구 |
| `previews/` | `frontend/previews/` | 위 화면이 쓰던 방법별 예시 이미지 (`/previews/...`) |

되살리려면 원래 위치로 `git mv` 한다.
