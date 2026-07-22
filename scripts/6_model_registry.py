"""6_model_registry.py - 모델 릴리스 조회/롤백.

오토러닝이 지속적으로 새 모델을 배포하므로, 문제가 생겼을 때
"직전에 잘 되던 버전"으로 즉시 되돌릴 수단이 필요하다.
2_train_pipeline.py 가 학습마다 models/releases/v<타임스탬프>/ 에
(best.pt / best.onnx / metrics.json / train.log / results.csv) 를 보관하고,
이 스크립트가 그 보관본을 조회·복원한다.

실행:
  python scripts/6_model_registry.py list                 # 릴리스 목록·지표·상태
  python scripts/6_model_registry.py rollback             # 직전 '채택(promoted)' 버전으로 복원
  python scripts/6_model_registry.py rollback v20260716_093012   # 특정 버전으로 복원
"""
import argparse
import json
import shutil

import config


def load_releases():
    """릴리스 목록을 (최신순) [{version, status, map50, ...}, ...] 로 반환."""
    out = []
    if not config.RELEASES_DIR.exists():
        return out
    for d in sorted(config.RELEASES_DIR.iterdir(), reverse=True):
        mfile = d / "metrics.json"
        if d.is_dir() and mfile.exists():
            m = json.loads(mfile.read_text(encoding="utf-8"))
            m["_dir"] = d
            out.append(m)
    return out


def cmd_list():
    rels = load_releases()
    if not rels:
        print("보관된 릴리스가 없습니다. (2_train_pipeline.py 실행 시 자동 생성)")
        return
    # 현재 서빙 중인 모델과 크기 비교로 현재 버전 추정 표시
    cur_size = config.NEW_MODEL_PT.stat().st_size if config.NEW_MODEL_PT.exists() else -1
    print(f"{'버전':<20}{'학습 일시':<18}{'상태':<10}{'mAP50':>8}{'mAP50-95':>10}{'epochs':>8}  비고")
    for m in rels:
        mark = ""
        best = m["_dir"] / "best.pt"
        if best.exists() and best.stat().st_size == cur_size:
            mark = "<- 현재 서빙 중(추정)"
        when = str(m.get('trained_at', ''))[:16].replace('T', ' ')
        print(f"{m.get('version',''):<20}{when:<18}{m.get('status','?'):<10}"
              f"{m.get('map50','-'):>8}{m.get('map50_95','-'):>10}"
              f"{m.get('epochs','-'):>8}  {mark}")


def cmd_rollback(version):
    rels = load_releases()
    if not rels:
        raise SystemExit("[오류] 보관된 릴리스가 없어 롤백할 수 없습니다.")

    if version:
        target = next((m for m in rels if m.get("version") == version), None)
        if target is None:
            raise SystemExit(f"[오류] 버전을 찾지 못했습니다: {version}\n"
                             f"      보유: {[m.get('version') for m in rels]}")
    else:
        # 버전 미지정 시: 가장 최근 '채택본'을 건너뛰고 그 이전 채택본 = "직전에 잘 되던 버전"
        promoted = [m for m in rels if m.get("status") == "promoted"]
        if len(promoted) < 2:
            raise SystemExit("[오류] 되돌아갈 이전 채택본이 없습니다. 버전을 직접 지정하세요.")
        target = promoted[1]

    d = target["_dir"]
    if not (d / "best.pt").exists():
        raise SystemExit(f"[오류] {d} 에 best.pt 가 없습니다.")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(d / "best.pt", config.NEW_MODEL_PT)
    msg = [f"[롤백] {target['version']} -> {config.NEW_MODEL_PT}"]
    if (d / "best.onnx").exists():
        shutil.copy2(d / "best.onnx", config.NEW_MODEL_ONNX)
        msg.append(f"[롤백] ONNX 도 복원 -> {config.NEW_MODEL_ONNX}")
    print("\n".join(msg))
    print("API 서버(3_api_server.py)를 재시작하면 복원된 모델로 서빙됩니다.")


def main():
    ap = argparse.ArgumentParser(description="모델 릴리스 조회/롤백")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="릴리스 목록·지표·상태 출력")
    rb = sub.add_parser("rollback", help="이전 릴리스로 서빙 모델 복원")
    rb.add_argument("version", nargs="?", default=None,
                    help="복원할 버전(미지정 시 직전 채택본)")
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    else:
        cmd_rollback(args.version)


if __name__ == "__main__":
    main()
