"""GPU 별 점유 프로세스와 사용자를 표로 낸다."""
import subprocess
q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                    "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
uuid2idx = {}
for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines():
    idx, uuid, used, total = [x.strip() for x in line.split(",")]
    uuid2idx[uuid] = (idx, used, total)
rows = []
for line in q.splitlines() if q else []:
    uuid, pid, mem = [x.strip() for x in line.split(",")]
    ps = subprocess.run(["ps", "-o", "user=,etime=,cmd=", "-p", pid], capture_output=True, text=True).stdout.strip()
    user, etime, cmd = (ps.split(None, 2) + ["", "", ""])[:3] if ps else ("?", "?", "(종료됨)")
    rows.append((uuid2idx.get(uuid, ("?",))[0], pid, mem, user, etime, cmd[:70]))
print(f"{'GPU':<5}{'메모리':>12}  전체")
for uuid, (idx, used, total) in sorted(uuid2idx.items(), key=lambda x: x[1][0]):
    print(f"{idx:<5}{used:>12}  / {total}")
print()
print(f"{'GPU':<5}{'PID':<10}{'메모리':>10}  {'사용자':<12}{'경과':<12}명령")
for r in sorted(rows):
    print(f"{r[0]:<5}{r[1]:<10}{r[2]:>10}  {r[3]:<12}{r[4]:<12}{r[5]}")
if not rows:
    print("  점유 프로세스 없음")
