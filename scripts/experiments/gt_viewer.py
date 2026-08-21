# -*- coding: utf-8 -*-
"""GT 사진에 정답·예측 박스를 겹쳐 브라우저에서 비교하는 뷰어.

두 단계로 쓴다. HTML 은 한 번만 쓰이고, 그 다음부터는 데이터만 갈아끼우면 된다.

  python scripts/experiments/gt_viewer.py dump   실험·모델별 예측 -> scripts/gt_preds.js
  python scripts/experiments/gt_viewer.py html   화면 파일(gt_viewer.html) 다시 쓰기(화면 고칠 때만)
  python scripts/experiments/gt_viewer.py share  외부 전달용 한 파일(사진·데이터 내장)

  화면은 레포 루트의 gt_viewer.html 을 열면 된다(file:// 로도 동작).
  사진은 복사하지 않고 data/bell412/<부품>/gt/images 원본을 그대로 참조한다.

새 실험이 생기면 아래 EXPS 에 한 줄 추가하고 dump 만 다시 돌린다(HTML 은 그대로).
preds.js 는 fetch 가 아니라 <script> 로 읽는다 -> file:// 로 열어도 동작한다.
"""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
GT_ROOT = BASE / "data" / "bell412"
CONF = 0.25              # 낮게 뽑아 두고 화면에서 슬라이더로 올린다

# 실험 = (클래스 수, 합성, 배경, 증강) 조합. base 는 {m} 에 모델명이 들어가는 가중치 경로.
EXPS = [
    {"id": "c36_syn_bg0", "label": "36클래스 · 합성 390 · 배경 없음",
     "classes": 36, "synth": 390, "bg": 0, "train": 8030,
     "base": "results/bench/260820_113527/{m}/weights/best.pt",
     "models": ["yolo11n", "yolo11s", "yolo11m", "yolo26s"]},
    {"id": "c36_syn_bg417", "label": "36클래스 · 합성 390 · 배경 417",
     "classes": 36, "synth": 390, "bg": 417, "train": 8447,
     "base": "results/bench/260820_191442_bg/{m}/weights/best.pt",
     "models": ["yolo11n", "yolo11s", "yolo11m", "yolo26s"]},
    {"id": "c36_bg417_aug", "label": "36클래스 · 배경 417 · 강한 증강",
     "classes": 36, "synth": 390, "bg": 417, "train": 8447,
     "base": "results/bench/260821_091951_aug/{m}/weights/best.pt",
     "models": ["yolo11n", "yolo11s", "yolo11m", "yolo26s"]},
    {"id": "c2_syn", "label": "2클래스 · 합성 462 · 배경 없음",
     "classes": 2, "synth": 462, "bg": 0, "train": 934,
     "base": "results/ablation_synth/gearbox_atest/20260810_093338/runs/model_{m}/weights/best.pt",
     "models": ["yolo11n", "yolo11s", "yolo11m", "yolo26n", "yolo26s", "yolo26m",
                "yolov8n", "yolov8s", "yolov8m"]},
    {"id": "c2_nosyn", "label": "2클래스 · 합성 없음 · 배경 없음",
     "classes": 2, "synth": 0, "bg": 0, "train": 472,
     "base": "results/ablation/gearbox_atest/20260806_205714/runs/model_{m}/weights/best.pt",
     "models": ["yolo11n", "yolo11s", "yolo11m", "yolo26n", "yolo26s", "yolo26m",
                "yolov8n", "yolov8s", "yolov8m"]},
]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def gt_images():
    """GT(사람이 그린 정답)가 있는 부품의 사진 목록. 라벨 없는 사진은 제외."""
    import cv2
    out = []
    for part in sorted(d.name for d in GT_ROOT.iterdir()
                       if (d / "gt" / "images").is_dir() and (d / "gt" / "labels").is_dir()):
        for ip in sorted((GT_ROOT / part / "gt" / "images").glob("*.jpg")):
            lp = GT_ROOT / part / "gt" / "labels" / f"{ip.stem}.txt"
            if not lp.exists():
                continue
            h, w = cv2.imread(str(ip)).shape[:2]
            gts = []
            for ln in lp.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if len(f) != 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in f[1:])
                gts.append([round((cx-bw/2)*w), round((cy-bh/2)*h),
                            round((cx+bw/2)*w), round((cy+bh/2)*h)])
            out.append({"part": part, "file": f"{part}__{ip.stem}.jpg",
                        "src": ip.relative_to(BASE).as_posix(),   # 원본 위치를 그대로 쓴다(복사 안 함)
                        "abs": str(ip), "w": w, "h": h, "gt": gts})
    return out


PREDS = BASE / "scripts" / "gt_preds.js"       # 예측 데이터(window.D). 화면이 <script> 로 읽는다
HTML_PATH = BASE / "gt_viewer.html"            # 화면. 레포 루트에 두고 그대로 쓴다


def dump(dst: Path):
    """실험·모델별 예측 -> preds.js (window.D 에 넣는다)."""
    from ultralytics import YOLO
    images = gt_images()
    print("GT 이미지", len(images), flush=True)
    paths = [im["abs"] for im in images]
    data = {"images": [{k: v for k, v in im.items() if k != "abs"} for im in images], "exps": []}
    for e in EXPS:
        ent = {k: e[k] for k in ("id", "label", "classes", "synth", "bg", "train")}
        ent["models"] = {}
        for m in e["models"]:
            w = BASE / e["base"].format(m=m)
            if not w.exists():
                print("없음", w, flush=True)
                continue
            mod = YOLO(str(w))
            res = []
            for k in range(0, len(paths), 16):
                res += list(mod.predict(source=paths[k:k+16], conf=CONF, imgsz=640,
                                        device=0, verbose=False))
            per = {}
            for im, r in zip(images, res):
                bx = [[round(float(b[0])), round(float(b[1])), round(float(b[2])), round(float(b[3])),
                       round(float(cf), 2), mod.names[int(cl)]]
                      for b, cf, cl in zip(r.boxes.xyxy.cpu().numpy(),
                                           r.boxes.conf.cpu().numpy(),
                                           r.boxes.cls.cpu().numpy())]
                best = max([iou(b[:4], g) for b in bx if b[5] == im["part"] for g in im["gt"]] or [0.0])
                per[im["file"]] = {"p": bx, "iou": round(best, 3)}
            ent["models"][m] = per
            hit = sum(1 for v in per.values() if v["iou"] >= 0.5)
            print("{:16} {:9} IoU>=0.5 {}/{}".format(e["id"], m, hit, len(per)), flush=True)
            del mod
        data["exps"].append(ent)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("window.D = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print("저장", dst, f"{dst.stat().st_size/1024:.0f} KB")


HTML = r"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>검출 비교 - GT vs 예측</title>
<style>
 :root{--bg:#0f1420;--panel:#161d2e;--line:#26304a;--ink:#e8ecf5;--muted:#8e9ab5;
       --gt:#22c55e;--ok:#f59e0b;--ng:#ef4444}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
   font:14px/1.55 -apple-system,"Malgun Gothic",sans-serif}
 header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);
   padding:12px 16px;display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center}
 select{background:#1e2740;color:var(--ink);border:1px solid var(--line);border-radius:8px;
   padding:7px 10px;font:inherit}
 select#exp{min-width:260px} select#model{min-width:130px}
 label{font:inherit;color:var(--muted);display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
 .row{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;width:100%}
 h2.sec{margin:26px 14px 10px;font-size:20px;font-weight:700;color:var(--ink)}
 h2.sec:first-child{margin-top:16px}
 .meta{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
 .legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px;padding:14px}
 figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
 .wrap{position:relative;background:#000;cursor:zoom-in}
 .wrap img{display:block;width:100%;height:auto}
 svg{position:absolute;inset:0;width:100%;height:100%}
 rect.box{fill:none;stroke-width:2.5}
 rect.chip{stroke:none}
 text{font-weight:700;font-family:sans-serif}
 figcaption{display:flex;justify-content:space-between;gap:8px;padding:7px 10px;font-size:12px;
   color:var(--muted);font-variant-numeric:tabular-nums}
 .iou{font-weight:700} .bad{color:var(--ng)} .mid{color:var(--ok)} .good{color:var(--gt)}
 dialog{border:0;padding:0;background:transparent;max-width:96vw;max-height:96vh}
 dialog::backdrop{background:rgba(0,0,0,.86)}
 dialog .wrap{cursor:zoom-out} dialog img{max-height:92vh;width:auto}
 table.sum{border-collapse:collapse;margin:8px 14px 10px;font-size:12px;font-variant-numeric:tabular-nums}
 table.sum th,table.sum td{border:1px solid var(--line);padding:5px 9px;text-align:right}
 table.sum th{background:var(--panel);color:var(--muted);vertical-align:bottom;font-weight:600;line-height:1.35}
 table.sum td:first-child,table.sum th:first-child{text-align:left}
 table.sum td.on{outline:2px solid var(--ok);outline-offset:-2px}
 table.sum td.best{background:#14361f;color:#c8f7d4;font-weight:700}
 table.sum td.na{color:#3d4762}
 table.sum .sub{color:var(--muted);font-weight:400}
</style>
<h2 class="sec">실험 결과 표</h2>
<table class="sum" id="sum"></table>
<h2 class="sec">실제 예측 박스</h2>
<header>
 <select id="exp"></select>
 <select id="model"></select>
 <select id="part"></select>
 <label>conf <input type="range" id="conf" min="0.25" max="0.95" step="0.05" value="0.25"> <b id="cv">0.25</b></label>
 <span class="meta" id="stat"></span>
 <div class="row meta legend">
  <span><i style="background:#22c55e"></i>GT</span>
  <span><i style="background:#f59e0b"></i>예측</span>
  <span><i style="background:#ef4444"></i>오검</span></div>
</header>
<div class="grid" id="grid"></div>
<dialog id="dlg"><div class="wrap" id="dwrap"></div></dialog>
<script src="scripts/gt_preds.js"></script>
<script>
const $ = i => document.getElementById(i);
$('exp').innerHTML = D.exps.map((e, i) => `<option value="${i}">${e.label}</option>`).join('');
// 부품 목록은 데이터에서 만든다(GT 가 늘어나면 자동으로 늘어난다)
const parts = [...new Set(D.images.map(im => im.part))].sort();
$('part').innerHTML = `<option value="">부품 전체(${D.images.length})</option>`
  + parts.map(p => `<option value="${p}">${p} (${D.images.filter(im => im.part === p).length})</option>`).join('');

function curExp() { return D.exps[+$('exp').value]; }
function fillModels() {
  const keep = $('model').value;
  const ms = Object.keys(curExp().models);
  $('model').innerHTML = ms.map(m => `<option>${m}</option>`).join('');
  if (ms.includes(keep)) $('model').value = keep;
}
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  if (x2 <= x1 || y2 <= y1) return 0;
  const i = (x2 - x1) * (y2 - y1);
  return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i);
}
function boxesFor(im, exp, model, conf) {
  const r = (exp.models[model] || {})[im.file] || {p: []};
  const p = r.p.filter(b => b[4] >= conf);
  let best = 0;                              // conf 를 올리면 IoU 도 다시 계산해야 맞다
  for (const b of p) if (b[5] === im.part) for (const g of im.gt) best = Math.max(best, iou(b, g));
  return {p, iou: best};
}
function svgFor(im, r) {
  const fz = Math.max(im.w, im.h) / 45;
  let s = `<svg viewBox="0 0 ${im.w} ${im.h}" preserveAspectRatio="xMidYMid meet">`;
  const chip = (x, y1, y2, txt, col) => {
    const ch = fz * 1.35;
    const cw = txt.length * fz * 0.58 + fz * 0.6;           // 어림(fitChips 가 실측으로 교체)
    const cx = Math.max(0, Math.min(x, im.w - cw));
    const cy = (y1 - ch >= 0) ? y1 - ch : y2;               // 위가 잘리면 박스 아래(박스 밖)
    return `<rect class="chip" data-x="${cx}" data-y1="${y1}" data-y2="${y2}"`
         + ` x="${cx}" y="${cy}" width="${cw}" height="${ch}" fill="${col}"/>`
         + `<text x="${cx + fz * 0.3}" y="${cy + ch * 0.75}" font-size="${fz}" fill="#fff" class="lbl">${txt}</text>`;
  };
  for (const g of im.gt)     // GT 는 테두리만(라벨 달면 예측 칩과 겹친다)
    s += `<rect class="box" x="${g[0]}" y="${g[1]}" width="${g[2]-g[0]}" height="${g[3]-g[1]}" stroke="#22c55e" vector-effect="non-scaling-stroke"/>`;
  for (const b of r.p) {
    const col = b[5] === im.part ? '#f59e0b' : '#ef4444';
    s += `<rect class="box" x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" stroke="${col}" vector-effect="non-scaling-stroke"/>`
       + chip(b[0], b[1], b[3], `${b[5]} ${b[4].toFixed(2)}`, col);
  }
  return s + '</svg>';
}
// 그린 뒤 (1) 칩 폭을 실제 글자폭으로 (2) 겹치면 박스 위 -> 박스 아래(바깥) -> 더 아래로 피한다.
// 박스 안쪽에는 넣지 않는다(부품을 가린다).
function fitChips(root) {
  for (const sv of root.querySelectorAll('svg')) {
    const [, , vbW, vbH] = sv.getAttribute('viewBox').split(' ').map(Number);
    const placed = [];
    for (const t of sv.querySelectorAll('text.lbl')) {
      const rc = t.previousElementSibling;
      if (!rc || rc.tagName !== 'rect') continue;
      const fz = parseFloat(t.getAttribute('font-size'));
      const h = parseFloat(rc.getAttribute('height'));
      const w = t.getBBox().width + fz * 0.6;
      const x = Math.max(0, Math.min(parseFloat(rc.dataset.x), vbW - w));
      const y1 = parseFloat(rc.dataset.y1), y2 = parseFloat(rc.dataset.y2);
      const hits = y => placed.some(q => x < q.x + q.w && x + w > q.x && y < q.y + q.h && y + h > q.y);
      const cands = [];
      if (y1 - h >= 0) cands.push(y1 - h);
      for (let k = 0; k < 6 && y2 + h * (k + 1) <= vbH; k++) cands.push(y2 + h * k);
      for (let k = 1; k < 6 && y1 - h * (k + 1) >= 0; k++) cands.push(y1 - h * (k + 1));
      const y = cands.find(c => !hits(c)) ?? (cands[0] ?? Math.max(0, y1 - h));
      rc.setAttribute('x', x); rc.setAttribute('y', y); rc.setAttribute('width', w);
      t.setAttribute('x', x + fz * 0.3); t.setAttribute('y', y + h * 0.75);
      placed.push({x, y, w, h});
    }
  }
}
function score(exp, model, part, conf) {
  const rows = D.images.filter(im => !part || im.part === part)
                       .map(im => boxesFor(im, exp, model, conf));
  const hit = rows.filter(r => r.iou >= 0.5).length;
  const zero = rows.filter(r => r.iou === 0).length;
  const avg = rows.reduce((s, r) => s + r.iou, 0) / (rows.length || 1);
  return {n: rows.length, hit, zero, avg};
}
function render() {
  const exp = curExp(), model = $('model').value, part = $('part').value;
  const conf = parseFloat($('conf').value);
  $('cv').textContent = conf.toFixed(2);
  const rows = D.images.filter(im => !part || im.part === part)
                       .map(im => ({im, r: boxesFor(im, exp, model, conf)}))
                       .sort((a, b) => a.im.file.localeCompare(b.im.file));   // 이름순
  const s = score(exp, model, part, conf);
  $('stat').innerHTML = `<b>${s.n}장 · IoU≥0.5 ${s.hit}장 · 미검출 ${s.zero}장 · 평균 IoU ${s.avg.toFixed(3)}</b>`;
  $('grid').innerHTML = rows.map(({im, r}) => {
    const cls = r.iou >= 0.7 ? 'good' : r.iou >= 0.5 ? 'mid' : 'bad';
    return `<figure><div class="wrap" data-file="${im.file}">`
         + `<img src="${im.src}" alt="${im.file}" loading="lazy">${svgFor(im, r)}</div>`
         + `<figcaption><span>${im.file.replace('__', ' / ').replace('.jpg', '')}</span>`
         + `<span class="iou ${cls}">IoU ${r.iou.toFixed(2)}</span></figcaption></figure>`;
  }).join('');
  fitChips($('grid'));

  // 표: 모델(행) x 실험(열). 칸 = 평균 IoU, 괄호 = 미검출 장수, 열 최고값 강조
  const allModels = [...new Set(D.exps.flatMap(e => Object.keys(e.models)))];
  let h = `<tr><th>모델</th>${D.exps.map(e => `<th>${e.label.replace(/ · /g, '<br>')}</th>`).join('')}</tr>`;
  const colBest = D.exps.map(e => Math.max(...allModels.filter(m => e.models[m])
      .map(m => score(e, m, part, conf).avg)));
  for (const m of allModels) {
    h += `<tr><td>${m}</td>`;
    D.exps.forEach((e, i) => {
      if (!e.models[m]) { h += `<td class="na">-</td>`; return; }
      const v = score(e, m, part, conf);
      const cls = [v.avg === colBest[i] ? 'best' : '', (e.id === exp.id && m === model) ? 'on' : ''].join(' ').trim();
      h += `<td class="${cls}">${v.avg.toFixed(3)} <span class="sub">(${v.zero})</span></td>`;
    });
    h += `</tr>`;
  }
  $('sum').innerHTML = h;
}
$('grid').addEventListener('click', ev => {
  const w = ev.target.closest('.wrap'); if (!w) return;
  const im = D.images.find(x => x.file === w.dataset.file);
  const r = boxesFor(im, curExp(), $('model').value, parseFloat($('conf').value));
  $('dwrap').innerHTML = `<img src="${im.src}">` + svgFor(im, r);
  fitChips($('dwrap'));
  $('dlg').showModal();
});
$('dlg').addEventListener('click', () => $('dlg').close());
$('exp').addEventListener('input', () => { fillModels(); render(); });
for (const id of ['model', 'part', 'conf']) $(id).addEventListener('input', render);
fillModels(); render();
</script>
</html>
"""


def share(dst: Path, long_side: int = 1024, q: int = 80):
    """외부 전달용 한 파일 HTML. 사진을 data URI 로, 예측을 <script> 로 함께 박아 넣는다.
    이 파일만 있으면 어디서든 열린다(레포·사진·js 불필요). 대신 용량이 커진다."""
    import base64
    import cv2
    data = json.loads(re.sub(r"^window\.D = |;\s*$", "", PREDS.read_text(encoding="utf-8").strip()))
    for im in data["images"]:
        img = cv2.imread(str(BASE / im["src"]))
        r = long_side / max(img.shape[:2])
        if r < 1:
            img = cv2.resize(img, (round(img.shape[1]*r), round(img.shape[0]*r)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        im["src"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    html = HTML.replace('<script src="scripts/gt_preds.js"></script>',
                        "<script>window.D = " + json.dumps(data, ensure_ascii=False) + ";</script>")
    dst.write_text(html, encoding="utf-8")
    print(f"외부용 한 파일: {dst} ({dst.stat().st_size/1048576:.1f} MB · 사진 {len(data['images'])}장 내장)")


def write_html():
    """화면 파일을 레포 루트에 쓴다. 화면을 고칠 때만 돌리면 된다(데이터는 dump 가 갱신)."""
    HTML_PATH.write_text(HTML, encoding="utf-8")
    print("화면:", HTML_PATH)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dump", "html", "share"):
        print(__doc__)
        raise SystemExit(1)
    if sys.argv[1] == "dump":
        dump(Path(sys.argv[2]) if len(sys.argv) > 2 else PREDS)
    elif sys.argv[1] == "share":
        share(Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "gt_viewer_공유.html")
    else:
        write_html()
