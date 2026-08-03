import { useEffect, useState, useCallback } from 'react'

// **굵게** 마커를 <b>로 변환
function md(text) {
  const html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

function Badge({ kind, label }) {
  return <span className={`badge ${kind}`}>{label}</span>
}

// 요약: 줄바꿈(\n)이 있으면 개조식 불릿, 없으면 한 문단
function Summary({ text }) {
  if (!text) return null
  const lines = String(text).split('\n').filter(Boolean)
  if (lines.length <= 1) return <p>{md(text)}</p>
  return <ul>{lines.map((l, i) => <li key={i}>{md(l)}</li>)}</ul>
}

function MetricsTable({ metrics }) {
  if (!metrics) return null
  return (
    <table>
      <thead><tr>{metrics.headers.map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
      <tbody>
        {metrics.rows.map((r, i) => (
          <tr key={i}>{r.map((v, j) => <td key={j}>{v}</td>)}</tr>
        ))}
      </tbody>
    </table>
  )
}

function FlowDiagram({ steps }) {
  return (
    <div className="flow">
      {steps.map((s, i) => (
        <div key={i}>
          {i > 0 && <div className="flow-arrow">↓</div>}
          <div className="flow-node">
            <span className="flow-step">{s.step}</span>
            <span className="flow-text">{md(s.text)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function SubTables({ tables }) {
  if (!tables?.length) return null
  return tables.map((t, i) => (
    <div className="subtable" key={i}>
      <h4 className="subtable-title">{t.title}</h4>
      <MetricsTable metrics={{ headers: t.headers, rows: t.rows }} />
    </div>
  ))
}

// 평가셋 전체 탐색 비교 (방법 1)
function CompareBrowser() {
  const [data, setData] = useState(null)
  const [idx, setIdx] = useState(0)
  const [total, setTotal] = useState(1)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async (i) => {
    setLoading(true)
    const res = await fetch(`/api/compare?idx=${i}`)
    const d = await res.json()
    setLoading(false)
    if (d.error) { setData(d); return }
    setData(d); setIdx(d.idx); setTotal(d.total)
  }, [])

  useEffect(() => { load(0) }, [load])

  if (data?.error) return <p className="fn">{data.error}</p>
  if (!data) return <p className="fn">불러오는 중...</p>

  return (
    <div className="cmp">
      <div className="cmp-bar">
        <div className="cmp-nav">
          <button onClick={() => load(idx - 1)} disabled={loading} aria-label="이전">‹</button>
          <span className="cmp-count">{idx + 1} <span className="cmp-total">/ {total}</span></span>
          <button onClick={() => load(idx + 1)} disabled={loading} aria-label="다음">›</button>
        </div>
        <button className="cmp-random" onClick={() => load(Math.floor(Math.random() * total))} disabled={loading}>
          무작위 이미지
        </button>
        <span className="cmp-file">{data.file}</span>
      </div>

      <div className="pair" style={loading ? { opacity: 0.5 } : {}}>
        <figure className="pane">
          <figcaption>모델 예측 <span className="cap-sub">Conf &gt; 0.6</span></figcaption>
          <img src={data.pred} alt="모델 예측" />
        </figure>
        <figure className="pane">
          <figcaption>정답 라벨 <span className="cap-sub">사람 라벨</span></figcaption>
          <img src={data.gt} alt="정답 라벨" />
        </figure>
      </div>

      {data.legend?.length > 0 && (
        <div className="legend">
          {data.legend.map((g, i) => {
            const c = data.counts?.find(x => x.name === g.name)
            return (
              <span className="legend-item" key={i}>
                <span className="legend-chip" style={{ background: g.color }} />
                {g.name}{c ? <> <b>{c.pred}/{c.gt}</b></> : ''}
              </span>
            )
          })}
          <span className="legend-note">(탐지 / 정답 개수)</span>
        </div>
      )}

      <ul className="cmp-info">
        <li><b>파일명</b>: {data.file}</li>
        <li><b>현재 모델</b>: 임시 2클래스 (bolt·nut 전용)</li>
        <li><b>알림</b>: 베어링(bearing), 기어(gear)는 탐지 불가. 서버 복구 후 5클래스 모델로 대체 예정.</li>
      </ul>
    </div>
  )
}

function MethodView({ id }) {
  const [m, setM] = useState(null)
  useEffect(() => {
    setM(null)
    fetch(`/api/method/${id}`).then(r => r.json()).then(setM)
  }, [id])

  if (!m) return <p className="fn">불러오는 중...</p>

  return (
    <div>
      <h2>{m.title} <Badge kind={m.badge} label={m.badge_label} /></h2>

      {m.subtitle && <p className="method-desc">{m.subtitle}</p>}

      {m.tech?.length > 0 && (<>
        <h3 className="section-h">사용 기술</h3>
        <div className="tech-list">
          {m.tech.map((t, i) => (
            <div className="tech-row" key={i}>
              <span className="tech-name">{t.name}</span>
              <span className="tech-desc">
                {t.desc}{t.usage && <span className="tech-usage"> · {t.usage}</span>}
              </span>
            </div>
          ))}
        </div>
      </>)}

      <h3 className="section-h">실험 순서</h3>
      {m.flow?.length
        ? <FlowDiagram steps={m.flow} />
        : m.ordered
          ? <ol>{m.bullets.map((b, i) => <li key={i}>{md(b)}</li>)}</ol>
          : <ul>{m.bullets.map((b, i) => <li key={i}>{md(b)}</li>)}</ul>}

      {m.code?.length > 0 && <>
        <h3 className="section-h">실제 코드</h3>
        {m.code.map((sn, i) => (
          <div key={i}>
            <p className="snip-note">{sn.note}</p>
            <div className="snippet">
              <div className="snip-head"><code className="snip-file">{sn.file}</code></div>
              <pre><code>{sn.src}</code></pre>
            </div>
          </div>
        ))}
      </>}

      <h3 className="section-h">실제 입출력 결과</h3>
      {m.live && <CompareBrowser />}
      {m.gallery.length > 0 && (
        <div className="gallery">
          {m.gallery.map((g, i) => (
            <figure key={i}>
              <img src={g.url} alt={g.caption} loading="lazy" />
              <figcaption>{g.caption}</figcaption>
            </figure>
          ))}
        </div>
      )}
      {!m.live && m.gallery.length === 0 && (
        <p className="fn">저장된 비교 이미지 없음 (서버 유실로 미보존) - 아래 지표로 확인</p>
      )}

      <h3 className="section-h">실험 결과</h3>
      <Summary text={m.metrics.summary} />
      <MetricsTable metrics={m.metrics} />
      <SubTables tables={m.metrics.subtables} />

      {m.extras?.map((ex, i) => (
        <div key={i}>
          <h3 className="section-h">{ex.title}</h3>
          {ex.desc && <p>{ex.desc}</p>}
          <MetricsTable metrics={ex.table} />
        </div>
      ))}
    </div>
  )
}

// 모델 선정(벤치마크) — 라벨링 이전 단계라 방법과 분리해 맨 위에 둠
function BenchmarkView() {
  const [cats, setCats] = useState({})        // {카테고리: [토픽,...]}
  const [cat, setCat] = useState(null)
  const [topic, setTopic] = useState(null)
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    fetch('/api/experiments').then(r => r.json()).then(d => {
      setCats(d)
      const c = Object.keys(d)[0]
      setCat(c); setTopic((d[c] || [])[0])
    })
  }, [])

  useEffect(() => {
    if (!cat || !topic) return
    setMetrics(null)
    fetch(`/api/experiment?cat=${encodeURIComponent(cat)}&topic=${encodeURIComponent(topic)}`)
      .then(r => r.json()).then(setMetrics)
  }, [cat, topic])

  const catNames = Object.keys(cats)
  const topics = cats[cat] || []
  return (
    <div>
      <h2>실험 기록</h2>
      <p className="method-desc">모델·입력크기 선정(벤치마크)과 도메인갭 극복(밤샘 스윕·copy-paste·수동 GT 실측 mAP) 기록.</p>
      {catNames.length > 1 && (
        <div className="chips" style={{ marginBottom: 6 }}>
          {catNames.map(c => (
            <button key={c} className={c === cat ? 'chip on' : 'chip'}
                    onClick={() => { setCat(c); setTopic((cats[c] || [])[0]) }}
                    style={{ fontWeight: 600 }}>{c}</button>
          ))}
        </div>
      )}
      <div className="chips">
        {topics.map(t => (
          <button key={t} className={t === topic ? 'chip on' : 'chip'}
                  onClick={() => setTopic(t)}>{t}</button>
        ))}
      </div>
      {metrics && <>
        {metrics.desc && <p className="method-desc">{metrics.desc}</p>}
        <Summary text={metrics.summary} />
        <MetricsTable metrics={metrics} />
        <SubTables tables={metrics.subtables} />
      </>}
    </div>
  )
}

// 포인트 참조 오토라벨: 프레임 넘기며 클릭 → 자동 라벨 생성
function AutoLabelView() {
  const [folders, setFolders] = useState([])        // [{folder,label,videos:[{name,count,ready}]}]
  const [folder, setFolder] = useState(null)        // 선택한 폴더(data 기준 rel 경로)
  const [src, setSrc] = useState(null)              // 폴더 안에서 지금 탭 중인 영상
  const [count, setCount] = useState(0)
  const [idx, setIdx] = useState(0)
  const [ptsBySrc, setPtsBySrc] = useState(() => {  // { 영상: { 프레임: [{rx,ry,lab}] } } 영상별 보관(폴더 넘어 누적)
    try { return JSON.parse(localStorage.getItem('autolabel_shots_v1') || '{}') } catch { return {} }
  })
  const [preparing, setPreparing] = useState(false)
  const [masks, setMasks] = useState({})            // {"영상:프레임": {pts_img,mask_img,box_img,area_frac,bbox}}
  const [activeShot, setActiveShot] = useState(null)// 지금 이미지 보고 있는 참조샷 키
  const [maskBusy, setMaskBusy] = useState(false)
  const [maskProg, setMaskProg] = useState('')
  const [propJob, setPropJob] = useState(null)
  const [propStatus, setPropStatus] = useState(null)
  const [trainJob, setTrainJob] = useState(null)
  const [trainStatus, setTrainStatus] = useState(null)
  const [labeled, setLabeled] = useState({})        // {영상: [실행들]} 라벨 생성된 영상
  const [trainSel, setTrainSel] = useState([])      // 학습에 쓸 영상 이름들(폴더 넘어)
  const [evalSel, setEvalSel] = useState([])        // 평가할 영상 이름들
  const running = !!propStatus?.running || !!trainStatus?.running

  const loadLabeled = useCallback(() => {
    fetch('/api/sam2/labeled').then(r => r.json()).then(setLabeled).catch(() => {})
  }, [])

  const markReady = (name, cnt) => setFolders(prev => prev.map(f => ({
    ...f, videos: f.videos.map(v => v.name === name ? { ...v, ready: true, count: cnt } : v)
  })))

  const prepareSrc = async (name, ready) => {   // 프레임 미컷이면 컷
    if (ready) return
    setPreparing(true)
    const d = await fetch(`/api/autolabel/prepare?src=${encodeURIComponent(name)}`).then(r => r.json())
    setPreparing(false); setCount(d.count || 0); markReady(name, d.count || 0)
  }

  const pickSrc = (fo, name) => {   // 폴더 fo 안의 영상 name 을 탭 대상으로
    const v = fo?.videos.find(x => x.name === name)
    setSrc(name); setCount(v?.count || 0); setIdx(0); setPropStatus(null); setPropJob(null)
    prepareSrc(name, v?.ready)
  }
  const chooseSrc = (name) => pickSrc(folders.find(f => f.folder === folder), name)

  const nameSel = (fo, kw) => fo.videos.filter(v => v.name.toLowerCase().includes(kw)).map(v => v.name)
  const applyFolder = (fo) => {   // 폴더 선택 시 첫 영상 준비 + 학습/평가 기본선택(이름 기준, 이 폴더 안만)
    setTrainSel(nameSel(fo, 'train')); setEvalSel(nameSel(fo, 'test'))
    const first = fo.videos.find(v => !v.name.toLowerCase().startsWith('test')) || fo.videos[0]
    if (first) pickSrc(fo, first.name); else setSrc(null)
  }
  const chooseFolder = (rel) => {
    const fo = folders.find(f => f.folder === rel)
    if (!fo) return
    setFolder(rel); setTrainStatus(null); setTrainJob(null); applyFolder(fo)
  }

  const loadFolders = useCallback(() => {
    fetch('/api/autolabel/folders').then(r => r.json()).then(d => {
      setFolders(d)
      setFolder(cur => {
        if (cur || !d[0]) return cur
        const fo = d[0]
        setTrainSel(fo.videos.filter(v => v.name.toLowerCase().includes('train')).map(v => v.name))
        setEvalSel(fo.videos.filter(v => v.name.toLowerCase().includes('test')).map(v => v.name))
        const first = fo.videos.find(v => !v.name.toLowerCase().startsWith('test')) || fo.videos[0]
        if (first) { setSrc(first.name); setCount(first.count || 0) }
        return fo.folder
      })
    })
  }, [])
  useEffect(() => { loadFolders() }, [loadFolders])
  useEffect(() => { loadLabeled() }, [loadLabeled])

  useEffect(() => {   // 참조샷을 브라우저에 저장 (껐다 켜도 유지)
    try { localStorage.setItem('autolabel_shots_v1', JSON.stringify(ptsBySrc)) } catch { /* 용량초과 무시 */ }
  }, [ptsBySrc])

  const pts = ptsBySrc[src] || {}            // 현재 영상의 점들
  const cur = pts[idx] || []
  const shotKey = (v, i) => `${v}:${i}`
  const dropMask = (v, i) => setMasks(m => { const n = { ...m }; delete n[shotKey(v, i)]; return n })   // 점 바뀐 샷 캐시 무효
  const updateCur = (fn) => setPtsBySrc(prev => ({ ...prev, [src]: fn(prev[src] || {}) }))
  const addPoint = (e, lab) => {
    e.preventDefault()
    if (running) return
    const rect = e.currentTarget.getBoundingClientRect()
    const rx = +((e.clientX - rect.left) / rect.width).toFixed(4)
    const ry = +((e.clientY - rect.top) / rect.height).toFixed(4)
    if (rx < 0 || rx > 1 || ry < 0 || ry > 1) return
    dropMask(src, idx)
    updateCur(c => ({ ...c, [idx]: [...(c[idx] || []), { rx, ry, lab }] }))
  }
  const undo = () => { dropMask(src, idx); updateCur(c => ({ ...c, [idx]: (c[idx] || []).slice(0, -1) })) }
  const clearFrame = () => { dropMask(src, idx); updateCur(c => { const n = { ...c }; delete n[idx]; return n }) }
  const clearAllForSrc = () => {
    setMasks(m => Object.fromEntries(Object.entries(m).filter(([k]) => !k.startsWith(`${src}:`))))
    updateCur(() => ({}))
  }
  useEffect(() => {   // Ctrl+Z(맥 Cmd+Z) = 이 프레임 마지막 탭 취소
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z') && !running && cur.length) {
        e.preventDefault(); undo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [src, idx, cur.length, running])

  const shotFrames = Object.keys(pts).map(Number).filter(i => (pts[i] || []).length).sort((a, b) => a - b)
  const validShots = shotFrames.filter(i => (pts[i] || []).some(p => p.lab === 1))

  // (1) 마스크 확인: 이 폴더의 모든 참조샷(부품 점 있는 프레임)을 SAM2로 한 번에 만들어 캐시
  const confirmMasks = async () => {
    const jobs = []
    for (const v of tappedVideos) {
      const vp = ptsBySrc[v.name] || {}
      for (const i of Object.keys(vp).map(Number)) {
        const p = vp[i] || []
        if (p.some(x => x.lab === 1)) jobs.push({ v: v.name, i, points: p.map(x => [x.rx, x.ry, x.lab]) })
      }
    }
    if (!jobs.length) return
    setMaskBusy(true)
    let done = 0
    for (const jb of jobs) {
      setMaskProg(`마스크 생성 중 ${done + 1}/${jobs.length} (${jb.v} #${jb.i})`)
      const d = await fetch('/api/sam2/mask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ src: jb.v, frame: jb.i, points: jb.points })
      }).then(r => r.json()).catch(() => ({ error: '요청 실패' }))
      setMasks(m => ({ ...m, [shotKey(jb.v, jb.i)]: d }))
      done++
      if (!activeShot) setActiveShot(shotKey(jb.v, jb.i))
    }
    setMaskBusy(false); setMaskProg('')
  }

  // (2) 라벨 생성: 이 영상 참조샷 → SAM2 영상 전파
  const runPropagate = async () => {
    const shots = validShots.map(i => [i, pts[i].map(p => [p.rx, p.ry, p.lab])])
    const d = await fetch('/api/sam2/propagate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ src, shots })
    }).then(r => r.json())
    if (d.error) { setPropStatus({ error: d.error }); return }
    setPropJob(d.job); setPropStatus({ stage: 'start', running: true, done: 0, total: count })
  }
  useEffect(() => {
    if (!propJob || !propStatus?.running) return
    const t = setInterval(async () => {
      const d = await fetch(`/api/sam2/status?job=${propJob}`).then(r => r.json())
      setPropStatus(d)
      if (!d.running) { clearInterval(t); loadLabeled() }
    }, 1000)
    return () => clearInterval(t)
  }, [propJob, propStatus?.running])

  const propStage = { start: '준비 중', load: 'SAM2 로딩 중', propagate: '영상 전파 중', done: '완료', error: '오류' }
  const trainStage = { start: '준비 중', propagate: '라벨 생성 중', train: 'YOLO 학습 중', eval: 'test 영상 평가 중', done: '완료', error: '오류' }
  const propDone = propStatus?.stage === 'done'
  const folderObj = folders.find(f => f.folder === folder)
  const folderVideos = folderObj?.videos || []
  const hasTaps = (name) => Object.values(ptsBySrc[name] || {}).some(a => a.length)
  const tappedVideos = folderVideos.filter(v => hasTaps(v.name))                 // 이 폴더에서 점 찍은 영상
  // 학습/평가는 영상 단위(폴더 넘어 선택). 학습은 라벨 생성된 영상만.
  const trainRuns = trainSel.filter(n => labeled[n]?.length).map(n => ({ src: n, run: labeled[n][0].run }))
  const testSrcs = evalSel

  // (3) 학습+평가: 이 폴더 train 영상 라벨 합쳐 학습 → test 영상 평가
  const toggleTrain = (n) => setTrainSel(cur => cur.includes(n) ? cur.filter(x => x !== n) : [...cur, n])
  const toggleEval = (n) => setEvalSel(cur => cur.includes(n) ? cur.filter(x => x !== n) : [...cur, n])
  const goShot = (v, i) => { if (v !== src) chooseSrc(v); setIdx(i); setActiveShot(shotKey(v, i)) }
  const deleteShot = (v, i) => {                       // 참조샷(그 프레임 점) 개별 삭제
    setMasks(m => { const n = { ...m }; delete n[shotKey(v, i)]; return n })
    setPtsBySrc(prev => { const vp = { ...(prev[v] || {}) }; delete vp[i]; return { ...prev, [v]: vp } })
    setActiveShot(cur => cur === shotKey(v, i) ? null : cur)
  }
  const buildShots = (name) => {   // 영상의 참조샷 → [[프레임, [[rx,ry,lab],...]], ...] (부품점 있는 프레임만)
    const vp = ptsBySrc[name] || {}
    return Object.keys(vp).map(Number).filter(i => (vp[i] || []).some(p => p.lab === 1)).sort((a, b) => a - b)
      .map(i => [i, vp[i].map(p => [p.rx, p.ry, p.lab])])
  }
  const partName = (folderObj?.label || folder || '').split('/').pop()   // 부품명 = 폴더 끝 (예: gearbox)
  const trainReady = trainSel.filter(n => buildShots(n).length)          // 점 찍은 학습영상
  // 한 방에(부품 세션): train 영상 각자 전파→라벨 한 폴더 통합(results/<부품>/<시각>/) → 학습 → 평가
  const runSession = async () => {
    const train_shots = {}
    trainReady.forEach(n => { train_shots[n] = buildShots(n) })
    const r = await fetch('/api/sam2/session', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ part: partName, train_shots, test_srcs: evalSel })
    }).then(x => x.json())
    if (r.error) { setTrainStatus({ error: r.error }); return }
    setTrainJob(r.job); setTrainStatus({ stage: 'start', running: true })
  }
  useEffect(() => {
    if (!trainJob || !trainStatus?.running) return
    const t = setInterval(async () => {
      const d = await fetch(`/api/sam2/status?job=${trainJob}`).then(r => r.json())
      setTrainStatus(d)
      if (!d.running) clearInterval(t)
    }, 1500)
    return () => clearInterval(t)
  }, [trainJob, trainStatus?.running])

  return (
    <div>
      <h2>오토라벨 (SAM2 탭)</h2>

      <h3 className="section-h">1. 폴더 선택</h3>
      <div className="chips">
        {folders.map(f => (
          <button key={f.folder} className={f.folder === folder ? 'chip on' : 'chip'}
                  onClick={() => chooseFolder(f.folder)} disabled={running || preparing}>
            {f.label.split('/').pop()}
          </button>
        ))}
        <button className="chip" onClick={loadFolders} disabled={running || preparing}
                title="data 폴더 다시 스캔">↻ 새로고침</button>
      </div>

      <h3 className="section-h">2. 영상 선택</h3>
      <div className="chips">
        {folderVideos.map(v => (
          <button key={v.name} className={v.name === src ? 'chip on' : 'chip'}
                  onClick={() => chooseSrc(v.name)} disabled={running || preparing}>
            {v.name}
          </button>
        ))}
      </div>

      <h3 className="section-h">3. 객체 포인트 지정 {src && <span className="al-hint">— {src}</span>}</h3>
      {preparing
        ? <div className="al-frame" style={{ padding: 44, textAlign: 'center', cursor: 'default' }}>
            <span className="al-hint" style={{ color: '#e2e8f0' }}>프레임 컷 중... (영상 길이에 따라 몇 초~수십 초)</span>
          </div>
        : <div className="al-frame" onClick={(e) => addPoint(e, 1)} onContextMenu={(e) => addPoint(e, 0)}>
            {src && <img src={`/api/autolabel/frame?src=${encodeURIComponent(src)}&idx=${idx}&w=960`} alt={`frame ${idx}`} draggable={false} />}
            {cur.map((p, i) => (
              <span key={i} className={`al-dot ${p.lab === 1 ? 'pos' : 'neg'}`}
                    style={{ left: `${p.rx * 100}%`, top: `${p.ry * 100}%` }} />
            ))}
          </div>}

      <div className="al-controls">
        <button className="chip" onClick={() => setIdx(i => Math.max(i - 10, 0))} disabled={running}>◀◀ 10</button>
        <button className="chip" onClick={() => setIdx(i => Math.max(i - 1, 0))} disabled={running}>◀</button>
        <input className="al-slider" type="range" min={0} max={Math.max(count - 1, 0)} value={idx}
               onChange={(e) => setIdx(+e.target.value)} disabled={running} />
        <button className="chip" onClick={() => setIdx(i => Math.min(i + 1, count - 1))} disabled={running}>▶</button>
        <button className="chip" onClick={() => setIdx(i => Math.min(i + 10, count - 1))} disabled={running}>10 ▶▶</button>
        <span className="al-hint">frame {idx + 1} / {count} · 이 프레임 점 {cur.length}개</span>
      </div>
      <div className="al-controls">
        <button className="chip" onClick={undo} disabled={running || !cur.length}>점 취소</button>
        <button className="chip" onClick={clearFrame} disabled={running || !cur.length}>이 프레임 지우기</button>
      </div>

      <h3 className="section-h">4. 포인트별 마스크 확인</h3>
      <div className="al-controls">
        <button className="cmp-random" onClick={confirmMasks} disabled={running || maskBusy || tappedVideos.length === 0}>
          {maskBusy ? (maskProg || '마스크 생성 중...') : '마스크 확인'}
        </button>
      </div>
      {tappedVideos.map(v => {
        const vp = ptsBySrc[v.name] || {}
        const frames = Object.keys(vp).map(Number).filter(i => (vp[i] || []).length).sort((a, b) => a - b)
        return (
          <div key={v.name} style={{ marginBottom: 8 }}>
            <span className="al-hint">
              <b style={{ color: v.name === src ? '#6366f1' : undefined }}>{v.name}</b>
            </span>
            <div className="al-shots">
              {frames.map(i => {
                const npos = vp[i].filter(p => p.lab === 1).length
                const nneg = vp[i].length - npos
                const k = shotKey(v.name, i)
                const done = !!masks[k] && !masks[k].error
                return (
                  <span key={i} className="al-shot-wrap">
                    <button className={`al-shot ${activeShot === k ? 'on' : ''} ${npos >= 1 ? '' : 'bad'}`}
                            onClick={() => goShot(v.name, i)} disabled={running}>
                      {done ? '✓ ' : ''}#{i} <b>+{npos}</b>{nneg ? `/-${nneg}` : ''}
                    </button>
                    <span className="al-shot-x" title="이 참조샷 삭제"
                          onClick={() => !running && deleteShot(v.name, i)}>×</span>
                  </span>
                )
              })}
            </div>
          </div>
        )
      })}

      {activeShot && masks[activeShot] && (masks[activeShot].error
        ? <p className="fn" style={{ color: '#b91c1c' }}>마스크 오류({activeShot}): {masks[activeShot].error}</p>
        : (() => {
            const mk = masks[activeShot]
            return (
              <div className="al-result">
                <div className="al-controls">
                  <span className="al-hint">{activeShot}</span>
                </div>
                <div className="al-thumbs"><img src={mk.combo} alt="mask" style={{ maxWidth: 560 }} /></div>
              </div>
            )
          })())}

      <h3 className="section-h">5. 라벨 생성 및 훈련 학습 평가</h3>

      <h4 className="subtable-title">학습 영상</h4>
      <div className="chips">
        {folderVideos.map(v => {
          const isTrain = v.name.toLowerCase().includes('train')
          const on = trainSel.includes(v.name)
          const nTap = buildShots(v.name).length
          return (
            <button key={v.name} className={on ? 'chip on' : 'chip'} disabled={running || !isTrain}
                    onClick={() => toggleTrain(v.name)}>
              {isTrain ? (on ? '☑ ' : '☐ ') : ''}{v.name}
              {nTap > 0 ? <span className="al-hint"> 탭 {nTap}</span> : null}
            </button>
          )
        })}
      </div>

      <h4 className="subtable-title">평가 영상</h4>
      <div className="chips">
        {folderVideos.map(v => {
          const isTest = v.name.toLowerCase().includes('test')
          const on = evalSel.includes(v.name)
          return (
            <button key={v.name} className={on ? 'chip on' : 'chip'} disabled={running || !isTest}
                    onClick={() => toggleEval(v.name)}>
              {isTest ? (on ? '☑ ' : '☐ ') : ''}{v.name} <span className="al-hint">{v.ready ? `${v.count}컷` : `~${v.count}`}</span>
            </button>
          )
        })}
      </div>

      <div className="al-controls" style={{ marginTop: 10 }}>
        <button className="cmp-random" onClick={runSession}
                disabled={running || trainReady.length === 0 || evalSel.length === 0}>
          {trainStatus?.running ? '진행 중...' : '라벨 생성 + 학습 + 평가'}
        </button>
      </div>

      {trainStatus?.error && <p className="fn" style={{ color: '#b91c1c' }}>오류: {trainStatus.error}</p>}
      {trainStatus && !trainStatus.error && (
        <div className="al-result">
          <div className="al-controls">
            <b>{trainStage[trainStatus.stage] || trainStatus.stage}</b>
            {trainStatus.note && <span className="al-hint">{trainStatus.note}</span>}
            {trainStatus.train_labels && <span className="al-hint">학습 라벨 <b>{trainStatus.train_labels}</b>
              {trainStatus.train_srcs && <> ({trainStatus.train_srcs.join('+')})</>}</span>}
            {trainStatus.stage === 'eval' && <span className="al-hint">평가 {trainStatus.eval_done || 0} / {trainStatus.eval_total}</span>}
          </div>
          {trainStatus.stage === 'done' && trainStatus.run && (
            <p className="al-hint">라벨 <code>results/{trainStatus.part}/{trainStatus.run}/</code> · 모델 <code>results/model/{trainStatus.run}/</code></p>
          )}
          {trainStatus.eval?.length > 0 && (
            <>
              <table className="cmp-table"><thead><tr>
                <th>test 영상</th><th>프레임</th><th>검출</th><th>검출률</th><th>평균 신뢰도</th>
              </tr></thead><tbody>
                {trainStatus.eval.map(e => (
                  <tr key={e.src}><td>{e.src}</td><td>{e.frames}</td><td>{e.detected}</td>
                    <td><b>{Math.round(e.rate * 100)}%</b></td><td>{e.mean_conf}</td></tr>
                ))}
              </tbody></table>
              {trainStatus.eval.map(e => e.samples?.length > 0 && (
                <div key={e.src}>
                  <h4 className="subtable-title">{e.src}</h4>
                  <div className="al-thumbs">{e.samples.map((u, i) => <img key={i} src={u} alt={`${e.src} ${i}`} />)}</div>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}

function GlossaryView() {
  const [rows, setRows] = useState([])
  useEffect(() => { fetch('/api/glossary').then(r => r.json()).then(setRows) }, [])
  return (
    <div>
      <h2>용어 안내</h2>
      <MetricsTable metrics={{ headers: ['용어', '뜻'], rows }} />
    </div>
  )
}

export default function App() {
  const [methods, setMethods] = useState([])
  const [view, setView] = useState('autolabel')
  const [exportMsg, setExportMsg] = useState('')
  const [exporting, setExporting] = useState(false)

  useEffect(() => { fetch('/api/methods').then(r => r.json()).then(setMethods) }, [])

  const doExport = async () => {
    setExporting(true); setExportMsg('생성 중... (약 30초)')
    try {
      const res = await fetch('/api/export', { method: 'POST' })
      const d = await res.json()
      setExportMsg(`완료: ${d.path}`)
    } catch {
      setExportMsg('내보내기 실패')
    }
    setExporting(false)
  }

  const badgeText = { adopt: '채택', drop: '탈락', partial: '부분' }

  return (
    <div className="layout">
      <aside>
        <div className="brand">
          <h1>오토라벨링 검증</h1>
          <p>학습 라벨을 자동 생성하는<br />10가지 방법의 실험 기록</p>
        </div>
        <nav>
          <div className="nav-title">오토라벨</div>
          <button className={view === 'autolabel' ? 'nav-item on' : 'nav-item'}
                  onClick={() => setView('autolabel')}>오토라벨 (SAM2 탭)</button>
          <div className="nav-title">실험 기록</div>
          <button className={view === 'benchmark' ? 'nav-item on' : 'nav-item'}
                  onClick={() => setView('benchmark')}>실험 (벤치마크 · 도메인갭 mAP)</button>
          <div className="nav-title">시도 방법 (번호 = 시도 순서)</div>
          {methods.map(m => (
            <button key={m.id} className={view === m.id ? 'nav-item on' : 'nav-item'}
                    onClick={() => setView(m.id)}>
              <span>{m.no}. {m.title}</span>
              <span className={`mini-badge ${m.badge}`}>{badgeText[m.badge]}</span>
            </button>
          ))}
          <div className="nav-title">그 외</div>
          <button className={view === 'glossary' ? 'nav-item on' : 'nav-item'}
                  onClick={() => setView('glossary')}>용어 안내</button>
        </nav>
        <div className="export-box">
          <button className="export-btn" onClick={doExport} disabled={exporting}>
            HTML 리포트 내보내기
          </button>
          {exportMsg && <p className="fn">{exportMsg}</p>}
        </div>
      </aside>
      <main>
        <div className="card">
          {view === 'benchmark' ? <BenchmarkView />
            : view === 'glossary' ? <GlossaryView />
            : view === 'autolabel' ? <AutoLabelView />
            : <MethodView id={view} />}
        </div>
      </main>
    </div>
  )
}
