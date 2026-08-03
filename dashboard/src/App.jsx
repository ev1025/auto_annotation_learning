import { useEffect, useState, useCallback, useRef } from 'react'

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

// data/bell412/<그룹>/<부품>/videos → 그룹·부품 파싱. 폴더=부품, 폴더 안 영상=그 부품의 train/test 테이크
const groupOf = (rel) => (rel.split('/')[1] || rel)
const partOf = (rel) => { const p = rel.replace(/\/videos$/, '').split('/'); return p.length > 2 ? p.slice(2).join('/') : (p[1] || rel) }
const trailNum = (s) => { const m = String(s).match(/(\d+)\s*$/); return m ? +m[1] : 0 }
// 부품 폴더의 테이크 → {train:[...], test} : test* 있으면 그게 테스트, 없고 2개+면 끝수 최대("2")가 테스트, 단일이면 학습영상으로 테스트
const takeRoles = (videos) => {
  const names = (videos || []).map(v => v.name)
  if (!names.length) return { train: [], test: null }
  const explicit = names.filter(n => /test/i.test(n))
  if (explicit.length) return { train: names.filter(n => !/test/i.test(n)), test: explicit[explicit.length - 1] }
  if (names.length >= 2) {
    const sorted = [...names].sort((a, b) => trailNum(a) - trailNum(b))
    return { train: sorted.slice(0, -1), test: sorted[sorted.length - 1] }
  }
  return { train: names, test: names[0] }   // 단일 테이크 → 학습영상으로 테스트
}

// 부품 라벨링(SAM2): 데이터셋(그룹)→부품(폴더) 순회. 부품마다 학습 테이크 탭 → 입력 마스크 확인 → 라벨 생성 → 다음 부품 → 멀티클래스 학습
function AutoLabelView() {
  const [folders, setFolders] = useState([])        // [{folder,label,videos:[{name,count,ready}]}]
  const [partIdx, setPartIdx] = useState(0)         // 부품(폴더) 인덱스
  const [takeIdx, setTakeIdx] = useState(0)         // 부품 안 학습 테이크 인덱스(보통 0)
  const [count, setCount] = useState(0)             // 현재 테이크 프레임 수
  const [idx, setIdx] = useState(0)                 // 현재 프레임
  const [ptsBySrc, setPtsBySrc] = useState(() => {  // { 영상: { 프레임: [{rx,ry,lab}] } } 영상별 보관
    try { return JSON.parse(localStorage.getItem('autolabel_shots_v1') || '{}') } catch { return {} }
  })
  const [preparing, setPreparing] = useState(false)
  const [prepProg, setPrepProg] = useState('')      // 전체 프레임 미리 컷 진행표시
  const [masks, setMasks] = useState({})            // {"영상:프레임": {combo,area_frac,bbox}}
  const [activeShot, setActiveShot] = useState(null)// 크게 보고 있는 마스크 참조샷 키
  const [maskBusy, setMaskBusy] = useState(false)

  const [session, setSession] = useState(() => localStorage.getItem('parts_session_v1') || null)
  const [labeledMap, setLabeledMap] = useState({})  // {영상: {labels,frames}} 현재 세션에 라벨된 테이크
  const [labelJob, setLabelJob] = useState(null)
  const [labelStatus, setLabelStatus] = useState(null)

  const [trainJob, setTrainJob] = useState(null)
  const [trainStatus, setTrainStatus] = useState(null)
  const [excluded, setExcluded] = useState([])      // 학습에서 뺀 부품(클래스). 기본=라벨된 것 전부 포함
  const preppedRef = useRef(new Set())              // 그룹별 '전체 프레임 미리 컷' 1회만

  const running = !!labelStatus?.running || !!trainStatus?.running

  const isPartFolder = (rel) => rel.replace(/\/videos$/, '').split('/').length >= 3   // bell412/<컨테이너>/<부품> = 중첩 = 부품
  const nestedParts = folders.filter(f => isPartFolder(f.folder))
  const partFolders = nestedParts.length ? nestedParts : folders          // 부품 폴더들(중첩 없으면 전체)
  const curPartFolder = partFolders[partIdx]
  const folderVideos = curPartFolder?.videos || []                       // 이 부품의 테이크들
  const roles = takeRoles(folderVideos)
  const src = roles.train[takeIdx] || roles.train[0] || null             // 지금 탭할 학습 테이크
  const testTake = roles.test                                            // 이 부품 테스트 테이크(2 또는 학습영상 자체)
  const testIsSelf = testTake && roles.train.includes(testTake)

  const markReady = (name, cnt) => setFolders(prev => prev.map(f => ({
    ...f, videos: f.videos.map(v => v.name === name ? { ...v, ready: true, count: cnt } : v)
  })))
  const prepareSrc = async (name) => {   // 프레임 미컷이면 컷
    setPreparing(true)
    const d = await fetch(`/api/autolabel/prepare?src=${encodeURIComponent(name)}`).then(r => r.json())
    setPreparing(false); setCount(d.count || 0); markReady(name, d.count || 0)
  }
  const prepareAll = async (pfs) => {   // 그룹 전체 프레임 미리 컷(탭 전에 자동, 백그라운드)
    for (const pf of pfs) {
      for (const v of pf.videos) {
        if (v.ready) continue
        setPrepProg(`프레임 미리 컷: ${v.name}`)
        try { const d = await fetch(`/api/autolabel/prepare?src=${encodeURIComponent(v.name)}`).then(r => r.json()); markReady(v.name, d.count || 0) } catch { /* skip */ }
      }
    }
    setPrepProg('')
  }

  const loadFolders = useCallback(() => {
    fetch('/api/autolabel/folders').then(r => r.json()).then(d => setFolders(d))
  }, [])
  const loadSessions = useCallback(() => {
    fetch('/api/sam2/parts_sessions').then(r => r.json()).then(list => {
      const saved = localStorage.getItem('parts_session_v1')
      const found = list.find(s => s.session === saved)   // 내가 쓰던 세션만 복원(자동탭 baseline 등 남의 세션 자동 채택 안 함)
      if (found) { setSession(found.session); setLabeledMap(found.videos || {}) }
    }).catch(() => {})
  }, [])
  useEffect(() => { loadFolders() }, [loadFolders])
  useEffect(() => { loadSessions() }, [loadSessions])

  useEffect(() => {   // 폴더 로드되면 부품 전체 프레임 미리 컷(1회, 백그라운드) — 탭 전에 자동 준비
    if (!folders.length || preppedRef.current.has('all')) return
    preppedRef.current.add('all')
    const nf = folders.filter(f => f.folder.replace(/\/videos$/, '').split('/').length >= 3)
    prepareAll(nf.length ? nf : folders)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [folders])

  useEffect(() => {   // 부품(테이크) 바뀌면 그 영상 프레임 준비
    if (!src) { setCount(0); return }
    setIdx(0); setActiveShot(null); setLabelStatus(null); setLabelJob(null)
    const v = folderVideos.find(x => x.name === src)
    if (v?.ready) setCount(v.count || 0); else prepareSrc(src)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src])

  useEffect(() => {   // 참조샷 브라우저 저장(껐다 켜도 유지)
    try { localStorage.setItem('autolabel_shots_v1', JSON.stringify(ptsBySrc)) } catch { /* 용량초과 무시 */ }
  }, [ptsBySrc])

  const pts = ptsBySrc[src] || {}
  const cur = pts[idx] || []
  const shotKey = (v, i) => `${v}:${i}`
  const dropMask = (v, i) => setMasks(m => { const n = { ...m }; delete n[shotKey(v, i)]; return n })
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
  useEffect(() => {   // Ctrl+Z = 이 프레임 마지막 탭 취소
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z') && !running && cur.length) {
        e.preventDefault(); undo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, idx, cur.length, running])

  const buildShots = (name) => {   // 영상 참조샷 → [[프레임,[[rx,ry,lab],...]],...] (부품점 있는 프레임만)
    const vp = ptsBySrc[name] || {}
    return Object.keys(vp).map(Number).filter(i => (vp[i] || []).some(p => p.lab === 1)).sort((a, b) => a - b)
      .map(i => [i, vp[i].map(p => [p.rx, p.ry, p.lab])])
  }
  const hasTaps = (name) => Object.values(ptsBySrc[name] || {}).some(a => a.length)
  const isLabeled = (name) => !!labeledMap[name]
  const pfTrain = (pf) => takeRoles(pf.videos).train                     // 부품 폴더의 학습 테이크들
  const pfStatus = (pf) => {                                             // 부품 상태(학습 테이크 기준)
    const tr = pfTrain(pf)
    if (tr.some(isLabeled)) return 'done'
    if (tr.some(hasTaps)) return 'tapped'
    return 'todo'
  }
  const labeledParts = partFolders.filter(pf => pfTrain(pf).some(isLabeled)).map(pf => partOf(pf.folder))  // 라벨된 부품(클래스)명
  const nLabeled = labeledParts.length
  const selectedClasses = labeledParts.filter(p => !excluded.includes(p))   // 학습에 포함할 클래스
  const toggleExcluded = (p) => setExcluded(c => c.includes(p) ? c.filter(x => x !== p) : [...c, p])
  const curShots = buildShots(src)                 // 현재 학습 테이크의 유효 참조샷
  const shotFrames = Object.keys(pts).map(Number).filter(i => (pts[i] || []).length).sort((a, b) => a - b)  // 이 영상에서 탭한 프레임들
  const goShot = (i) => { setIdx(i); setActiveShot(shotKey(src, i)) }   // 그 프레임으로 이동 + (캐시 있으면) 마스크 표시
  const deleteShotFrame = (i) => {                  // 그 프레임 탭 삭제
    dropMask(src, i)
    setPtsBySrc(prev => { const vp = { ...(prev[src] || {}) }; delete vp[i]; return { ...prev, [src]: vp } })
    setActiveShot(c => c === shotKey(src, i) ? null : c)
  }

  // 현재 프레임 마스크 확인(가볍게, 단일 프레임) → 크게 표시
  const previewMask = async () => {
    if (!cur.length) return
    setMaskBusy(true)
    const d = await fetch('/api/sam2/mask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ src, frame: idx, points: cur.map(p => [p.rx, p.ry, p.lab]) })
    }).then(r => r.json()).catch(() => ({ error: '요청 실패' }))
    setMasks(m => ({ ...m, [shotKey(src, idx)]: d }))
    setActiveShot(shotKey(src, idx))
    setMaskBusy(false)
  }

  // 이 부품 라벨 생성: 참조샷 → SAM2 영상 전파 → 세션 폴더 누적
  const genLabel = async () => {
    if (!curShots.length) return
    const r = await fetch('/api/sam2/parts_label', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session, video: src, shots: curShots })
    }).then(x => x.json())
    if (r.error) { setLabelStatus({ error: r.error }); return }
    if (r.session) { setSession(r.session); localStorage.setItem('parts_session_v1', r.session) }
    setLabelJob(r.job); setLabelStatus({ stage: 'start', running: true, video: src })
  }
  useEffect(() => {
    if (!labelJob || !labelStatus?.running) return
    const t = setInterval(async () => {
      const d = await fetch(`/api/sam2/status?job=${labelJob}`).then(r => r.json())
      setLabelStatus(d)
      if (!d.running) {
        clearInterval(t)
        if (d.stage === 'done') {
          if (d.session) { setSession(d.session); localStorage.setItem('parts_session_v1', d.session) }
          setLabeledMap(m => ({ ...m, [d.video]: { labels: d.labels, frames: d.frames } }))
        }
      }
    }, 1200)
    return () => clearInterval(t)
  }, [labelJob, labelStatus?.running])

  const goPart = (i) => { setPartIdx(Math.max(0, Math.min(i, partFolders.length - 1))); setTakeIdx(0) }

  // 멀티클래스 학습: 세션 누적 라벨 → 클래스 통합 → YOLO 학습 → 부품별 테스트 테이크 검출 평가
  const testSrcs = () => {   // 라벨된 부품마다 테스트 테이크(끝수 "2" 또는 학습영상 자체)
    const s = []
    partFolders.forEach(pf => { const r = takeRoles(pf.videos); if (r.train.some(isLabeled) && r.test && !s.includes(r.test)) s.push(r.test) })
    return s
  }
  const runTrain = async () => {
    const tests = testSrcs().filter(t => {   // 선택 클래스에 해당하는 테스트 테이크만
      const pf = partFolders.find(pf => takeRoles(pf.videos).test === t)
      return pf && selectedClasses.includes(partOf(pf.folder))
    })
    const r = await fetch('/api/sam2/multiclass', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session, epochs: 100, test_srcs: tests, classes: selectedClasses })
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
  const trainStage = { start: '준비 중', build: '라벨 통합(클래스 매핑)', train: 'YOLO 학습 중', eval: 'test 영상 평가 중', done: '완료', error: '오류' }

  const activeMask = activeShot ? masks[activeShot] : null
  const genDone = labelStatus?.stage === 'done' && labelStatus?.video === src
  const partName = curPartFolder ? partOf(curPartFolder.folder) : ''
  const evalList = testSrcs()

  return (
    <div>
      <h2>부품 라벨링 (SAM2)</h2>

      <h3 className="section-h">부품 선택 <span className="al-hint" style={{ fontWeight: 400 }}>{nLabeled}/{partFolders.length}</span>
        {prepProg && <span className="al-hint" style={{ fontWeight: 400, marginLeft: 10 }}>⏳ {prepProg}</span>}
      </h3>

      {!src ? <p className="al-hint">부품 폴더가 없습니다. (data/bell412/parts/&lt;부품&gt;/videos)</p> : (
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
          {/* 부품 목록: 세로 스크롤 박스. 라벨된 부품은 체크박스로 학습 포함/제외 */}
          <div className="part-list">
            {partFolders.map((pf, i) => {
              const part = partOf(pf.folder)
              const labeled = pfTrain(pf).some(isLabeled)
              return (
                <div key={pf.folder} className={`part-item ${pfStatus(pf)} ${i === partIdx ? 'on' : ''}`}
                     onClick={() => !running && goPart(i)} title={part}>
                  {labeled && <input type="checkbox" className="part-ck" checked={!excluded.includes(part)}
                                     onClick={e => e.stopPropagation()} onChange={() => toggleExcluded(part)} disabled={running} />}
                  <span className="part-name">{part}</span>
                </div>
              )
            })}
          </div>

          {/* 오른쪽: 액션 + 이미지 + 마스크 + 프레임 */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="wiz-head">
              <span className="wiz-title">{partName}</span>
              <span className="al-hint">부품을 좌클릭(포함점)·우클릭(제외점) 후 <b>입력 마스크 확인</b> → 오른쪽에 마스크가 나옵니다.</span>
            </div>
            <div className="al-controls">
              <button className="al-secondary sm" onClick={() => goPart(partIdx - 1)} disabled={running || partIdx === 0}>◀ 이전</button>
              <button className="al-secondary sm" onClick={() => goPart(partIdx + 1)} disabled={running || partIdx >= partFolders.length - 1}>다음 ▶</button>
              <button className="cmp-random" onClick={previewMask} disabled={running || maskBusy || !cur.length}>
                {maskBusy ? '생성 중...' : '입력 마스크 확인'}
              </button>
              <button className="al-primary sm" onClick={genLabel} disabled={running || curShots.length === 0}>
                {labelStatus?.running ? '라벨 생성 중...' : (isLabeled(src) ? '↻ 라벨 다시 생성' : '라벨 생성')}
              </button>
              <button className="al-primary sm" style={{ background: '#0891b2' }} onClick={runTrain}
                      disabled={running || !session || selectedClasses.length === 0}>
                {trainStatus?.running ? '학습 중...' : `멀티클래스 학습 (${selectedClasses.length})`}
              </button>
              {labelStatus && !labelStatus.error && labelStatus.video === src &&
                <span className="al-hint">{labelStatus.running ? '전파 중...' : (labelStatus.stage === 'done' ? `✓ 라벨 ${labelStatus.labels}장` : '')}</span>}
            </div>

            {/* 탭 이미지 · 입력 마스크 결과 (가로로 나란히, 같은 크기) */}
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', flexWrap: 'nowrap', overflowX: 'auto' }}>
              {preparing
                ? <div className="al-frame" style={{ display: 'inline-block', padding: 30, textAlign: 'center', cursor: 'default' }}>
                    <span className="al-hint" style={{ color: '#e2e8f0' }}>프레임 컷 중...</span>
                  </div>
                : <div className="al-frame" style={{ display: 'inline-block', flex: '0 0 auto', width: 'auto', maxWidth: '100%' }}
                       onClick={(e) => addPoint(e, 1)} onContextMenu={(e) => addPoint(e, 0)}>
                    {src && <img src={`/api/autolabel/frame?src=${encodeURIComponent(src)}&idx=${idx}&w=720`} alt={`frame ${idx}`} draggable={false}
                                 style={{ width: 'auto', maxHeight: 360, maxWidth: 330 }} />}
                    {cur.map((p, i) => (
                      <span key={i} className={`al-dot ${p.lab === 1 ? 'pos' : 'neg'}`}
                            style={{ left: `${p.rx * 100}%`, top: `${p.ry * 100}%` }} />
                    ))}
                  </div>}
              {activeMask && (activeMask.error
                ? <p className="fn" style={{ color: '#b91c1c' }}>마스크 오류: {activeMask.error}</p>
                : <div className="al-maskbig">
                    <div className="al-maskbig-cap">
                      입력 마스크 · <span className="mk-g">초록=마스크</span> <span className="mk-o">주황=박스</span> <span className="mk-b">파랑=포함점</span> <span className="mk-r">빨강=제외점</span>
                      {typeof activeMask.area_frac === 'number' && <> · 면적 {(activeMask.area_frac * 100).toFixed(1)}%</>}
                    </div>
                    <img src={activeMask.combo} alt="입력 마스크" />
                  </div>)}
            </div>

            {/* 프레임 이동 */}
            <div className="al-controls">
              <button className="chip" onClick={() => setIdx(i => Math.max(i - 10, 0))} disabled={running}>◀◀10</button>
              <button className="chip" onClick={() => setIdx(i => Math.max(i - 1, 0))} disabled={running}>◀</button>
              <input className="al-slider" type="range" min={0} max={Math.max(count - 1, 0)} value={idx}
                     onChange={(e) => setIdx(+e.target.value)} disabled={running} />
              <span className="al-hint" style={{ minWidth: 54, textAlign: 'center' }}>{idx + 1}/{count}</span>
              <button className="chip" onClick={() => setIdx(i => Math.min(i + 1, count - 1))} disabled={running}>▶</button>
              <button className="chip" onClick={() => setIdx(i => Math.min(i + 10, count - 1))} disabled={running}>10▶▶</button>
              <button className="chip" onClick={undo} disabled={running || !cur.length}>점 취소</button>
              <button className="chip" onClick={clearFrame} disabled={running || !cur.length}>지우기</button>
            </div>

            {/* 이 영상에서 탭한 프레임(참조샷): 클릭하면 그 프레임으로 이동 확인, ×로 삭제 */}
            {shotFrames.length > 0 && (
              <div className="al-shots">
                <span className="al-hint" style={{ marginRight: 4 }}>참조샷:</span>
                {shotFrames.map(i => {
                  const npos = (pts[i] || []).filter(p => p.lab === 1).length
                  const nneg = (pts[i] || []).length - npos
                  const done = !!masks[shotKey(src, i)] && !masks[shotKey(src, i)].error
                  return (
                    <span key={i} className="al-shot-wrap">
                      <button className={`al-shot ${idx === i ? 'on' : ''} ${npos >= 1 ? '' : 'bad'}`}
                              onClick={() => goShot(i)} disabled={running}>
                        {done ? '✓ ' : ''}#{i} +{npos}{nneg ? `/-${nneg}` : ''}
                      </button>
                      <span className="al-shot-x" title="이 프레임 탭 삭제" onClick={() => !running && deleteShotFrame(i)}>×</span>
                    </span>
                  )
                })}
              </div>
            )}
            {labelStatus?.error && <p className="fn" style={{ color: '#b91c1c' }}>오류: {labelStatus.error}</p>}
            {genDone && labelStatus.taps?.length > 0 &&
              <div className="al-thumbs">{labelStatus.taps.map((u, i) => <img key={i} src={u} alt={`tap ${i}`} />)}</div>}
          </div>
        </div>
      )}

      {/* 멀티클래스 학습 결과 (버튼은 위 라벨생성 옆) */}
      <>
        {trainStatus?.error && <p className="fn" style={{ color: '#b91c1c' }}>학습 오류: {trainStatus.error}</p>}
        {trainStatus && !trainStatus.error && (
          <div className="al-result">
            <div className="al-controls">
              <b>{trainStage[trainStatus.stage] || trainStatus.stage}</b>
              {trainStatus.note && <span className="al-hint">{trainStatus.note}</span>}
              {trainStatus.n_images && <span className="al-hint">통합 <b>{trainStatus.n_images}</b>장 / {trainStatus.n_classes}클래스</span>}
              {trainStatus.stage === 'eval' && <span className="al-hint">평가 {trainStatus.eval_done || 0} / {trainStatus.eval_total}</span>}
            </div>
            {trainStatus.stage === 'done' && (
              <p className="al-hint">모델 <code>results/parts/{trainStatus.session}/multiclass/model/</code></p>
            )}
            {trainStatus.eval?.length > 0 && (
              <>
                <table><thead><tr>
                  <th>영상(학습=평가)</th><th>프레임</th><th>검출</th><th>검출률</th><th>평균 신뢰도</th><th>주요 클래스</th>
                </tr></thead><tbody>
                  {trainStatus.eval.map(e => (
                    <tr key={e.src}><td>{e.src}</td><td>{e.frames}</td><td>{e.detected}</td>
                      <td><b>{Math.round(e.rate * 100)}%</b></td><td>{e.mean_conf}</td>
                      <td>{(e.top_classes || []).map(([c, n]) => `${c}(${n})`).join(', ')}</td></tr>
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
      </>
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
                  onClick={() => setView('autolabel')}>부품 라벨링 (SAM2 탭)</button>
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
