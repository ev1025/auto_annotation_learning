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

// 부품 라벨링(SAM2): 장비(부품)마다 탭 → 입력 마스크 확인 → 라벨 생성 → 다음 부품 → 다 모으면 멀티클래스 학습
function AutoLabelView() {
  const [folders, setFolders] = useState([])        // [{folder,label,videos:[{name,count,ready}]}]
  const [folder, setFolder] = useState(null)        // 선택한 폴더(data 기준 rel 경로)
  const [partIdx, setPartIdx] = useState(0)         // 폴더 안 현재 부품(영상) 인덱스
  const [count, setCount] = useState(0)             // 현재 부품 프레임 수
  const [idx, setIdx] = useState(0)                 // 현재 프레임
  const [ptsBySrc, setPtsBySrc] = useState(() => {  // { 영상: { 프레임: [{rx,ry,lab}] } } 영상별 보관
    try { return JSON.parse(localStorage.getItem('autolabel_shots_v1') || '{}') } catch { return {} }
  })
  const [preparing, setPreparing] = useState(false)
  const [masks, setMasks] = useState({})            // {"영상:프레임": {combo,area_frac,bbox}}
  const [activeShot, setActiveShot] = useState(null)// 크게 보고 있는 마스크 참조샷 키
  const [maskBusy, setMaskBusy] = useState(false)

  const [session, setSession] = useState(() => localStorage.getItem('parts_session_v1') || null)
  const [labeledMap, setLabeledMap] = useState({})  // {영상: {labels,frames}} 현재 세션에 라벨된 부품
  const [labelJob, setLabelJob] = useState(null)
  const [labelStatus, setLabelStatus] = useState(null)

  const [evalSel, setEvalSel] = useState([])        // 평가(test) 영상
  const [trainJob, setTrainJob] = useState(null)
  const [trainStatus, setTrainStatus] = useState(null)

  const running = !!labelStatus?.running || !!trainStatus?.running

  const folderObj = folders.find(f => f.folder === folder)
  const folderVideos = folderObj?.videos || []
  const curPart = folderVideos[partIdx]
  const src = curPart?.name || null

  const markReady = (name, cnt) => setFolders(prev => prev.map(f => ({
    ...f, videos: f.videos.map(v => v.name === name ? { ...v, ready: true, count: cnt } : v)
  })))
  const prepareSrc = async (name) => {   // 프레임 미컷이면 컷
    setPreparing(true)
    const d = await fetch(`/api/autolabel/prepare?src=${encodeURIComponent(name)}`).then(r => r.json())
    setPreparing(false); setCount(d.count || 0); markReady(name, d.count || 0)
  }

  const loadFolders = useCallback(() => {
    fetch('/api/autolabel/folders').then(r => r.json()).then(d => {
      setFolders(d)
      setFolder(cur => {
        if (cur || !d[0]) return cur
        setEvalSel(d[0].videos.filter(v => /test/i.test(v.name)).map(v => v.name))
        return d[0].folder
      })
    })
  }, [])
  const loadSessions = useCallback(() => {
    fetch('/api/sam2/parts_sessions').then(r => r.json()).then(list => {
      const saved = localStorage.getItem('parts_session_v1')
      const found = list.find(s => s.session === saved) || list[0]
      if (found) { setSession(found.session); setLabeledMap(found.videos || {}) }
    }).catch(() => {})
  }, [])
  useEffect(() => { loadFolders() }, [loadFolders])
  useEffect(() => { loadSessions() }, [loadSessions])

  useEffect(() => {   // 부품 바뀌면 그 영상 프레임 준비
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
  const partStatus = (name) => isLabeled(name) ? 'done' : (hasTaps(name) ? 'tapped' : 'todo')
  const nLabeled = folderVideos.filter(v => isLabeled(v.name)).length
  const curShots = buildShots(src)                 // 현재 부품의 유효 참조샷

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

  const goPart = (i) => {
    const n = Math.max(0, Math.min(i, folderVideos.length - 1))
    setPartIdx(n)
  }
  const chooseFolder = (rel) => {
    const fo = folders.find(f => f.folder === rel); if (!fo) return
    setFolder(rel); setPartIdx(0); setTrainStatus(null); setTrainJob(null)
    setEvalSel(fo.videos.filter(v => /test/i.test(v.name)).map(v => v.name))
  }
  const newSession = () => {
    setSession(null); setLabeledMap({}); localStorage.removeItem('parts_session_v1')
    setTrainStatus(null); setTrainJob(null)
  }

  // 멀티클래스 학습: 세션 누적 라벨 → 34클래스 통합 → YOLO 학습 → test 검출 평가
  const toggleEval = (n) => setEvalSel(c => c.includes(n) ? c.filter(x => x !== n) : [...c, n])
  const runTrain = async () => {
    const r = await fetch('/api/sam2/multiclass', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session, epochs: 100, test_srcs: evalSel })
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
  const testVideos = folderVideos.filter(v => /test/i.test(v.name))

  return (
    <div>
      <h2>부품 라벨링 (SAM2)</h2>

      <h3 className="section-h">폴더 선택</h3>
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

      {/* 세션 + 전체 진행 */}
      <div className="wiz-session">
        <span className="al-hint">세션 <b>{session || '(첫 라벨 생성 시 자동)'}</b></span>
        <span className="al-hint">라벨 완료 <b>{nLabeled}</b> / {folderVideos.length} 부품</span>
        <button className="al-secondary sm" onClick={newSession} disabled={running}>새 세션</button>
      </div>
      <div className="al-progress"><i style={{ width: `${folderVideos.length ? nLabeled / folderVideos.length * 100 : 0}%` }} /></div>

      {/* 부품 진행 레일: 클릭하면 그 부품으로 이동 */}
      <div className="part-rail">
        {folderVideos.map((v, i) => (
          <button key={v.name} className={`part-chip ${partStatus(v.name)} ${i === partIdx ? 'on' : ''}`}
                  onClick={() => goPart(i)} disabled={running} title={v.name}>
            {v.name}
          </button>
        ))}
      </div>

      {!src ? <p className="al-hint">폴더를 선택하세요.</p> : <>
        {/* 현재 부품 */}
        <div className="wiz-head">
          <span className="wiz-title">{src}</span>
          <span className="wiz-idx">부품 {partIdx + 1} / {folderVideos.length}</span>
          {isLabeled(src) && <span className="badge adopt">✓ 라벨 {labeledMap[src].labels}장</span>}
          <span style={{ marginLeft: 'auto' }} />
          <button className="al-secondary sm" onClick={() => goPart(partIdx - 1)} disabled={running || partIdx === 0}>◀ 이전 부품</button>
          <button className="al-secondary sm" onClick={() => goPart(partIdx + 1)} disabled={running || partIdx >= folderVideos.length - 1}>다음 부품 ▶</button>
        </div>

        <p className="al-hint">부품을 <b>좌클릭</b>(포함점), 배경 오채택되면 <b>우클릭</b>(제외점). 여러 각도가 필요하면 프레임을 넘겨 한 번 더 탭.</p>

        {preparing
          ? <div className="al-frame" style={{ padding: 44, textAlign: 'center', cursor: 'default' }}>
              <span className="al-hint" style={{ color: '#e2e8f0' }}>프레임 컷 중... (몇 초~수십 초)</span>
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
          <span className="al-hint">frame {idx + 1} / {count} · 이 프레임 점 {cur.length}개 · 참조샷 {curShots.length}</span>
        </div>
        <div className="al-controls">
          <button className="chip" onClick={undo} disabled={running || !cur.length}>점 취소</button>
          <button className="chip" onClick={clearFrame} disabled={running || !cur.length}>이 프레임 지우기</button>
          <button className="cmp-random" onClick={previewMask} disabled={running || maskBusy || !cur.length}>
            {maskBusy ? '마스크 생성 중...' : '입력 마스크 확인'}
          </button>
        </div>

        {/* 입력 마스크 크게 확인 (주요 콘텐츠) */}
        {activeMask && (activeMask.error
          ? <p className="fn" style={{ color: '#b91c1c' }}>마스크 오류: {activeMask.error}</p>
          : <div className="al-maskbig">
              <div className="al-maskbig-cap">
                입력 마스크 · <span className="mk-g">초록=마스크</span> <span className="mk-o">주황=박스</span> <span className="mk-b">파랑=포함점</span> <span className="mk-r">빨강=제외점</span>
                {typeof activeMask.area_frac === 'number' && <> · 면적 {(activeMask.area_frac * 100).toFixed(1)}%</>}
              </div>
              <img src={activeMask.combo} alt="입력 마스크" />
            </div>)}

        {/* 라벨 생성 → 다음 부품 */}
        <div className="al-controls" style={{ marginTop: 14 }}>
          <button className="al-primary" onClick={genLabel}
                  disabled={running || curShots.length === 0}>
            {labelStatus?.running ? '라벨 생성 중...' : (isLabeled(src) ? '↻ 라벨 다시 생성' : '라벨 생성')}
          </button>
          {(genDone || isLabeled(src)) && partIdx < folderVideos.length - 1 &&
            <button className="al-primary next" onClick={() => goPart(partIdx + 1)} disabled={running}>다음 부품 ▶</button>}
          {labelStatus && !labelStatus.error && labelStatus.video === src &&
            <span className="al-hint">
              {labelStatus.running ? `${src} 영상 전파 중...` : (labelStatus.stage === 'done' ? `✓ 라벨 ${labelStatus.labels}장 / ${labelStatus.frames}프레임` : '')}
            </span>}
        </div>
        {labelStatus?.error && <p className="fn" style={{ color: '#b91c1c' }}>오류: {labelStatus.error}</p>}
        {genDone && labelStatus.taps?.length > 0 &&
          <div className="al-thumbs">{labelStatus.taps.map((u, i) => <img key={i} src={u} alt={`tap ${i}`} />)}</div>}
      </>}

      {/* 멀티클래스 학습 */}
      {nLabeled > 0 && <>
        <h3 className="section-h">멀티클래스 학습</h3>
        <p className="al-hint">라벨 완료 부품 <b>{nLabeled}</b>개를 클래스별로 통합해 YOLO 학습. 평가는 아래 test 영상(정답 라벨 없어 검출률·신뢰도·클래스 분포).</p>

        <h4 className="subtable-title">평가(test) 영상</h4>
        <div className="chips">
          {testVideos.length === 0 && <span className="al-hint">이 폴더에 test 영상이 없습니다.</span>}
          {testVideos.map(v => {
            const on = evalSel.includes(v.name)
            return (
              <button key={v.name} className={on ? 'chip on' : 'chip'} disabled={running} onClick={() => toggleEval(v.name)}>
                {on ? '☑ ' : '☐ '}{v.name}
              </button>
            )
          })}
        </div>

        <div className="al-controls" style={{ marginTop: 10 }}>
          <button className="al-primary" onClick={runTrain} disabled={running || !session}>
            {trainStatus?.running ? '학습 중...' : '멀티클래스 학습 시작'}
          </button>
        </div>

        {trainStatus?.error && <p className="fn" style={{ color: '#b91c1c' }}>오류: {trainStatus.error}</p>}
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
                  <th>test 영상</th><th>프레임</th><th>검출</th><th>검출률</th><th>평균 신뢰도</th><th>주요 클래스</th>
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
      </>}
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
