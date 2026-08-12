import { useEffect, useState, useCallback, useRef } from 'react'

// Lucide 스타일 인라인 SVG 아이콘 (특수문자 ◀▶ 대체)
const SVG = (children) => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
       strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block', flexShrink: 0 }}>{children}</svg>
)
const IcChevronLeft = () => SVG(<polyline points="15 18 9 12 15 6" />)
const IcChevronRight = () => SVG(<polyline points="9 18 15 12 9 6" />)
const IcChevronDown = () => SVG(<polyline points="6 9 12 15 18 9" />)
const IcSkipBack = () => SVG(<><polygon points="19 20 9 12 19 4 19 20" /><line x1="5" y1="19" x2="5" y2="5" /></>)
const IcSkipForward = () => SVG(<><polygon points="5 4 15 12 5 20 5 4" /><line x1="19" y1="5" x2="19" y2="19" /></>)
const IcX = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6"
       strokeLinecap="round" style={{ display: 'block' }}><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
)
const IcCheck = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"
       strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}><polyline points="20 6 9 17 4 12" /></svg>
)
const IcWarn = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
       strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block', flexShrink: 0 }}>
    <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
)
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

// data/bell412/<그룹>/<부품>/videos → 부품 파싱. 폴더=부품, 폴더 안 영상=그 부품의 train/test 테이크
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

// 부품 라벨링(SAM2): 데이터셋(그룹)→부품(폴더) 순회. 부품마다 학습 테이크 탭 → 입력 마스크 확인 → 라벨 생성 → 다음 부품
function AutoLabelView() {
  const [folders, setFolders] = useState([])        // [{folder,label,videos:[{name,count,ready}]}]
  const [partIdx, setPartIdx] = useState(0)         // 부품(폴더) 인덱스
  const [takeIdx, setTakeIdx] = useState(0)         // (구) 학습 테이크 인덱스
  const [selVideo, setSelVideo] = useState(null)    // 선택해서 보고 있는 영상(없으면 기본=첫 학습영상)
  const [showVideoPick, setShowVideoPick] = useState(false)  // 영상 선택 모달(라벨검수식)
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
  const [servedSet, setServedSet] = useState(new Set())   // 현재 서비스 모델이 보유한 부품(학습됨 판단 기준)
  const [labeledAnywhere, setLabeledAnywhere] = useState(new Set())   // train 라벨이 하나라도 있는 부품(세션 무관) — 라벨 검수 활성 판단
  useEffect(() => {   // 현재 서비스 모델 클래스 + 라벨 보유 부품 로드
    fetch('/api/sam2/served').then(r => r.json())
      .then(d => setServedSet(new Set((d && !d.none && d.classes) || [])))
      .catch(() => {})
    fetch('/api/sam2/labeled_parts').then(r => r.json())
      .then(d => setLabeledAnywhere(new Set(d.parts || [])))
      .catch(() => {})
  }, [])
  useEffect(() => {   // 서버 영속 참조샷(shots.json) = 로컬 기준 단일 소스 → ptsBySrc 를 서버값으로 교체
    // localStorage 는 오프라인 폴백/캐시일 뿐. 서버 로드 성공 시 통째 교체해서 stale 키(공백 변형 등 중복)를 폐기하고
    // 항상 서버(=로컬 영상 stem) 기준으로 맞춘다. 서버 shots.json = {영상: {프레임: [[rx,ry,lab],...]}}.
    fetch('/api/sam2/shots').then(r => r.json()).then(d => {
      if (!d || typeof d !== 'object') return
      const conv = {}
      for (const [video, frames] of Object.entries(d)) {
        const fc = {}
        for (const [fi, arr] of Object.entries(frames || {})) {
          fc[fi] = (arr || []).map(p => ({ rx: p[0], ry: p[1], lab: p[2] }))
        }
        conv[video] = fc
      }
      setPtsBySrc(conv)   // 서버가 durable 소스 → 통째 교체(stale localStorage 키 제거, 로컬 기준 통일)
    }).catch(() => {})   // 실패 시 localStorage 초기값 유지(오프라인 폴백)
  }, [])
  const [showReview, setShowReview] = useState(false)   // 라벨 검수 모달
  const [reviewFrames, setReviewFrames] = useState([])
  const [selParts, setSelParts] = useState(() => new Set())   // 일괄 라벨 생성 대상 부품(체크박스 선택)

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
  const src = (selVideo && folderVideos.some(v => v.name === selVideo))   // 선택한 영상(없으면 기본=첫 학습영상 or 첫 영상)
    ? selVideo : (roles.train[0] || folderVideos[0]?.name || null)
  const testTake = roles.test                                            // 이 부품 테스트 테이크(2 또는 학습영상 자체)
  const testIsSelf = testTake && roles.train.includes(testTake)
  // 부품경로 기반 유니크 키(bell412/<부품>/videos/<stem>). 같은 이름 영상이 다른 부품에 있어도 안 꼬이게
  // API(prepare·frame·mask·parts_label)엔 이 키를 넘긴다. 화면/상태 키는 여전히 stem(부품 폴더 내 유니크).
  const keyOf = (name) => {
    const v = folderVideos.find(x => x.name === name)
    return v?.key || (curPartFolder ? `${curPartFolder.folder}/${name}` : name)
  }
  const srcKey = src ? keyOf(src) : null

  const markReady = (name, cnt) => setFolders(prev => prev.map(f => ({
    ...f, videos: f.videos.map(v => v.name === name ? { ...v, ready: true, count: cnt } : v)
  })))
  const prepareSrc = async (name, key) => {   // 프레임 미컷이면 컷 (key=부품경로 키, 없으면 name)
    setPreparing(true)
    const d = await fetch(`/api/autolabel/prepare?src=${encodeURIComponent(key || name)}`).then(r => r.json())
    setPreparing(false); setCount(d.count || 0); markReady(name, d.count || 0)
  }
  const prepareAll = async (pfs) => {   // 그룹 전체 프레임 미리 컷(탭 전에 자동, 백그라운드)
    for (const pf of pfs) {
      for (const v of pf.videos) {
        if (v.ready) continue
        setPrepProg(`프레임 미리 컷: ${v.name}`)
        const k = v.key || `${pf.folder}/${v.name}`
        try { const d = await fetch(`/api/autolabel/prepare?src=${encodeURIComponent(k)}`).then(r => r.json()); markReady(v.name, d.count || 0) } catch { /* skip */ }
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
      const found = list.find(s => s.session === saved) || list[0]   // 단일 영속 세션(부품별 저장소 합성) 자동 채택
      if (found) { setSession(found.session); setLabeledMap(found.videos || {}); localStorage.setItem('parts_session_v1', found.session) }
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
    if (v?.ready) setCount(v.count || 0); else prepareSrc(src, srcKey)
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
  // 탭해둔(참조샷 있는) 학습 테이크 전체 → 한 번에 라벨 생성용 (video=부품경로 키)
  const tappedItems = partFolders.map(pf => ({ pf, stem: pfTrain(pf)[0] })).filter(x => x.stem)
    .filter(x => buildShots(x.stem).length > 0).map(x => ({ video: `${x.pf.folder}/${x.stem}`, shots: buildShots(x.stem) }))
  // 부품 선택(체크박스) → 선택한 부품 중 참조샷 있는 것들 일괄 라벨 생성 대상
  const toggleSelPart = (folder) => setSelParts(s => { const n = new Set(s); n.has(folder) ? n.delete(folder) : n.add(folder); return n })
  const allPartsSelected = partFolders.length > 0 && partFolders.every(pf => selParts.has(pf.folder))
  const toggleAllParts = () => setSelParts(allPartsSelected ? new Set() : new Set(partFolders.map(pf => pf.folder)))
  const selItems = partFolders.filter(pf => selParts.has(pf.folder))
    .map(pf => ({ pf, stem: pfTrain(pf)[0] })).filter(x => x.stem)
    .filter(x => buildShots(x.stem).length > 0).map(x => ({ video: `${x.pf.folder}/${x.stem}`, shots: buildShots(x.stem) }))
  const batchMode = selParts.size > 0
  // 학습 체크박스 전체선택/해제
  const allSelected = labeledParts.length > 0 && labeledParts.every(p => !excluded.includes(p))
  const toggleAll = () => setExcluded(allSelected ? [...labeledParts] : [])
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
      body: JSON.stringify({ src: srcKey, frame: idx, points: cur.map(p => [p.rx, p.ry, p.lab]) })
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
      body: JSON.stringify({ session, video: srcKey, shots: curShots })
    }).then(x => x.json())
    if (r.error) { setLabelStatus({ error: r.error }); return }
    if (r.session) { setSession(r.session); localStorage.setItem('parts_session_v1', r.session) }
    setLabelJob(r.job); setLabelStatus({ stage: 'start', running: true, video: src })
  }
  // 선택한 부품들 한 번에 라벨 생성(배치)
  const genLabelBatch = async (items) => {
    const list = items && items.length ? items : selItems
    if (!list.length) return
    const r = await fetch('/api/sam2/parts_label_batch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session, items: list })
    }).then(x => x.json())
    if (r.error) { setLabelStatus({ error: r.error }); return }
    if (r.session) { setSession(r.session); localStorage.setItem('parts_session_v1', r.session) }
    setLabelJob(r.job); setLabelStatus({ stage: 'start', running: true, batch: true })
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
          if (d.results) setLabeledMap(m => { const n = { ...m }; d.results.forEach(x => { n[x.video] = { labels: x.labels, frames: x.frames } }); return n })
          else if (d.video) setLabeledMap(m => ({ ...m, [d.video]: { labels: d.labels, frames: d.frames } }))
          fetch('/api/sam2/labeled_parts').then(r => r.json()).then(x => setLabeledAnywhere(new Set(x.parts || []))).catch(() => {})   // 검수 활성 목록 갱신
        }
      }
    }, 1200)
    return () => clearInterval(t)
  }, [labelJob, labelStatus?.running])

  const goPart = (i) => { setPartIdx(Math.max(0, Math.min(i, partFolders.length - 1))); setTakeIdx(0); setSelVideo(null) }

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

  const activeMask = masks[shotKey(src, idx)] || null   // 현재 프레임의 마스크만 표시(프레임 바뀌면 자동으로 사라짐)
  const genDone = labelStatus?.stage === 'done' && labelStatus?.video === src
  const partName = curPartFolder ? partOf(curPartFolder.folder) : ''
  const evalList = testSrcs()
  const openReview = () => {   // 현재 부품의 생성된 학습 프레임(세션 무관)을 검수 모달로
    if (!partName) return
    fetch(`/api/sam2/part_frames?part=${encodeURIComponent(partName)}`).then(r => r.json())
      .then(d => { setReviewFrames(d.frames || []); setShowReview(true) }).catch(() => {})
  }
  const delReviewFrame = async (f) => {   // 잘못된 프레임 삭제(그 프레임이 속한 세션에서 제거)
    await fetch('/api/sam2/delete_train_frame', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: f.session, name: f.name, part: f.part })
    }).catch(() => {})
    setReviewFrames(fs => fs.filter(x => !(x.session === f.session && x.name === f.name)))
  }
  // 영상 삭제(모달): 원본은 서버 _trash 로 이동(복구 가능), 프레임캐시·오토라벨은 삭제
  const deleteVideo = async (name) => {
    if (running) return
    if (!window.confirm(`'${name}' 영상을 삭제할까요?\n· 원본은 _trash 폴더로 이동(복구 가능)\n· 이 영상의 프레임캐시와 자동생성 라벨은 삭제됩니다.`)) return
    const key = keyOf(name)
    const r = await fetch('/api/sam2/delete_video', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ src: key })
    }).then(x => x.json()).catch(() => ({ error: '삭제 요청 실패' }))
    if (r.error) { window.alert(r.error); return }
    setPtsBySrc(prev => { const n = { ...prev }; delete n[name]; return n })   // 이 영상 탭 정리
    if (selVideo === name) setSelVideo(null)
    loadFolders(); loadSessions()
    fetch('/api/sam2/labeled_parts').then(x => x.json()).then(x => setLabeledAnywhere(new Set(x.parts || []))).catch(() => {})
  }

  return (
    <div>
      {!src ? <p className="al-hint">부품 폴더가 없습니다. (data/bell412/parts/&lt;부품&gt;/videos)</p> : (
        <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 196px)', minHeight: 420, overflow: 'hidden' }}>
          {/* 상단 범례 제거 — 범례는 이미지·입력마스크 아래로 이동 */}
          {/* 상단 액션 버튼줄 */}
          <div className="al-controls" style={{ flexShrink: 0 }}>
            {roles.train.length >= 1 && (
              <button className="act-btn neutral" onClick={() => setShowVideoPick(true)} disabled={running} title="학습 영상 선택·미리보기">영상 선택</button>
            )}
            <button className="act-btn neutral" onClick={undo} disabled={running || !cur.length}>점 취소</button>
            <button className="act-btn neutral" onClick={clearFrame} disabled={running || !cur.length}>지우기</button>
            <button className="act-btn primary" onClick={previewMask} disabled={running || maskBusy || !cur.length}>
              {maskBusy ? '생성 중...' : '입력 마스크 확인'}
            </button>
            <button className="act-btn primary"
                    onClick={batchMode ? () => genLabelBatch() : genLabel}
                    disabled={running || (batchMode ? selItems.length === 0 : curShots.length === 0)}
                    title={batchMode ? '선택한 부품들의 참조샷으로 한 번에 라벨 생성' : ''}>
              {labelStatus?.running ? '라벨 생성 중...'
                : batchMode ? `선택 부품 라벨 생성 (${selItems.length})`
                : ((isLabeled(src) || servedSet.has(partName)) ? '↻ 라벨 다시 생성' : '라벨 생성')}
            </button>
            <button className="act-btn neutral" onClick={openReview} disabled={running || !(isLabeled(src) || labeledAnywhere.has(partName))}
                    title="현재 부품에 생성된 학습 라벨 프레임을 확인하고 잘못된 사진을 삭제">라벨 검수</button>
            {labelStatus && !labelStatus.error && labelStatus.video === src && labelStatus.running &&
              <span className="al-hint">전파 중...</span>}
          </div>

          {/* (영상 선택 버튼은 상단 액션줄로 이동) */}

          {/* 참조샷: 액션 버튼 바로 아래, 가로 칩 리스트 */}
          {shotFrames.length > 0 && (
            <div className="al-shots" style={{ flexShrink: 0, marginTop: 6 }}>
              <span className="ref-shots-label">참조샷 {shotFrames.length}</span>
              {shotFrames.map(i => {
                const npos = (pts[i] || []).filter(p => p.lab === 1).length
                const done = !!masks[shotKey(src, i)] && !masks[shotKey(src, i)].error
                return (
                  <span key={i} className={`al-shot ${idx === i ? 'on' : ''} ${npos >= 1 ? '' : 'bad'}`}>
                    <span className="al-shot-lbl" onClick={() => !running && goShot(i)}>{done ? '✓ ' : ''}#{i + 1}</span>
                    <button className="al-shot-close" title="이 프레임 탭 삭제" onClick={() => !running && deleteShotFrame(i)}><IcX /></button>
                  </span>
                )
              })}
            </div>
          )}

          {/* 본문: 좌(이미지+범례+재생) / 우(리스트). 남는 공간 채움 */}
          <div style={{ display: 'flex', gap: 16, flex: 1, minHeight: 0, overflow: 'hidden', marginTop: 8 }}>
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
             {/* 이미지+범례+재생 = 남는 폭을 채우되 상한(사이드바 넓혀도 겹치지 않게 반응형) */}
             <div style={{ width: '100%', maxWidth: 720, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              {/* 마스크 상태: 참조샷과 이미지 사이 */}
              {activeMask && !activeMask.error && (
                activeMask.verdict === 'over'
                  ? <div className="mask-alert warn" style={{ flexShrink: 0, marginBottom: 8 }}><IcWarn /><span>배경까지 넓게 잡힘 · 부품 안쪽 점만 남기고 배경엔 제외점을 찍어보세요</span></div>
                  : <div className="mask-alert ok" style={{ flexShrink: 0, marginBottom: 8 }}><IcCheck /><span>부품이 잘 잡혔어요</span></div>
              )}
              {/* 듀얼 이미지: 좌우 동일 비율로 축소(스크롤 없음), 여백 흰색 */}
              <div style={{ display: 'flex', gap: 12, flex: 1, minHeight: 0 }}>
                <div className="img-pane">
                  {preparing
                    ? <span className="al-hint">프레임 컷 중...</span>
                    : <div className="tap-box" onClick={(e) => addPoint(e, 1)} onContextMenu={(e) => addPoint(e, 0)}>
                        {src && <img src={`/api/autolabel/frame?src=${encodeURIComponent(srcKey)}&idx=${idx}&w=720`} alt={`frame ${idx}`} draggable={false}
                                     style={{ maxHeight: 'calc(100vh - 444px)', maxWidth: '100%' }} />}
                        {cur.map((p, i) => (
                          <span key={i} className={`al-dot ${p.lab === 1 ? 'pos' : 'neg'}`}
                                style={{ left: `${p.rx * 100}%`, top: `${p.ry * 100}%` }} />
                        ))}
                      </div>}
                </div>
                <div className="img-pane">
                  {activeMask
                    ? (activeMask.error
                        ? <span className="fn" style={{ color: '#b91c1c' }}>마스크 오류: {activeMask.error}</span>
                        : <img src={activeMask.combo} alt="입력 마스크" style={{ maxHeight: 'calc(100vh - 444px)', maxWidth: '100%' }} />)
                    : <span className="al-hint" style={{ padding: 12, textAlign: 'center' }}>입력 마스크 확인을 누르면 여기에 표시됩니다</span>}
                </div>
              </div>


              {/* 범례: 이미지·입력마스크 아래, 작은 배경 박스(존) 안에 색상만 */}
              <div className="mask-panel" style={{ flexShrink: 0, marginTop: 8 }}>
                <div className="mask-legend">
                  <span className="lg-item"><i className="lg-dot" style={{ background: '#3b82f6' }} />포함점</span>
                  <span className="lg-item"><i className="lg-dot" style={{ background: '#f87171' }} />제외점</span>
                  <span className="lg-item"><i className="lg-dot" style={{ background: '#34d399' }} />마스크</span>
                  <span className="lg-item"><i className="lg-dot" style={{ background: '#fb923c' }} />박스</span>
                  {prepProg && <span className="al-hint" style={{ marginLeft: 'auto' }}>⏳ {prepProg}</span>}
                </div>
              </div>

              {/* 재생 컨트롤 바: 이미지 바로 아래, 이미지 폭에 꽉(비디오 플레이어 느낌) */}
              <div className="al-controls" style={{ flexShrink: 0, marginTop: 8 }}>
                <button className="pb-btn" title="10프레임 뒤로" onClick={() => setIdx(i => Math.max(i - 10, 0))} disabled={running}><IcSkipBack /><span>10</span></button>
                <button className="pb-btn" title="이전 프레임" onClick={() => setIdx(i => Math.max(i - 1, 0))} disabled={running}><IcChevronLeft /></button>
                <input className="al-slider" type="range" min={0} max={Math.max(count - 1, 0)} value={idx}
                       onChange={(e) => setIdx(+e.target.value)} disabled={running} />
                <span className="frame-jump">
                  <input type="number" min={1} max={count || 1} value={idx + 1} disabled={running}
                         onChange={(e) => { const v = Math.min(Math.max(1, Math.floor(+e.target.value) || 1), count || 1); setIdx(v - 1) }} />
                  <span className="al-hint">/ {count}</span>
                </span>
                <button className="pb-btn" title="다음 프레임" onClick={() => setIdx(i => Math.min(i + 1, count - 1))} disabled={running}><IcChevronRight /></button>
                <button className="pb-btn" title="10프레임 앞으로" onClick={() => setIdx(i => Math.min(i + 10, count - 1))} disabled={running}><span>10</span><IcSkipForward /></button>
              </div>
             </div>
              {labelStatus?.error && <p className="fn" style={{ color: '#b91c1c', flexShrink: 0 }}>오류: {labelStatus.error}</p>}
            </div>

            {/* 오른쪽: 부품 패널 = 상단 고정 이전/다음 헤더 + 스크롤 리스트 */}
            <div className="part-panel">
              <div className="part-panel-head">
                <input type="checkbox" className="part-sel" checked={allPartsSelected} disabled={running || partFolders.length === 0}
                       ref={el => { if (el) el.indeterminate = selParts.size > 0 && !allPartsSelected }}
                       onChange={toggleAllParts} title="전체 선택 / 해제" aria-label="부품 전체 선택" />
                <span className="al-hint" style={{ fontWeight: 600 }}>부품 {nLabeled}/{partFolders.length}</span>
                <span style={{ flex: 1 }} />
                <button className="pb-btn ico" onClick={() => goPart(partIdx - 1)} disabled={running || partIdx === 0} title="이전 부품"><IcChevronLeft /></button>
                <button className="pb-btn ico" onClick={() => goPart(partIdx + 1)} disabled={running || partIdx >= partFolders.length - 1} title="다음 부품"><IcChevronRight /></button>
              </div>
              <div className="part-list">
                {partFolders.map((pf, i) => {
                  const part = partOf(pf.folder)
                  const st = pfStatus(pf)
                  return (
                    <div key={pf.folder} className={`part-item ${st} ${i === partIdx ? 'on' : ''}`}
                         onClick={() => !running && goPart(i)} title={st === 'done' ? `${part} (라벨됨)` : part}>
                      <input type="checkbox" className="part-sel" checked={selParts.has(pf.folder)} disabled={running}
                             onClick={(e) => e.stopPropagation()} onChange={() => toggleSelPart(pf.folder)} aria-label={`${part} 선택`} />
                      <span className="part-name">{part}</span>
                      {servedSet.has(part)
                        ? <span className="tchip-badge">학습됨</span>
                        : <span className="tchip-badge new">신규</span>}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 라벨 검수 모달: 생성된 학습 프레임(bbox) 확인 + 잘못된 사진 삭제 */}
      {showReview && (
        <div className="modal-scrim" onClick={() => setShowReview(false)}>
          <div className="modal-card wide" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <b>라벨 검수 — {partName} {reviewFrames.length}장</b>
              <button className="icon-x" onClick={() => setShowReview(false)} aria-label="닫기"><IcX /></button>
            </div>
            <div className="review-grid">
              {reviewFrames.length === 0 && <p className="al-hint" style={{ padding: 18 }}>이 부품의 생성된 라벨이 없습니다.</p>}
              {reviewFrames.map(f => (
                <figure key={`${f.session}/${f.name}`} className="review-cell">
                  <img loading="lazy" alt={f.part}
                       src={`/api/sam2/train_frame?session=${encodeURIComponent(f.session)}&name=${encodeURIComponent(f.name)}&w=240&part=${encodeURIComponent(f.part || '')}`} />
                  <figcaption title={f.name}>{f.part}</figcaption>
                  <button className="review-del" title="이 프레임 삭제" onClick={() => delReviewFrame(f)}><IcX /></button>
                </figure>
              ))}
            </div>
            <div className="modal-foot">
              <span className="al-hint" style={{ marginRight: 'auto' }}>잘못 잡힌 프레임은 ×로 삭제하세요. 삭제 즉시 학습셋에서 빠집니다.</span>
              <button className="act-btn neutral" onClick={() => setShowReview(false)}>닫기</button>
            </div>
          </div>
        </div>
      )}

      {/* 영상 선택 모달: 이 부품 학습 영상(test 제외) 중 골라 보기 */}
      {showVideoPick && (
        <div className="modal-scrim" onClick={() => setShowVideoPick(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <b>{partName} 영상 선택</b>
              <button className="icon-x" onClick={() => setShowVideoPick(false)} aria-label="닫기"><IcX /></button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, padding: 16, maxHeight: '64vh', overflow: 'auto' }}>
              {roles.train.map(v => {
                const on = v === src
                return (
                  <div key={v} role="button" tabIndex={0} onClick={() => { setSelVideo(v); setShowVideoPick(false) }}
                       onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setSelVideo(v); setShowVideoPick(false) } }}
                       style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 6, padding: 8, borderRadius: 10, cursor: 'pointer', textAlign: 'left',
                                border: on ? '2px solid var(--accent)' : '1px solid var(--line-strong)', background: on ? 'var(--accent-bg)' : 'var(--surface)' }}>
                    {on && <span style={{ position: 'absolute', top: 12, left: 12, zIndex: 1, background: 'var(--accent)', color: '#fff',
                                          fontSize: 12, fontWeight: 700, padding: '3px 9px', borderRadius: 6 }}>선택됨</span>}
                    {/* 영상 삭제(X): 원본은 _trash 로 이동(복구 가능), 캐시·라벨은 삭제 */}
                    <button className="review-del" title="이 영상 삭제" disabled={running}
                            onClick={(e) => { e.stopPropagation(); deleteVideo(v) }}
                            style={{ position: 'absolute', top: 10, right: 10, zIndex: 2 }}><IcX /></button>
                    <img src={`/api/autolabel/frame?src=${encodeURIComponent(keyOf(v))}&idx=0&w=360`} alt={v} loading="lazy"
                         style={{ width: '100%', aspectRatio: '16 / 10', objectFit: 'cover', borderRadius: 6, background: 'var(--bg)' }} />
                    <span style={{ fontSize: 14, fontWeight: on ? 700 : 500, color: on ? 'var(--accent-ink)' : 'var(--ink-strong)' }}>
                      {isLabeled(v) ? '✓ ' : ''}{v}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

// 부품 인식 앱: 라벨 생성 → 학습 → 학습 결과 3단계를 한 화면에서 스텝 전환
function LogConsole({ log, compact }) {   // 터미널 로그: 새 줄마다 자동 하단 스크롤. compact=학습중 축소(1/3)
  const ref = useRef(null)
  const stick = useRef(true)
  useEffect(() => {
    const el = ref.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [log])
  const onScroll = () => {
    const el = ref.current
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 28
  }
  return (
    <div className={`term${compact ? ' compact' : ''}`} ref={ref} onScroll={onScroll}>
      {(log || []).map((l, i) => <div key={i} className={`term-line ${l.level || 'info'}`}>{l.msg}</div>)}
      {!(log && log.length) && <div className="term-line dim">로그 대기 중...</div>}
    </div>
  )
}

// 원형 프로그레스: 진행률(pct)에 맞춰 링이 차오르고 중앙 숫자가 부드럽게 카운트업. 완료 시 녹색 체크
function CircularProgress({ pct, done }) {
  const [disp, setDisp] = useState(0)
  useEffect(() => {
    let raf, cur = disp
    const tick = () => {
      cur += (pct - cur) * 0.25
      if (Math.abs(pct - cur) < 0.5) { setDisp(pct); return }
      setDisp(Math.round(cur)); raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pct])
  const r = 32, c = 2 * Math.PI * r
  const off = c * (1 - (done ? 100 : pct) / 100)
  return (
    <div className={`circ ${done ? 'done' : 'run'}`}>
      <svg width="80" height="80" viewBox="0 0 80 80">
        <circle className="circ-track" cx="40" cy="40" r={r} />
        <circle className="circ-bar" cx="40" cy="40" r={r} style={{ strokeDasharray: c, strokeDashoffset: off }} />
      </svg>
      <div className="circ-label">{done ? <IcCheck /> : `${disp}%`}</div>
    </div>
  )
}

// 학습곡선: 에폭별 지표를 SVG 선그래프로(외부 라이브러리 없이). data=curve, series=[{key,name,color}]
function MiniLineChart({ title, data, series }) {
  const pts = (data || []).filter(d => series.some(s => d[s.key] != null))
  const W = 300, H = 130, pl = 40, pr = 10, pt = 10, pb = 22
  let body = <div className="chart-empty">데이터 대기 중...</div>
  if (pts.length) {
    const xs = pts.map(p => p.epoch)
    const xmin = Math.min(...xs), xmax = Math.max(...xs)
    const ys = []
    series.forEach(s => pts.forEach(p => { if (p[s.key] != null) ys.push(p[s.key]) }))
    let ymin = Math.min(...ys), ymax = Math.max(...ys)
    if (ymin === ymax) { ymin -= 0.01; ymax += 0.01 }
    const sx = x => pl + (xmax === xmin ? 0 : (x - xmin) / (xmax - xmin)) * (W - pl - pr)
    const sy = y => pt + (1 - (y - ymin) / (ymax - ymin)) * (H - pt - pb)
    const fmt = v => Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)
    body = (
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg">
        {[ymax, (ymax + ymin) / 2, ymin].map((v, i) => (
          <g key={i}>
            <line x1={pl} y1={sy(v)} x2={W - pr} y2={sy(v)} className="chart-grid" />
            <text x={pl - 4} y={sy(v) + 3} className="chart-tick" textAnchor="end">{fmt(v)}</text>
          </g>
        ))}
        {series.map(s => {
          const sp = pts.filter(p => p[s.key] != null)
          const d = sp.map((p, i) => `${i ? 'L' : 'M'}${sx(p.epoch).toFixed(1)} ${sy(p[s.key]).toFixed(1)}`).join(' ')
          return <path key={s.key} d={d} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
        })}
        <text x={pl} y={H - 5} className="chart-tick">{xmin}</text>
        <text x={W - pr} y={H - 5} className="chart-tick" textAnchor="end">{xmax}</text>
      </svg>
    )
  }
  return (
    <div className="chart">
      <div className="chart-title">{title}</div>
      {body}
      <div className="chart-legend">{series.map(s => <span key={s.key} className="cl"><i style={{ background: s.color }} />{s.name}</span>)}</div>
    </div>
  )
}

// 라벨링·학습 두 페이지가 공유하는 공통 헤더(제목 타이포·뒤로가기·하단 구분선 통일)
function PageHead({ title, back, right, flat }) {
  return (
    <div className={`page-head${flat ? ' flat' : ''}`}>
      <div className="page-head-l">
        {back && <button className="icon-back" onClick={back} aria-label="뒤로"><IcChevronLeft /></button>}
        <h2 className="page-title">{title}</h2>
      </div>
      {right && <div className="page-head-r">{right}</div>}
    </div>
  )
}

// 모델ID(YYYYMMDD_HHMMSS) → 사람이 읽는 날짜
function fmtId(s) {
  if (!s) return ''
  const m = String(s).match(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/)
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : String(s)
}

// 모델 버전 카드(기존/신규 나란히 비교). highlightNew=신규 부품 뱃지 강조
// 표시 항목은 딱 4가지: ①모델 ID ②생성 일시 ③학습 부품 뱃지 ④인식률(있으면). 중복 캡션 없음
function VerCard({ tag, cls, id, time, classes, map50, baseSet, highlightNew }) {
  const list = classes || []
  return (
    <div className={`verc ${cls}`}>
      <div className={`verc-tag ${cls}`}>{tag}</div>
      <div className="verc-id">{id ? <>#{id}</> : <span className="verc-none">모델 없음</span>}</div>
      <dl className="verc-meta">
        <div><dt>생성 일시</dt><dd>{time || '—'}</dd></div>
        {map50 != null && (
          <div><dt>인식률<span className="verc-qual">(자동라벨)</span></dt><dd className="verc-map">{map50}</dd></div>
        )}
      </dl>
      <div className="verc-parts">
        <span className="verc-parts-lbl">학습 부품 <span className="verc-parts-n">{list.length}종</span></span>
        <div className="verc-badges">
          {list.length === 0 && <span className="al-hint">—</span>}
          {list.slice(0, 14).map(c => (
            <span key={c} className={`cbadge${highlightNew && baseSet && !baseSet.has(c) ? ' new' : ''}`}>{c}</span>
          ))}
          {list.length > 14 && <span className="cbadge more">+{list.length - 14}</span>}
        </div>
      </div>
    </div>
  )
}

// 스코어 타일(모던 SaaS 위젯): 상단 제목·종수(좌) + 증감 뱃지(우), 중앙 큰 숫자(무채색), 하단 이전값(옅게).
// 하락 경고(≤-10%p)는 배경/숫자색이 아니라 카드 왼쪽 빨간 포인트선(inset)으로만 은은하게 표시.
function ScoreTile({ label, n, before, after, pctv, deltaEl, warnDown }) {
  const d = (before != null && after != null) ? Math.round((after - before) * 100) : null
  const bad = warnDown && d != null && d <= -10
  return (
    <div className={`score-tile${bad ? ' bad' : ''}`}>
      <div className="score-top">
        <div className="score-title">{label}{n != null && <span className="score-n">{n}종</span>}</div>
        {before != null && after != null && deltaEl(before, after)}
      </div>
      {after == null ? (                          /* 신규 모델 결과 자체가 없음 */
        <div className="score-na">비교 대상 없음 · 첫 배포</div>
      ) : (
        <>
          <div className="score-big">{pctv(after)}</div>
          {before != null && <div className="score-foot">이전 {pctv(before)}</div>}
        </>
      )}
    </div>
  )
}

// 하단 액션바 모델 액션. onApply(신규 적용) 있으면 주 CTA=분할버튼(적용 + ▾메뉴: 폐기·과거 롤백),
// 없으면(관리 모드) [과거 모델로 롤백 ▾] 단일 드롭다운. 항목 클릭=롤백 · ×=삭제 · 바깥 클릭 시 닫힘
function RollbackMenu({ models, servedId, onRollbackTo, onDeleteModel, onKeep, onApply, applied }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  useEffect(() => {                         // 팝업이 열려 있을 때만 바깥 클릭 감지 리스너 부착
    if (!open) return
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    const onKey = (ev) => { if (ev.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey) }
  }, [open])
  const recent = (models || []).slice(0, 10)     // 최근 10개 버전만
  return (
    <div className={`rbmenu${onApply ? ' split' : ''}`} ref={ref}>
      {onApply ? (
        // 주 CTA(분할버튼): 왼쪽=신규 적용, 오른쪽 ▾=다른 작업 메뉴
        <>
          <button className="act-btn train big split-main" onClick={onApply} disabled={applied}>
            {applied ? '✓ 서비스 적용됨' : '신규 모델 서비스 적용'}
          </button>
          <button className="act-btn train big split-caret" onClick={() => setOpen(o => !o)}
                  aria-haspopup="true" aria-expanded={open} aria-label="다른 작업 (과거 롤백 · 신규 폐기)">
            <IcChevronDown />
          </button>
        </>
      ) : (
        <button className="act-btn ghost rbmenu-btn" onClick={() => setOpen(o => !o)}
                aria-haspopup="true" aria-expanded={open}>
          과거 모델 조회 <IcChevronDown />
        </button>
      )}
      {open && (
        <div className="rbmenu-pop" role="menu">
          <div className="rbmenu-head">{onApply ? '다른 작업' : '저장된 모델 버전'}</div>
          {onKeep && (
            <button className="rbmenu-item keep" role="menuitem"
                    onClick={() => { setOpen(false); onKeep() }}>
              <span className="rbmenu-main">
                <span className="rbmenu-t">신규 학습 폐기</span>
                <span className="rbmenu-s">라벨 화면으로 돌아가기</span>
              </span>
            </button>
          )}
          {onApply && <div className="rbmenu-sec">과거 버전으로 롤백</div>}
          {recent.length === 0 && <div className="rbmenu-empty">등록된 과거 버전이 없습니다.</div>}
          {recent.map(m => {
            const active = m.is_active || m.model_id === servedId
            return (
              <div key={m.model_id} role="menuitem" tabIndex={0}
                   className={`rbmenu-item${active ? ' active' : ''}`}
                   onClick={() => {
                     if (active) return
                     setOpen(false); onRollbackTo(m.model_id)   // 확인은 인앱 모달(부모 askRollbackTo)에서
                   }}>
                <span className="rbmenu-main">
                  <span className="rbmenu-t">{m.time || fmtId(m.model_id)}</span>
                  <span className="rbmenu-s">
                    {m.n_classes}종
                    {m.gen_rate != null ? ` · 전체 인식률 ${Math.round(m.gen_rate * 100)}%` : ''}
                    {m.newp_rate != null ? ` · 신규 부품 ${Math.round(m.newp_rate * 100)}%` : ''}
                  </span>
                </span>
                {active
                  ? <span className="rbmenu-cur">현재</span>
                  : <button className="rbmenu-del" title="이 버전 삭제"
                            onClick={(e) => { e.stopPropagation(); onDeleteModel(m.model_id) }}><IcX /></button>}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// 부품 하나의 여러 테스트 프레임을 좌우 화살표로 넘겨보는 Before/After 슬라이더
// frames = [{ before: dataURI|null, after: dataURI|null }, ...] (같은 part의 여러 프레임)
function BaGroup({ part, kind, frames }) {
  const [idx, setIdx] = useState(0)          // 현재 보고 있는 프레임 인덱스
  const n = frames.length
  const i = Math.min(idx, n - 1)             // 프레임 수가 줄어도 인덱스 안전
  const cur = frames[i] || {}
  const go = (d) => setIdx(p => ((Math.min(p, n - 1) + d) % n + n) % n)   // 순환 이동(양끝 래핑)
  return (
    <div className="ba-group">
      <div className="ba-stage">
        {n > 1 && <button className="ba-nav prev" onClick={() => go(-1)} aria-label="이전 프레임"><IcChevronLeft /></button>}
        <div className="ba-pair big">
          <figure className="ba-fig">
            <figcaption>기존 모델</figcaption>
            {cur.before ? <img src={cur.before} alt={`${part} 기존 모델 검출`} /> : <div className="ba-none">기존 모델 없음</div>}
          </figure>
          <div className="ba-arrow" aria-hidden="true"><IcChevronRight /></div>
          <figure className="ba-fig after">
            <figcaption>신규 모델</figcaption>
            {cur.after ? <img src={cur.after} alt={`${part} 신규 모델 검출`} /> : <div className="ba-none">신규 모델 없음</div>}
          </figure>
        </div>
        {n > 1 && <button className="ba-nav next" onClick={() => go(1)} aria-label="다음 프레임"><IcChevronRight /></button>}
      </div>
      <div className="ba-legend">
        <span className="lg-item"><i className="lg-sw green" /> 정답 부품 검출</span>
        <span className="lg-item"><i className="lg-sw orange" /> 다른 부품으로 오검출</span>
      </div>
      {n > 1 && (
        <div className="ba-dots">
          {frames.map((_, k) => (
            <button key={k} className={`ba-dot${k === i ? ' on' : ''}`} onClick={() => setIdx(k)}
                    aria-label={`${k + 1}번째 프레임 보기`} />
          ))}
        </div>
      )}
    </div>
  )
}

// 부품 선택 커스텀 드롭다운(네이티브 select 대신 앱 톤 통일)
function PartSelect({ options, value, onChange }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  useEffect(() => {
    if (!open) return
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDoc); document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey) }
  }, [open])
  return (
    <div className="psel" ref={ref}>
      <button className="psel-btn" onClick={() => setOpen(o => !o)} aria-haspopup="listbox" aria-expanded={open}>
        <span>{options[value]}</span><IcChevronDown />
      </button>
      {open && (
        <div className="psel-pop" role="listbox">
          {options.map((o, i) => (
            <button key={o + '_' + i} role="option" aria-selected={i === value}
                    className={`psel-item${i === value ? ' on' : ''}`}
                    onClick={() => { onChange(i); setOpen(false) }}>
              <span>{o}</span>{i === value && <IcCheck />}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// 인앱 확인 모달(네이티브 window.confirm 대체). danger=파괴적 액션(빨강 버튼)
function ConfirmModal({ open, title, message, confirmLabel, danger, onConfirm, onCancel }) {
  useEffect(() => {
    if (!open) return
    const onKey = (e) => { if (e.key === 'Escape') onCancel() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onCancel])
  if (!open) return null
  return (
    <div className="cfm-overlay" onClick={onCancel}>
      <div className="cfm-box" role="dialog" aria-modal="true" onClick={e => e.stopPropagation()}>
        {title && <div className="cfm-title">{title}</div>}
        <div className="cfm-msg">{message}</div>
        <div className="cfm-act">
          <button className="act-btn ghost" onClick={onCancel}>취소</button>
          <button className={`act-btn ${danger ? 'stop' : 'train'}`} onClick={onConfirm} autoFocus>{confirmLabel || '확인'}</button>
        </div>
      </div>
    </div>
  )
}

function PartsApp() {
  const [folders, setFolders] = useState([])
  // F5 복구용: page·job·cmpJob·session을 세션스토리지에서 초기화(같은 탭 새로고침이면 있던 자리로 복원)
  const [session, setSession] = useState(() => sessionStorage.getItem('xr_session') || null)
  const [labeledMap, setLabeledMap] = useState({})   // {학습영상: {labels,frames}}
  const [trainedVideos, setTrainedVideos] = useState([])   // 기존 모델에 이미 학습된 영상
  const [picked, setPicked] = useState([])           // 선택한 부품(기본 = 선택 없음)
  const [epochs, setEpochs] = useState(100)
  const [job, setJob] = useState(() => sessionStorage.getItem('xr_job') || null)
  const [status, setStatus] = useState(null)
  const [page, setPage] = useState(() => sessionStorage.getItem('xr_page') || 'label')   // 'label' | 'training' | 'evaluate'
  const [cmpJob, setCmpJob] = useState(() => sessionStorage.getItem('xr_cmpJob') || null)
  const [cmp, setCmp] = useState(null)               // 신규↔기존 모델 비교(평가) 상태
  const [applied, setApplied] = useState(false)      // 신규 모델 서비스 적용 완료
  const [served, setServed] = useState(null)         // 현재 서비스(기존) 모델 정보
  const [servedLoaded, setServedLoaded] = useState(false)   // served fetch 완료 여부(기본선택 타이밍용)
  const didInitPick = useRef(false)                  // 학습됨 부품 기본선택을 최초 1회만 적용
  const [models, setModels] = useState([])           // 등록소 버전 목록(타임라인)
  const [rolledTo, setRolledTo] = useState(null)     // 롤백 완료된 model_id
  const [selVer, setSelVer] = useState(null)         // 타임라인에서 고른 비교 기준 버전(null=현재 서비스)
  const [baSel, setBaSel] = useState(0)              // 인식 결과 비교: 한 번에 보는 부품 인덱스
  const running = !!status?.running

  const loadTrain = useCallback(() => {   // 모달 열 때 폴더+라벨 현황 갱신
    fetch('/api/autolabel/folders').then(r => r.json()).then(setFolders).catch(() => {})
    fetch('/api/sam2/parts_sessions').then(r => r.json()).then(list => {
      const saved = localStorage.getItem('parts_session_v1')
      const found = list.find(s => s.session === saved) || list[0]   // 단일 영속 세션(부품별 저장소 합성) 자동 채택
      if (found) { setSession(found.session); setLabeledMap(found.videos || {}); setTrainedVideos(found.trained || []); localStorage.setItem('parts_session_v1', found.session) }
    }).catch(() => {})
  }, [])

  const folderOf = (video) => folders.find(f => f.videos.some(v => v.name === video))
  const items = Object.keys(labeledMap).map(video => {   // 라벨 1장 이상 생성된 부품만 학습 대상
    const f = folderOf(video)
    const teststem = f ? takeRoles(f.videos).test : video
    // test_srcs 는 부품경로 키로 넘긴다(같은 이름 test 영상이 다른 부품에 있어도 평가 대상 안 꼬임)
    const test = f && teststem ? `${f.folder}/${teststem}` : teststem
    return { video, part: f ? partOf(f.folder) : video, test,
             labels: labeledMap[video].labels, trained: trainedVideos.includes(video) }
  }).filter(it => it.labels > 0).sort((a, b) => a.part.localeCompare(b.part))
  const selected = items.filter(it => picked.includes(it.part))
  const allOn = items.length > 0 && selected.length === items.length
  const toggle = (p) => setPicked(c => c.includes(p) ? c.filter(x => x !== p) : [...c, p])
  const toggleAll = () => setPicked(allOn ? [] : items.map(it => it.part))
  // 이미 학습된(현재 서비스 모델 보유) 부품 = 학습됨. 배지 판정과 동일 기준. 망각 방지 위해 기본 선택 대상.
  const trainedPartList = () => items.filter(it => served ? (served.classes || []).includes(it.part) : it.trained).map(it => it.part)

  const openTrain = () => { loadTrain(); setPage('training') }
  const backToLabel = () => setPage('label')               // 학습은 계속 진행(폴링 유지), 라벨 화면으로 복귀
  const newRun = () => { setJob(null); setStatus(null); setCmpJob(null); setCmp(null); setApplied(false); setPicked(trainedPartList()); setPage('training') }

  const [confirmState, setConfirmState] = useState(null)   // 인앱 확인 모달 {title,message,confirmLabel,danger,onOk}
  const ask = (opts) => setConfirmState(opts)
  const onBackFromTrain = () => {                          // 부품 학습 뒤로가기: 학습 중이면 종료 확인
    if (running) ask({ title: '학습 종료', message: '학습을 종료하시겠습니까? 진행 중인 학습이 중단됩니다.',
                       confirmLabel: '종료', danger: true, onOk: () => { doCancel(); backToLabel() } })
    else backToLabel()
  }
  const askRollbackTo = (mid) => {                         // 과거 버전 롤백 확인(네이티브 confirm 대신 인앱 모달)
    const m = (models || []).find(x => x.model_id === mid)
    const when = (m && (m.time || fmtId(m.model_id))) || mid
    ask({ title: '과거 모델로 롤백', message: `이 버전(${when})으로 롤백할까요? 현재 서비스 모델이 됩니다.`,
          confirmLabel: '롤백', onOk: () => doRollbackTo(mid) })
  }

  const runTrain = async () => {
    const classes = selected.map(it => it.part)
    const tests = [...new Set(selected.map(it => it.test).filter(Boolean))]
    const r = await fetch('/api/sam2/multiclass', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session, epochs, test_srcs: tests, classes, augment: true })   // 배경 합성 증강 항상 적용
    }).then(x => x.json())
    if (r.error) { setJob('err'); setStatus({ error: r.error, running: false, log: [] }); return }
    setJob(r.job); setStatus({ stage: 'start', running: true, log: [] })   // 학습 시작 → 로그 뷰어로 전환
  }
  useEffect(() => {
    if (!job || job === 'err' || !status?.running) return
    const t = setInterval(async () => {
      const d = await fetch(`/api/sam2/status?job=${job}`).then(r => r.json())
      setStatus(d); if (!d.running) clearInterval(t)
    }, 1500)
    return () => clearInterval(t)
  }, [job, status?.running])

  const runCompare = (baseId) => {   // 비교 실행/재실행. baseId=기준(기존) 버전, 없으면 현재 서비스 모델
    setSelVer(baseId || null)
    setCmpJob(null); setCmp({ stage: 'compare', running: true })
    ;(async () => {
      const r = await fetch('/api/sam2/compare', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session, base_model_id: baseId || null })
      }).then(x => x.json()).catch(() => ({ error: 'compare 요청 실패' }))
      if (r.error) { setCmpJob('err'); setCmp({ error: r.error, running: false }); return }
      setCmpJob(r.job); setCmp({ stage: 'compare', running: true })
    })()
  }
  const triggerCompare = () => { if (!cmpJob) runCompare(null) }
  const goEvaluate = () => { setPage('evaluate'); triggerCompare() }
  const goManage = () => {   // 학습/비교 없이 모델 관리(롤백·삭제) 페이지로 바로 이동
    setPage('evaluate')
    fetch('/api/sam2/served').then(r => r.json()).then(d => setServed(d && !d.none ? d : null)).catch(() => {})
    fetch('/api/sam2/models').then(r => r.json()).then(d => setModels(d.models || [])).catch(() => {})
  }
  const onTimelineClick = (mid) => {   // 신규 학습 있으면 그 버전 기준 재비교, 없으면(관리 모드) 아래에 상세 표시
    if (status?.model_id) runCompare(mid)
    else setSelVer(mid)
  }
  const doDeleteModel = async (mid) => {   // 타임라인/히스토리에서 버전 삭제(현재 서비스 모델은 백엔드가 거부)
    if (!window.confirm(`모델 #${mid} 을(를) 삭제할까요? 되돌릴 수 없습니다.`)) return
    const r = await fetch('/api/sam2/delete_model', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: mid })
    }).then(x => x.json()).catch(() => ({ error: '삭제 실패' }))
    if (r.error) { window.alert(r.error); return }
    if (selVer === mid) setSelVer(null)
    fetch('/api/sam2/models').then(x => x.json()).then(d => setModels(d.models || [])).catch(() => {})
  }

  useEffect(() => {   // 비교 잡 폴링
    if (!cmpJob || cmpJob === 'err' || !cmp?.running) return
    const t = setInterval(async () => {
      const d = await fetch(`/api/sam2/status?job=${cmpJob}`).then(r => r.json())
      setCmp(d); if (!d.running) clearInterval(t)
    }, 1500)
    return () => clearInterval(t)
  }, [cmpJob, cmp?.running])

  // 현재 위치·잡을 세션스토리지에 저장 → F5(같은 탭 새로고침) 때 있던 자리로 복원
  useEffect(() => { sessionStorage.setItem('xr_page', page) }, [page])
  useEffect(() => { if (job && job !== 'err') sessionStorage.setItem('xr_job', job) }, [job])
  useEffect(() => { if (cmpJob && cmpJob !== 'err') sessionStorage.setItem('xr_cmpJob', cmpJob) }, [cmpJob])
  useEffect(() => { if (session) sessionStorage.setItem('xr_session', session) }, [session])

  // 마운트 시 복구: 세션스토리지로 되살린 page/job 에 실제 상태를 다시 붙인다(백엔드 JOBS 가 완료 잡도 보관).
  // 잡이 소실됐으면(백엔드 재시작 등) 처음(라벨)으로 안전 복귀. 세션 복구가 없으면 실행 중 잡만 재진입.
  useEffect(() => {
    const jb = sessionStorage.getItem('xr_job')
    const cj = sessionStorage.getItem('xr_cmpJob')
    if (page === 'training') {
      loadTrain()                                    // 학습 설정/결과 어느 쪽이든 부품목록 재로딩
      if (jb) fetch(`/api/sam2/status?job=${jb}`).then(r => r.json())
        .then(d => { if (d && !d.error) setStatus(d); else { setJob(null) } })
        .catch(() => { setJob(null) })
    } else if (page === 'evaluate') {
      fetch('/api/sam2/served').then(r => r.json()).then(d => setServed(d && !d.none ? d : null)).catch(() => {})
      fetch('/api/sam2/models').then(r => r.json()).then(d => setModels(d.models || [])).catch(() => {})
      if (cj) fetch(`/api/sam2/status?job=${cj}`).then(r => r.json())
        .then(d => { if (d && !d.error) setCmp(d) }).catch(() => {})
    } else {
      // 세션 복구 없음(새 탭 등): 다른 곳에서 실행 중인 잡이 있으면 그거라도 재진입
      fetch('/api/sam2/active').then(r => r.json()).then(a => {
        if (!a || !a.job || !a.running) return
        if (a.session) setSession(a.session)
        if (a.kind === 'multiclass') { setJob(a.job); setStatus(a); setPage('training'); loadTrain() }
        else if (a.kind === 'compare') { setCmpJob(a.job); setCmp(a); setPage('evaluate') }
      }).catch(() => {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadTrain])

  // 산입 카운터 부드럽게 세어 올리기: 백엔드 스캔이 순식간에 끝나 0→N으로 튀어도, 화면은 1→N 으로 climbing.
  const [ingShown, setIngShown] = useState(0)
  const ingShownRef = useRef(0)
  useEffect(() => { ingShownRef.current = ingShown }, [ingShown])
  useEffect(() => {
    if (status?.stage !== 'build') { ingShownRef.current = 0; setIngShown(0); return }
    const target = status?.ingest_done || 0
    const from = ingShownRef.current
    if (target <= from) { setIngShown(target); return }   // 후퇴/동일이면 즉시
    const dur = Math.min(1200, 250 + (target - from) * 4) // 개수 많을수록 조금 길게(최대 1.2s)
    const t0 = performance.now()
    let raf
    const tick = (t) => {
      const k = Math.min(1, (t - t0) / dur)
      const e = 1 - Math.pow(1 - k, 3)                    // easeOutCubic
      setIngShown(Math.round(from + (target - from) * e))
      if (k < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [status?.stage, status?.ingest_done])

  // 현재 서비스(기존) 모델 로드 — 라벨/학습 화면에서 '학습됨 vs 신규' 비교 기준
  const loadServed = useCallback(() => {
    fetch('/api/sam2/served').then(r => r.json())
      .then(d => { setServed(d && !d.none ? d : null); setServedLoaded(true) })
      .catch(() => setServedLoaded(true))
  }, [])
  useEffect(() => { loadServed() }, [loadServed])

  // 학습 대상 기본 선택: 이미 학습된(학습됨) 부품을 기본으로 체크(망각 방지). served·부품목록이 준비되면 최초 1회.
  useEffect(() => {
    // folders 로드 전엔 it.part 가 원본 영상명이라 served.classes 매칭이 안 됨 → folders 까지 준비된 뒤 1회 초기화
    if (didInitPick.current || !servedLoaded || items.length === 0 || folders.length === 0) return
    setPicked(trainedPartList())
    didInitPick.current = true
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servedLoaded, items, folders])
  // 평가 완료 시 버전 목록도 로드
  useEffect(() => {
    if (cmp?.stage !== 'done') return
    loadServed()
    fetch('/api/sam2/models').then(r => r.json()).then(d => setModels(d.models || [])).catch(() => {})
  }, [cmp?.stage, loadServed])

  const doRollbackTo = async (mid) => {   // 선택 버전으로 롤백(타임라인에서 직접 호출)
    if (!mid) return
    const r = await fetch('/api/sam2/rollback_to', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: mid })
    }).then(x => x.json()).catch(() => ({ error: '롤백 실패' }))
    if (!r.error) {
      setRolledTo(mid); setApplied(false)
      fetch('/api/sam2/served').then(x => x.json()).then(d => setServed(d && !d.none ? d : null)).catch(() => {})
      fetch('/api/sam2/models').then(x => x.json()).then(d => setModels(d.models || [])).catch(() => {})
    } else { window.alert(r.error) }
  }

  const doRollback = async () => {   // 신규 폐기 → 라벨 화면 복귀
    await fetch('/api/sam2/rollback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {})
    setJob(null); setStatus(null); setCmpJob(null); setCmp(null); setApplied(false); setPage('label')
  }
  const doApply = async () => {   // 신규 모델을 서비스에 적용
    const r = await fetch('/api/sam2/apply_model', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session })
    }).then(x => x.json()).catch(() => ({ error: 'apply failed' }))
    if (!r.error) setApplied(true)
  }
  const doCancel = async () => {   // 학습 중단
    if (!job || job === 'err') return
    await fetch('/api/sam2/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job }) }).catch(() => {})
  }

  const ep = status?.epoch || 0
  const tot = status?.total_epochs || epochs
  const ingD = status?.ingest_done || 0
  const ingT = status?.ingest_total || 0
  const stageText = {
    start: '학습 준비 중...',
    build: (status?.note && /증강|합성/.test(status.note))    // 산입 후 배경합성 증강 단계(수 분)면 라벨 구분
      ? '배경 합성 증강 생성 중...'
      : `학습 데이터 산입 중... (${Math.min(ingShown, ingT || ingShown)}/${ingT})`,
    train: `YOLO 모델 학습 중... (Epoch ${ep}/${tot})`,
    eval: `검출 평가 중... (${status?.eval_done || 0}/${status?.eval_total || 0})`,
    done: '학습 완료', cancelled: '학습 중단됨', error: '학습 오류',
  }
  const headText = status?.error ? '학습 오류' : (stageText[status?.stage] || '학습 중...')
  const pct = status?.stage === 'done' ? 100
    : status?.stage === 'eval' ? Math.round(90 + (status?.eval_frac || 0) * 10)     // 평가(프레임) 90→100
    : status?.stage === 'train' ? Math.round(10 + (status?.train_frac || 0) * 80)   // 학습(배치) 10→90
    : status?.stage === 'build' ? (ingT ? Math.round((Math.min(ingShown, ingT) / ingT) * 10) : 0)   // 산입 0→10
    : 0

  // ── 학습 완료 후: 평가(compare) 단계 진행/표시 ──
  const trainDone = status?.stage === 'done' && !status?.error
  const cmpDone = cmp?.stage === 'done'
  const cmpPct = cmpDone ? 100 : Math.round((cmp?.compare_frac || 0) * 100)
  const cmpTitle = cmpDone ? '모델 평가 완료'
    : (cmp?.error ? '평가 오류' : `모델 평가 중... (${cmp?.compare_done || 0}/${cmp?.compare_total || 0})`)
  const pctv = (x) => x == null ? '—' : `${Math.round(x * 100)}%`
  const deltaEl = (before, after) => {
    if (before == null || after == null) return <span className="delta flat">—</span>
    const d = Math.round((after - before) * 100)
    if (d > 0) return <span className="delta up">↑ {d}%p</span>
    if (d < 0) return <span className="delta down">↓ {Math.abs(d)}%p</span>
    return <span className="delta flat">±0%p</span>
  }
  // 평가 화면용 파생값
  const newClasses = status?.per_class ? Object.keys(status.per_class)
    : [...new Set((cmp?.rows || []).map(r => r.part))]
  const baseModel = (selVer && models.find(m => m.model_id === selVer)) || served   // 비교 기준(선택 or 현재 서비스)
  const baseSet = new Set(baseModel?.classes || [])
  // 검출 샘플: 백엔드가 부품(part)당 여러 프레임을 평평한 리스트로 줌 → part로 묶어 부품별 슬라이더 구성
  const baGroups = (() => {
    const raw = (cmp?.samples || []).filter(s => s && (s.after || s.before || s.img))
    const byPart = new Map()
    for (const s of raw) {
      const key = s.part || '기타'
      if (!byPart.has(key)) byPart.set(key, { part: key, kind: s.kind || 'base', frames: [] })
      const g = byPart.get(key)
      if (s.kind === 'new') g.kind = 'new'          // 한 프레임이라도 신규 부품이면 신규로 표기
      g.frames.push({ before: s.before || null, after: s.after || s.img || null })   // img는 구계약 하위호환
    }
    return [...byPart.values()]
  })()

  return (
    <div>
      {page === 'label' ? (
        <>
          <PageHead title="부품 학습 데이터 생성" flat
                    right={<button className="icon-back" onClick={openTrain} title="학습 설정으로 이동" aria-label="학습 설정"><IcChevronRight /></button>} />
          <AutoLabelView />
        </>
      ) : page === 'training' ? (
        // ===== 2단계: 학습 (설정 → 진행 → 결과 요약) =====
        <div className="train-page">
          <PageHead
            title="부품 학습"
            back={onBackFromTrain}
            right={<>
              {!job && <button className="icon-back" onClick={goManage} title="모델관리" aria-label="모델관리"><IcChevronRight /></button>}
              {trainDone && <button className="icon-back" onClick={goEvaluate} title="모델 평가 · 적용" aria-label="다음 단계: 모델 평가·적용"><IcChevronRight /></button>}
            </>}
          />

          <div className="train-split">
            {/* 좌측: 학습 대상 선택 폼 */}
            <aside className="train-config">
              <div className="tc-head">
                {items.length > 0 && !trainDone && status?.stage !== 'cancelled' &&
                  <input type="checkbox" className="part-sel" checked={allOn} disabled={running}
                         ref={el => { if (el) el.indeterminate = selected.length > 0 && !allOn }}
                         onChange={toggleAll} title="전체 선택 / 해제" aria-label="학습 대상 전체 선택" />}
                <b className="tc-head-title">학습 대상 선택</b>
                <span className="al-hint" style={{ marginLeft: 'auto' }}>{selected.length}/{items.length}</span>
              </div>
              <div className="tc-list">
                {items.length === 0
                  ? <p className="al-hint">라벨 생성된 부품이 없습니다. 라벨 화면에서 먼저 부품을 탭·라벨 생성하세요.</p>
                  : items.map(it => {
                      const on = picked.includes(it.part)
                      const inModel = served ? baseSet.has(it.part) : it.trained   // 현재 서비스 모델 보유 여부
                      return (
                        <button key={it.part} className={`tc-item${on ? ' on' : ''}`}
                                onClick={() => toggle(it.part)} disabled={running || trainDone || status?.stage === 'cancelled'}>
                          <span className="tc-check" aria-hidden="true" />
                          <span className="tc-name">{it.part}</span>
                          {inModel ? <span className="tchip-badge">학습됨</span> : <span className="tchip-badge new">신규</span>}
                        </button>
                      )
                    })}
              </div>
            </aside>

            {/* 우측: 실시간 로그 + 결과 */}
            <section className="train-monitor">
              {job && (
                <div className="prog-head">
                  <CircularProgress pct={pct} done={trainDone} />
                  <div className="prog-head-info">
                    <b className="prog-head-title">{headText}</b>
                    {running && <span className="prog-sub"><span className="spinner" /> 진행 중...</span>}
                  </div>
                  {running && <button className="act-btn stop" onClick={doCancel} style={{ marginLeft: 'auto' }}>■ 학습 중단</button>}
                  {(trainDone || status?.stage === 'cancelled') && <button className="act-btn ghost" onClick={newRun} style={{ marginLeft: 'auto' }}>↻ 새 학습</button>}
                </div>
              )}
              {!job && (
                // 학습 시작 전: 에포크 입력창 + 시작 버튼
                <div className="train-setup-bar">
                  <span className="al-hint">Epoch</span>
                  <input className="ep-in" type="number" min={1} value={epochs}
                         onChange={(e) => setEpochs(Math.max(1, Math.floor(+e.target.value) || 1))} />
                  <button className="act-btn train" onClick={runTrain} disabled={selected.length === 0}>학습 시작</button>
                </div>
              )}
              {job && status?.curve?.length > 0 && (
                <div className="ev2-card">
                  <h4 className="ev2-h">학습 곡선</h4>
                  <div className="charts">
                    <MiniLineChart title="Loss" data={status.curve}
                      series={[{ key: 'box', name: 'box_loss', color: '#ef4444' }, { key: 'cls', name: 'cls_loss', color: '#f59e0b' }, { key: 'dfl', name: 'dfl_loss', color: '#8b5cf6' }]} />
                    <MiniLineChart title="mAP" data={status.curve}
                      series={[{ key: 'map50', name: 'mAP50', color: '#10b981' }, { key: 'map5095', name: 'mAP50-95', color: '#06b6d4' }]} />
                  </div>
                </div>
              )}
              {job && <LogConsole log={status?.log} compact={running} />}
              {status?.error && <div className="reco-banner rollback"><IcWarn /><span>학습 오류: {status.error}</span></div>}
              {status?.stage === 'cancelled' && <div className="reco-banner review"><IcWarn /><span>학습이 중단되었습니다.</span></div>}
              {trainDone && (
                <div className="train-result">
                  {/* 결과 요약 카드 */}
                  <div className="ev2-card">
                    <h4 className="ev2-h">학습 결과 요약</h4>
                    <div className="result-cards">
                      <div className="rcard"><span>학습률(산입률)</span><b>{status?.learn_rate != null ? `${status.learn_rate}%` : '—'}</b></div>
                      <div className="rcard"><span>Epoch</span><b>{tot}/{tot}</b></div>
                      <div className="rcard"><span>학습셋</span>
                        {status?.n_images != null ? (<>
                          <b>{status.n_images + (status.n_augmented || 0)}장</b>
                          {status.n_augmented ? <small>(원본 {status.n_images}장 + 증강 {status.n_augmented}장)</small> : null}
                        </>) : <b>—</b>}
                      </div>
                      <div className="rcard"><span>데이터 종류</span><b>{status?.n_classes ?? '—'}종</b></div>
                    </div>
                  </div>
                  {/* 검출 평가 카드 삭제(사용자 요청). 인식 결과는 '모델 평가·적용' 화면의 인식 결과 비교에서 확인 */}
                </div>
              )}
            </section>
          </div>
        </div>
      ) : (
        // ===== 3단계: 모델 평가 · 적용 (버전 비교 → 서비스 적용 / 선택형 롤백) =====
        <div className="ev2">
          <PageHead title={cmpDone ? '모델 평가' : cmp?.running ? '모델 평가 중' : '모델관리'} back={() => setPage('training')}
                    right={(!cmp?.running && !cmp?.error) ? (
                      <div className="ev2-head-actions">
                        <RollbackMenu models={models} servedId={served?.model_id}
                                      onRollbackTo={askRollbackTo} onDeleteModel={doDeleteModel}
                                      onKeep={null} onApply={null} applied={applied} />
                        {cmpDone && cmp.recommend && !applied && (   // 과거 조회 | 기존 유지 | 신규 적용 나란히
                          <>
                            <button className={`act-btn ${cmp.recommend.level === 'rollback' ? 'danger' : 'ghost'}`} onClick={doRollback}>기존 모델 유지</button>
                            <button className={`act-btn ${cmp.recommend.level === 'rollback' ? 'ghost' : 'train'}`} onClick={doApply}>신규 모델 적용</button>
                          </>
                        )}
                        {cmpDone && applied && <span className="ok-flash"><IcCheck /> 적용됨</span>}
                      </div>
                    ) : null} />
          {cmp?.running && (
            <div className="ev2-progress">
              <CircularProgress pct={cmpPct} done={false} />
              <div className="prog-head-info">
                <b className="prog-head-title">{cmpTitle}</b>
                <span className="prog-sub"><span className="spinner" /> 신규 모델을 기존 모델과 비교 중...</span>
              </div>
            </div>
          )}
          {cmp?.error && <div className="ev2-body"><div className="reco-banner rollback"><IcWarn /><span>모델 평가 오류: {cmp.error}</span></div></div>}
          {!cmp?.running && !cmp?.error && (
            <>
              <div className="ev2-body">
                {cmpDone ? (
                  <>
                    {/* 1) 판정 히어로 — 권장 결정 + 이유 + 주 액션(적용/폐기)을 한곳에 */}
                    {cmp.recommend && (
                      <section className={`verdict ${cmp.recommend.level}`}>
                        <div className="verdict-badge">
                          {cmp.recommend.level === 'apply' ? <IcCheck /> : <IcWarn />}
                          <span>{cmp.recommend.level === 'apply' ? '신규 모델 적용 권장'
                            : cmp.recommend.level === 'rollback' ? '기존 모델 유지 권장'
                            : '검토 후 결정'}</span>
                        </div>
                        <p className="verdict-msg">{cmp.recommend.msg}</p>
                      </section>
                    )}

                    {/* 2) 스코어보드 — 판정을 뒷받침하는 인식률 2종(기존 유지 / 신규 학습) */}
                    <section className="scoreboard">
                      <ScoreTile label="기존 부품 인식" n={cmp.gen?.n ?? 0}
                                 before={cmp.gen?.before} after={cmp.gen?.after} pctv={pctv} deltaEl={deltaEl} warnDown />
                      <ScoreTile label="신규 부품 인식" n={cmp.newp?.n ?? 0}
                                 before={cmp.newp?.before} after={cmp.newp?.after} pctv={pctv} deltaEl={deltaEl} />
                    </section>

                    {/* 3) 눈으로 확인 — 실제 프레임 검출 비교(가장 신뢰되는 근거) */}
                    {baGroups.length > 0 && (() => {
                      const gi = baSel < baGroups.length ? baSel : 0
                      const g = baGroups[gi]
                      return (
                        <section className="ev2-card ba-primary">
                          <div className="ev2-h-row">
                            <h4 className="ev2-h">모델 결과 비교</h4>
                            {baGroups.length > 1 && (
                              <PartSelect options={baGroups.map(x => x.part)} value={gi} onChange={setBaSel} />
                            )}
                          </div>
                          <BaGroup key={g.part + '_' + gi} part={g.part} kind={g.kind} frames={g.frames} />
                        </section>
                      )
                    })()}
                  </>
                ) : (
                  <>
                    {/* 관리 모드(학습 없이 진입): 현재 서비스 모델 카드 + 선택 버전 상세 */}
                    <section className="served-card">
                      <div className="served-top">
                        <span className="served-lbl">현재 서비스 모델</span>
                        {rolledTo && <span className="served-flash"><IcCheck /> 롤백 완료</span>}
                      </div>
                      {served ? (
                        <div className="served-meta">
                          {(served.n_classes ?? (served.classes || []).length)}종
                          {served.gen_rate != null ? ` · 전체 인식률 ${Math.round(served.gen_rate * 100)}%` : ''}
                          {served.newp_rate != null ? ` · 신규 인식률 ${Math.round(served.newp_rate * 100)}%` : ''}
                        </div>
                      ) : (
                        <div className="served-none">서비스 중인 모델이 없습니다. 먼저 학습을 진행하세요.</div>
                      )}
                    </section>

                    {selVer && (() => {
                      const sel = models.find(m => m.model_id === selVer)
                      if (!sel) return null
                      return (
                        <section className="ev2-card">
                          <h4 className="ev2-h">선택 버전 상세 <span className="al-hint">(현재 서비스 모델과 비교)</span></h4>
                          <div className="verc-grid">
                            <VerCard tag="현재 서비스 모델" cls="base" id={served?.model_id}
                                     time={served?.time || served?.applied}
                                     classes={served?.classes || []} map50={served?.map50} baseSet={new Set()} />
                            <div className="verc-vs" aria-hidden="true"><IcChevronRight /></div>
                            <VerCard tag="선택한 버전" cls="new" id={sel.model_id}
                                     time={sel.time || fmtId(sel.model_id)}
                                     classes={sel.classes || []} map50={sel.map50}
                                     baseSet={new Set(served?.classes || [])} highlightNew />
                          </div>
                          <div className="verdetail-act">
                            {sel.is_active
                              ? <span className="ok-flash"><IcCheck /> 현재 서비스 중인 버전입니다</span>
                              : <button className="act-btn train" onClick={() => askRollbackTo(selVer)}>이 버전으로 롤백</button>}
                          </div>
                        </section>
                      )
                    })()}
                  </>
                )}
              </div>
            </>
          )}

        </div>
      )}
      <ConfirmModal open={!!confirmState} {...(confirmState || {})}
                    onCancel={() => setConfirmState(null)}
                    onConfirm={() => { const ok = confirmState?.onOk; setConfirmState(null); if (ok) ok() }} />
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
                  onClick={() => setView('autolabel')}>부품 학습 데이터 생성</button>
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
            : view === 'autolabel' ? <PartsApp />
            : <MethodView id={view} />}
        </div>
      </main>
    </div>
  )
}
