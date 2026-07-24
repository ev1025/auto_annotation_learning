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

function Callout({ children }) {
  return <div className="callout">{children}</div>
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

  return (
    <div>
      <p className="fn">평가셋 전체를 순서대로 탐색합니다. 신뢰도 기준 = 오토라벨 채택 조건(conf 0.6) 고정.</p>
      <div className="nav-row">
        <button onClick={() => load(idx - 1)} disabled={loading}>← 이전</button>
        <button onClick={() => load(idx + 1)} disabled={loading}>다음 →</button>
        <button onClick={() => load(Math.floor(Math.random() * total))} disabled={loading}>랜덤</button>
        <input type="range" min="0" max={total - 1} value={idx}
               onChange={e => setIdx(Number(e.target.value))}
               onMouseUp={e => load(Number(e.target.value))}
               onTouchEnd={e => load(Number(e.target.value))} />
        <span className="idx-label">{idx + 1} / {total}</span>
      </div>
      {data && (
        <>
          <div className="pair">
            <figure>
              <figcaption>모델이 만든 바운딩박스</figcaption>
              <img src={data.pred} alt="모델 예측" style={loading ? { opacity: 0.5 } : {}} />
            </figure>
            <figure>
              <figcaption>정답 라벨</figcaption>
              <img src={data.gt} alt="정답 라벨" style={loading ? { opacity: 0.5 } : {}} />
            </figure>
          </div>
          <p className="fn">{data.note}</p>
        </>
      )}
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

      <h3 className="section-h">실험 순서</h3>
      {m.ordered
        ? <ol>{m.bullets.map((b, i) => <li key={i}>{md(b)}</li>)}</ol>
        : <ul>{m.bullets.map((b, i) => <li key={i}>{md(b)}</li>)}</ul>}

      {m.code?.length > 0 && <>
        <h3 className="section-h">실제 코드</h3>
        {m.code.map((sn, i) => (
          <div className="snippet" key={i}>
            <div className="snip-head">
              <code className="snip-file">{sn.file}</code>
              <span className="snip-note">{sn.note}</span>
            </div>
            <pre><code>{sn.src}</code></pre>
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
      <p className="verdict">{m.metrics.summary}</p>
      <MetricsTable metrics={m.metrics} />
    </div>
  )
}

function ExtraView() {
  const [cats, setCats] = useState({})
  const [cat, setCat] = useState(null)
  const [topic, setTopic] = useState(null)
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    fetch('/api/experiments').then(r => r.json()).then(d => {
      setCats(d)
      const c = Object.keys(d)[0]
      setCat(c); setTopic(d[c][0])
    })
  }, [])

  useEffect(() => {
    if (!cat || !topic) return
    setMetrics(null)
    fetch(`/api/experiment?cat=${encodeURIComponent(cat)}&topic=${encodeURIComponent(topic)}`)
      .then(r => r.json()).then(setMetrics)
  }, [cat, topic])

  return (
    <div>
      <h2>기타 실험 결과</h2>
      <div className="chips">
        {Object.keys(cats).map(c => (
          <button key={c} className={c === cat ? 'chip on' : 'chip'}
                  onClick={() => { setCat(c); setTopic(cats[c][0]) }}>{c}</button>
        ))}
      </div>
      <div className="chips">
        {(cats[cat] || []).map(t => (
          <button key={t} className={t === topic ? 'chip on' : 'chip'}
                  onClick={() => setTopic(t)}>{t}</button>
        ))}
      </div>
      {metrics && <>
        <Callout>{metrics.summary}</Callout>
        <MetricsTable metrics={metrics} />
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
  const [view, setView] = useState('m1')
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
          <p>학습 라벨을 자동 생성하는<br />7가지 방법의 실험 기록</p>
        </div>
        <nav>
          <div className="nav-title">시도 방법 (번호 = 시도 순서)</div>
          {methods.map(m => (
            <button key={m.id} className={view === m.id ? 'nav-item on' : 'nav-item'}
                    onClick={() => setView(m.id)}>
              <span>{m.no}. {m.title}</span>
              <span className={`mini-badge ${m.badge}`}>{badgeText[m.badge]}</span>
            </button>
          ))}
          <div className="nav-title">그 외</div>
          <button className={view === 'extra' ? 'nav-item on' : 'nav-item'}
                  onClick={() => setView('extra')}>기타 실험 결과</button>
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
          {view === 'extra' ? <ExtraView />
            : view === 'glossary' ? <GlossaryView />
            : <MethodView id={view} />}
        </div>
      </main>
    </div>
  )
}
