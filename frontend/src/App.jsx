import { useEffect, useState, useCallback, useRef, useMemo } from 'react'

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
// 탭·폼용 아이콘
const IcPlus = () => SVG(<><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></>)
const IcList = () => SVG(<><line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" /><line x1="8" y1="18" x2="21" y2="18" /><line x1="3" y1="6" x2="3.01" y2="6" /><line x1="3" y1="12" x2="3.01" y2="12" /><line x1="3" y1="18" x2="3.01" y2="18" /></>)
const IcLayers = () => SVG(<><polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" /></>)
const IcCube = () => SVG(<><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /><polyline points="3.27 6.96 12 12.01 20.73 6.96" /><line x1="12" y1="22.08" x2="12" y2="12" /></>)
const IcVideo = () => SVG(<><polygon points="23 7 16 12 23 17 23 7" /><rect x="1" y="5" width="15" height="14" rx="2" ry="2" /></>)
const IcCamera = () => SVG(<><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" /><circle cx="12" cy="13" r="4" /></>)
const IcPencil = () => SVG(<><path d="M12 20h9" /><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" /></>)
const IcTrash = () => SVG(<><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></>)
const IcWarn = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
       strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block', flexShrink: 0 }}>
    <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
)
// **굵게** 마커를 <b>로 변환
// data/bell412/<그룹>/<부품>/videos → 부품 파싱. 폴더=부품, 폴더 안 영상=그 부품의 train/test 테이크
const partOf = (rel) => { const p = rel.replace(/\/videos$/, '').split('/'); return p.length > 2 ? p.slice(2).join('/') : (p[1] || rel) }
// 부품 폴더의 테이크 → {train:[...], test} : test* 있으면 그게 테스트, 없고 2개+면 끝수 최대("2")가 테스트, 단일이면 학습영상으로 테스트

// 부품 라벨링(SAM2): 데이터셋(그룹)→부품(폴더) 순회. 부품마다 학습 테이크 탭 → 입력 마스크 확인 → 라벨 생성 → 다음 부품
// onPrep: 전체 프레임 미리 컷 진행상황을 화면 제목줄(PageHead)로 올려 보낸다.
// 예전에는 색상 범례 줄 오른쪽에 끼워 넣어 배경작업 상태가 범례처럼 보였다.
function AutoLabelView({ onPrep, active }) {
  const [folders, setFolders] = useState([])        // [{folder,label,videos:[{name,count,ready}]}]
  const [partIdx, setPartIdx] = useState(0)         // 부품(폴더) 인덱스
  const [takeIdx, setTakeIdx] = useState(0)         // (구) 학습 테이크 인덱스
  const [selVideo, setSelVideo] = useState(null)    // 선택해서 보고 있는 영상(없으면 기본=첫 학습영상)
  const [confirmDelVid, setConfirmDelVid] = useState(null)   // 삭제 확인 중인 영상 이름
  const [alertMsg, setAlertMsg] = useState(null)             // 인앱 알림(네이티브 alert 대체)
  // 프레임 원본 크기(가로세로 비율). 부품마다 세로/가로 영상이 섞여 있어 박스 비율을 이걸로 맞춘다.
  const [natSize, setNatSize] = useState(null)
  const [showVideoPick, setShowVideoPick] = useState(false)  // 영상 선택 모달(라벨검수식)
  const [count, setCount] = useState(0)             // 현재 테이크 프레임 수
  const [idx, setIdx] = useState(0)                 // 현재 프레임
  const [ptsBySrc, setPtsBySrc] = useState(() => {  // { 영상: { 프레임: [{rx,ry,lab}] } } 영상별 보관
    try { return JSON.parse(localStorage.getItem('autolabel_shots_v1') || '{}') } catch { return {} }
  })
  const [preparing, setPreparing] = useState(false)
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
  // 서버 영속 참조샷(shots.json) = 단일 소스. localStorage 는 오프라인 폴백일 뿐이다.
  // 로드 성공 시 통째 교체해서 stale 키를 폐기하고 서버 기준으로 맞춘다.
  const loadShots = useCallback(() => {
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
  useEffect(() => { loadShots() }, [loadShots])
  const [showReview, setShowReview] = useState(false)   // 라벨 검수 모달
  const [reviewFrames, setReviewFrames] = useState([])
  const [zoomFrame, setZoomFrame] = useState(null)      // 검수에서 클릭해 확대한 프레임
  const [selParts, setSelParts] = useState(() => new Set())   // 일괄 라벨 생성 대상 부품(체크박스 선택)

  const [session, setSession] = useState(() => localStorage.getItem('parts_session_v1') || null)
  const [labeledMap, setLabeledMap] = useState({})  // {영상: {labels,frames}} 현재 세션에 라벨된 테이크
  const [labelJob, setLabelJob] = useState(null)
  const [labelStatus, setLabelStatus] = useState(null)

  const preppedRef = useRef(new Set())              // 그룹별 '전체 프레임 미리 컷' 1회만

  const running = !!labelStatus?.running

  const isPartFolder = (rel) => rel.replace(/\/videos$/, '').split('/').length >= 3   // bell412/<컨테이너>/<부품> = 중첩 = 부품
  const nestedParts = folders.filter(f => isPartFolder(f.folder))
  const partFolders = nestedParts.length ? nestedParts : folders          // 부품 폴더들(중첩 없으면 전체)
  const curPartFolder = partFolders[partIdx]
  const folderVideos = curPartFolder?.videos || []                       // 이 부품의 테이크들
  const src = (selVideo && folderVideos.some(v => v.name === selVideo))   // 선택한 영상(없으면 첫 영상)
    ? selVideo : (folderVideos[0]?.name || null)
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
    // 아직 안 잘린 영상만 센다(이미 캐시된 것은 건너뛰므로 진행률에 넣으면 안 됨)
    const todo = pfs.flatMap(pf => pf.videos.filter(v => !v.ready).map(v => ({ pf, v })))
    let done = 0
    for (const { pf, v } of todo) {
      onPrep?.({ done, total: todo.length, name: v.name })
      const k = v.key || `${pf.folder}/${v.name}`
      try { const d = await fetch(`/api/autolabel/prepare?src=${encodeURIComponent(k)}`).then(r => r.json()); markReady(v.name, d.count || 0) } catch { /* skip */ }
      done += 1
    }
    onPrep?.(null)
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
  useEffect(() => {                       // 탭을 다시 열 때마다 갱신(그 사이 등록한 부품 반영)
    if (active) { loadFolders(); loadSessions() }
  }, [active, loadFolders, loadSessions])

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
  const pfTrain = (pf) => (pf.videos || []).map(v => v.name)             // 부품 폴더의 영상들(전부 학습용)
  const pfStatus = (pf) => {                                             // 부품 상태
    const tr = pfTrain(pf)
    if (tr.some(isLabeled)) return 'done'
    if (tr.some(hasTaps)) return 'tapped'
    return 'todo'
  }
  const labeledParts = partFolders.filter(pf => pfTrain(pf).some(isLabeled)).map(pf => partOf(pf.folder))  // 라벨된 부품(클래스)명
  const nLabeled = labeledParts.length
  const curShots = buildShots(src)                 // 현재 영상의 유효 참조샷
  // 같은 부품의 다른 영상에 찍어둔 참조샷(개수만). 영상을 바꿔도 찍어둔 게 보이게 한다
  const otherShots = folderVideos.map(v => v.name).filter(n => n !== src)
    .map(n => ({ name: n, n: buildShots(n).length })).filter(o => o.n > 0)
  // 라벨 생성 대상 = 이 부품에서 참조샷이 있는 모든 영상(현재 영상만 하지 않는다)
  const labelTargets = folderVideos.map(v => v.name)
    .map(n => ({ name: n, shots: buildShots(n) })).filter(x => x.shots.length > 0)
  const shotFrames = Object.keys(pts).map(Number).filter(i => (pts[i] || []).length).sort((a, b) => a - b)  // 이 영상에서 탭한 프레임들
  // 부품 선택(체크박스) → 선택한 부품 중 참조샷 있는 것들 일괄 라벨 생성 대상
  const toggleSelPart = (folder) => setSelParts(s => { const n = new Set(s); n.has(folder) ? n.delete(folder) : n.add(folder); return n })
  const allPartsSelected = partFolders.length > 0 && partFolders.every(pf => selParts.has(pf.folder))
  const toggleAllParts = () => setSelParts(allPartsSelected ? new Set() : new Set(partFolders.map(pf => pf.folder)))
  const selItems = partFolders.filter(pf => selParts.has(pf.folder))
    .map(pf => ({ pf, stem: pfTrain(pf)[0] })).filter(x => x.stem)
    .filter(x => buildShots(x.stem).length > 0).map(x => ({ video: `${x.pf.folder}/${x.stem}`, shots: buildShots(x.stem) }))
  const batchMode = selParts.size > 0
  const goShot = (i) => { setIdx(i); setActiveShot(shotKey(src, i)) }   // 그 프레임으로 이동 + (캐시 있으면) 마스크 표시
  const deleteShotFrame = async (i) => {            // 그 프레임 탭 삭제(서버 shots.json 까지 즉시 반영)
    dropMask(src, i)
    setPtsBySrc(prev => { const vp = { ...(prev[src] || {}) }; delete vp[i]; return { ...prev, [src]: vp } })
    setActiveShot(c => c === shotKey(src, i) ? null : c)
    // 화면만 지우면 새로고침 때 서버 값으로 되살아난다.
    // 서버 반영이 실패하면 조용히 넘기지 않고 알린다(화면과 서버가 갈리는 게 더 나쁘다)
    const r = await fetch('/api/sam2/delete_shot', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video: keyOf(src), frame: i })
    }).then(x => x.json()).catch(() => ({ error: '요청 실패' }))
    if (r.error) {
      setAlertMsg(`참조샷 삭제를 서버에 반영하지 못했습니다: ${r.error}`)
      loadShots()          // 서버 값으로 화면을 되돌린다
      return
    }
    if (r.removed_labels) {   // 참조샷이 0개가 되면 서버가 그 영상 라벨까지 지운다
      setAlertMsg(`참조샷이 없어져 ${src} 의 라벨 ${r.removed_labels}건도 함께 삭제했습니다.`)
      loadFolders(); loadSessions()
      fetch('/api/sam2/labeled_parts').then(x => x.json())
        .then(x => setLabeledAnywhere(new Set(x.parts || []))).catch(() => {})
    }
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
    if (!labelTargets.length) return
    // 이 부품에서 참조샷을 찍은 영상 전부를 한 번에 처리한다(예전에는 현재 영상만 했다)
    const single = labelTargets.length === 1
    const r = single
      ? await fetch('/api/sam2/parts_label', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session, video: keyOf(labelTargets[0].name), shots: labelTargets[0].shots })
        }).then(x => x.json())
      : await fetch('/api/sam2/parts_label_batch', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session, items: labelTargets.map(x => ({ video: keyOf(x.name), shots: x.shots })) })
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
          openReview()   // 라벨 생성이 끝나면 바로 검수 화면을 띄운다(눈으로 확인하는 것이 다음 단계라서)
        }
      }
    }, 1200)
    return () => clearInterval(t)
  }, [labelJob, labelStatus?.running])

  const goPart = (i) => { setPartIdx(Math.max(0, Math.min(i, partFolders.length - 1))); setTakeIdx(0); setSelVideo(null) }

  const activeMask = masks[shotKey(src, idx)] || null   // 현재 프레임의 마스크만 표시(프레임 바뀌면 자동으로 사라짐)
  const partName = curPartFolder ? partOf(curPartFolder.folder) : ''
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
  // 영상 삭제: 확인·오류 모두 인앱 모달(네이티브 dialog 안 씀). 프레임·라벨도 함께 정리된다
  const deleteVideo = async (name) => {
    if (running) return
    const key = keyOf(name)
    const r = await fetch('/api/sam2/delete_video', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ src: key })
    }).then(x => x.json()).catch(() => ({ error: '삭제 요청 실패' }))
    if (r.error) { setAlertMsg(r.error); return }
    setPtsBySrc(prev => { const n = { ...prev }; delete n[name]; return n })   // 이 영상 탭 정리
    if (selVideo === name) setSelVideo(null)
    loadFolders(); loadSessions()
    fetch('/api/sam2/labeled_parts').then(x => x.json()).then(x => setLabeledAnywhere(new Set(x.parts || []))).catch(() => {})
  }

  return (
    // 카드 높이를 flex 로 채운다(매직넘버 calc(100vh - N) 제거 — 탭 바 높이가 바뀌면 어긋났다)
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      {!src ? <p className="al-hint">부품 폴더가 없습니다. (data/bell412/parts/&lt;부품&gt;/videos)</p> : (
        // flex-basis 0 으로 높이를 확정해야 자식의 height:100%(프레임 이미지)가 부모 기준으로 풀린다.
        // basis auto 면 이미지가 원본 높이로 커져 페이지가 스크롤된다.
        <div className="al-view" style={{ display: 'flex', flexDirection: 'column', flex: '1 1 0', minHeight: 0, overflow: 'hidden' }}>
          {/* 상단 범례 제거 — 범례는 이미지·입력마스크 아래로 이동 */}
          {/* 상단 액션 버튼줄 */}
          <div className="al-controls" style={{ flexShrink: 0 }}>
            {folderVideos.length >= 1 && (
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
            {/* 라벨 생성 진행: 스피너 + 개수(tqdm 처럼). 전파는 프레임 수만큼 돌기 때문에
                숫자가 없으면 멈춘 것처럼 보인다. 배치 생성이면 부품 진행(3/5)도 함께 보여준다. */}
            {labelStatus && !labelStatus.error && labelStatus.running && (
              <span className="al-prog">
                <span className="spinner" />
                {labelStatus.prog_total > 0
                  ? `${labelStatus.prog_step === 'write' ? '라벨 쓰기 ' : ''}`
                    + `${labelStatus.prog_done}/${labelStatus.prog_total} `
                    + `(${Math.round((labelStatus.prog_done / labelStatus.prog_total) * 100)}%)`
                  : '준비 중'}
                {labelStatus.total > 1 && ` · 부품 ${Math.min((labelStatus.done || 0) + 1, labelStatus.total)}/${labelStatus.total}`}
              </span>
            )}
          </div>

          {/* (영상 선택 버튼은 상단 액션줄로 이동) */}

          {/* 참조샷: 액션 버튼 바로 아래, 가로 칩 리스트.
              현재 영상 프레임 + 같은 부품의 다른 영상 요약(클릭하면 그 영상으로 이동)
              참조샷이 없어도 줄을 항상 띄운다. 첫 탭에서 줄이 새로 생기면 아래 이미지가
              그만큼 밀려 크기가 바뀌었다(들썩임). 자리를 미리 잡아두면 그 현상이 없다. */}
          {(
            <div className="al-shots" style={{ flexShrink: 0, marginTop: 6 }}>
              <span className="ref-shots-label">참조샷 {shotFrames.length + otherShots.reduce((a, o) => a + o.n, 0)}</span>
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
              {otherShots.map(o => (   // 다른 영상에 찍어둔 참조샷 — 라벨 생성 시 함께 처리된다
                <span key={o.name} className="al-shot other" title={`${o.name} 으로 이동`}
                      onClick={() => !running && setSelVideo(o.name)}>
                  <span className="al-shot-lbl">{o.name} {o.n}</span>
                </span>
              ))}
              {shotFrames.length === 0 && otherShots.length === 0 && (
                <span className="al-hint">프레임을 눌러 참조샷을 만드세요</span>
              )}
            </div>
          )}

          {/* 본문: 좌(이미지+범례+재생) / 우(리스트). 남는 공간 채움 */}
          {/* al-row / al-work / al-panes = 폰(<=560px)에서 세로로 쌓기 위한 선택자(app.css 반응형) */}
          <div className="al-row" style={{ display: 'flex', gap: 16, flex: 1, minHeight: 0, overflow: 'hidden', marginTop: 8 }}>
            <div className="al-work" style={{ flex: '1 1 0', minWidth: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
             {/* 이미지+범례+재생 = 남는 폭을 채우되 상한(사이드바 넓혀도 겹치지 않게 반응형) */}
             <div style={{ width: '100%', flex: '1 1 0', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              {/* 마스크 판정 배너 제거(2026-08-19): 판정 문구가 실제와 잘 안 맞았고,
                  배너가 세로 47px 을 차지해 프레임 이미지가 12% 작아졌다 커지는 들썩임을 만들었다.
                  마스크는 오른쪽 칸에서 눈으로 확인한다. */}
              {/* 듀얼 이미지: 좌우 동일 비율로 축소(스크롤 없음), 여백 흰색 */}
              <div className="al-panes" style={{ display: 'flex', gap: 12, flex: '1 1 0', minHeight: 0 }}>
                <div className="img-pane">
                  {preparing
                    ? <span className="al-hint">프레임 컷 중...</span>
                    : <div className="tap-box" onClick={(e) => addPoint(e, 1)} onContextMenu={(e) => addPoint(e, 0)}
                           style={natSize ? { aspectRatio: `${natSize.w} / ${natSize.h}` } : undefined}>
                        {src && <img src={`/api/autolabel/frame?src=${encodeURIComponent(srcKey)}&idx=${idx}&w=720`} alt={`frame ${idx}`} draggable={false}
                                     onLoad={(e) => { const im = e.currentTarget; setNatSize({ w: im.naturalWidth, h: im.naturalHeight }) }} />}
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
                        : <img src={activeMask.combo} alt="입력 마스크" style={{ maxHeight: '100%', maxWidth: '100%' }} />)
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
        <div className="modal-scrim" onClick={() => { setShowReview(false); setZoomFrame(null) }}>
          <div className="modal-card wide" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <b>라벨 검수 — {partName} {reviewFrames.length}장</b>
              <button className="icon-x" onClick={() => setShowReview(false)} aria-label="닫기"><IcX /></button>
            </div>
            {zoomFrame && (
              <div className="review-zoom" onClick={() => setZoomFrame(null)}>
                <img alt={zoomFrame.name}
                     src={`/api/sam2/train_frame?session=${encodeURIComponent(zoomFrame.session)}&name=${encodeURIComponent(zoomFrame.name)}&w=1280&part=${encodeURIComponent(zoomFrame.part || '')}`} />
                <span className="rz-name">{zoomFrame.name} · 클릭하면 닫힙니다</span>
              </div>
            )}
            <div className="review-grid">
              {reviewFrames.length === 0 && <p className="al-hint" style={{ padding: 16 }}>이 부품의 생성된 라벨이 없습니다.</p>}
              {reviewFrames.map(f => (
                <figure key={`${f.session}/${f.name}`} className="review-cell">
                  <img loading="lazy" alt={f.part} onClick={() => setZoomFrame(f)}
                       src={`/api/sam2/train_frame?session=${encodeURIComponent(f.session)}&name=${encodeURIComponent(f.name)}&w=360&part=${encodeURIComponent(f.part || '')}`} />
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
              {folderVideos.map(o => o.name).map(v => {
                const on = v === src
                return (
                  <div key={v} role="button" tabIndex={0} onClick={() => { setSelVideo(v); setShowVideoPick(false) }}
                       onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setSelVideo(v); setShowVideoPick(false) } }}
                       className={`vpick-card${on ? ' on' : ''}`}>
                    {on && <span className="vpick-badge">선택됨</span>}
                    {/* 영상 삭제(X): 원본은 보관함으로 이동, 프레임·라벨은 삭제 */}
                    <button className="review-del vpick-del" title="이 영상 삭제" disabled={running}
                            onClick={(e) => { e.stopPropagation(); if (!running) setConfirmDelVid(v) }}><IcX /></button>
                    <img src={`/api/autolabel/frame?src=${encodeURIComponent(keyOf(v))}&idx=0&w=360`} alt={v} loading="lazy" />
                    <span className="vpick-name">{isLabeled(v) ? '✓ ' : ''}{v}</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      <ConfirmModal open={!!confirmDelVid} title="영상 삭제"
                    message={confirmDelVid ? `${confirmDelVid} 영상을 삭제할까요? 이 영상으로 만든 프레임·라벨·참조샷도 함께 사라집니다.` : ''}
                    confirmLabel="삭제" danger
                    onCancel={() => setConfirmDelVid(null)}
                    onConfirm={() => { const v = confirmDelVid; setConfirmDelVid(null); deleteVideo(v) }} />
      <ConfirmModal open={!!alertMsg} title="알림" message={alertMsg || ''} confirmLabel="확인" alertOnly
                    onCancel={() => setAlertMsg(null)} onConfirm={() => setAlertMsg(null)} />
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
    const sx = x => pl + (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin)) * (W - pl - pr)   // 1점이면 가운데
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
          const last = sp[sp.length - 1]
          return (
            <g key={s.key}>
              {sp.length > 1 && <path d={d} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />}
              {/* 마지막 값에 점을 찍는다. 에포크가 1개면 선이 그려지지 않아 예전에는
                  빈 격자만 보였다(1에포크 학습에서 "점이 안 찍힌다"). */}
              {last && <circle cx={sx(last.epoch)} cy={sy(last[s.key])} r={sp.length > 1 ? 3 : 4}
                               fill={s.color} stroke="#fff" strokeWidth="1.5" />}
            </g>
          )
        })}
        {/* 에포크가 1개면 축 양끝에 같은 숫자가 찍혀 이상해 보였다 → 가운데 하나만 */}
        {xmin === xmax
          ? <text x={(pl + W - pr) / 2} y={H - 5} className="chart-tick" textAnchor="middle">{xmin}</text>
          : (<>
              <text x={pl} y={H - 5} className="chart-tick">{xmin}</text>
              <text x={W - pr} y={H - 5} className="chart-tick" textAnchor="end">{xmax}</text>
            </>)}
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
    <div className={`score-tile${bad ? ' bad' : ''}`}>   {/* GA 카드: 헤더(이름·종) · 현재값+등락 */}
      <div className="score-head">
        <span className="score-title">{label}</span>
        {n != null && <span className="score-n">{n}종</span>}     {/* 종 개수는 헤더 우측 */}
      </div>
      {after == null ? (                          /* 신규 모델 결과 자체가 없음 */
        <div className="score-na">비교 대상 없음 · 첫 배포</div>
      ) : (
        <div className="score-row">                {/* 현재값 옆에 등락(빨강/파랑) */}
          <span className="score-big">{pctv(after)}</span>
          {d != null && deltaEl(before, after)}
        </div>
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

// 프레임 사진 위에 검출 박스를 SVG로 겹쳐 그린다(사진에는 굽지 않는다 -> 박스를 끌 수 있다).
// SVG 의 viewBox + preserveAspectRatio 는 img 의 object-fit:contain 과 같은 방식으로 여백을 잡으므로
// 사진 비율이 세로든 가로든 박스가 사진에 정확히 겹친다.
function Shot({ f, before, after, show, alt }) {
  // 구 백엔드(박스를 사진에 구워 보내던 버전)의 응답도 그대로 보여준다 — 서버 재시작 전 공백 방지
  if (typeof after === 'string') return <div className="ba-shot"><img src={after} alt={alt} /></div>
  const B = show ? (before?.box || []) : []     // 기존 모델: 회색 점선
  const A = show ? (after?.box || []) : []      // 신규 모델: 초록(정답)·주황(오검)
  const fz = Math.max(f.iw || 640, f.ih || 480) / 40      // 라벨 글자 크기(사진 원본 픽셀 기준)
  return (
    <div className="ba-shot">
      <img src={f.img} alt={alt} />
      {(B.length + A.length) > 0 && (
        <svg className="ba-ov" viewBox={`0 0 ${f.iw} ${f.ih}`} aria-hidden="true">
          {B.map((b, i) => (
            <rect key={`b${i}`} className="prev" x={b.x} y={b.y} width={b.w} height={b.h}
                  vectorEffect="non-scaling-stroke" />
          ))}
          {A.map((b, i) => (
            <g key={`a${i}`} className={b.ok ? 'ok' : 'ng'}>
              <rect x={b.x} y={b.y} width={b.w} height={b.h} vectorEffect="non-scaling-stroke" />
              {i < 3 && (   // 라벨은 신뢰도 상위 3개만. 박스가 겹치면 글자도 겹치므로 한 줄씩 내려 쓴다
                <text x={b.x} y={Math.max(b.y - fz * 0.3, fz) + i * fz * 1.15} fontSize={fz}>
                  {b.cls} {b.conf.toFixed(2)}
                </text>
              )}
            </g>
          ))}
        </svg>
      )}
    </div>
  )
}

// 부품 하나의 여러 테스트 프레임을 좌우 화살표로 넘겨보는 Before/After 슬라이더
// frames = [{ img, iw, ih, before: {n,box}|null, after: {n,box} }, ...] (같은 part의 여러 프레임)
function BaGroup({ part, kind, frames }) {
  const [idx, setIdx] = useState(0)          // 현재 보고 있는 프레임 인덱스
  const [showBox, setShowBox] = useState(true)   // 박스 표시(끄면 원본 사진만 본다)
  const n = frames.length
  const i = Math.min(idx, n - 1)             // 프레임 수가 줄어도 인덱스 안전
  const cur = frames[i] || {}
  const go = (d) => setIdx(p => ((Math.min(p, n - 1) + d) % n + n) % n)   // 순환 이동(양끝 래핑)
  return (
    <div className="ba-group">
      <div className="ba-stage">
        {n > 1 && <button className="ba-nav prev" onClick={() => go(-1)} aria-label="이전 프레임"><IcChevronLeft /></button>}
        <div className="ba-one">
          {cur.after || cur.before
            ? <Shot f={cur} before={cur.before} after={cur.after} show={showBox}
                    alt={`${part} 기존·신규 모델 검출 비교`} />
            : <div className="ba-none">검출 없음</div>}
        </div>
        {n > 1 && <button className="ba-nav next" onClick={() => go(1)} aria-label="다음 프레임"><IcChevronRight /></button>}
      </div>
      <div className="ba-legend">
        <span className="lg-item"><i className="lg-sw gray" /> 기존 모델</span>
        <span className="lg-item"><i className="lg-sw green" /> 신규 · 정답 부품</span>
        <span className="lg-item"><i className="lg-sw orange" /> 신규 · 오검출</span>
        <button type="button" className="lg-toggle" aria-pressed={showBox}
                onClick={() => setShowBox(v => !v)}>
          {showBox ? '박스 숨기기' : '박스 보기'}
        </button>
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

// 인앱 확인 모달(네이티브 confirm·alert 대체). danger=파괴적 액션(빨강), alertOnly=알림만(취소 없음)
function ConfirmModal({ open, title, message, confirmLabel, danger, alertOnly, onConfirm, onCancel }) {
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
          {!alertOnly && <button className="act-btn ghost" onClick={onCancel}>취소</button>}
          <button className={`act-btn ${danger ? 'stop' : 'train'}`} onClick={onConfirm} autoFocus>{confirmLabel || '확인'}</button>
        </div>
      </div>
    </div>
  )
}


// 고른 3D 파일을 저장 전에 바로 보여준다. 파일당 objectURL 한 번만 만들고 바뀔 때 해제한다.
function LocalModel({ file }) {
  const [url, setUrl] = useState(null)
  useEffect(() => {
    const u = URL.createObjectURL(file)
    setUrl(u)
    return () => URL.revokeObjectURL(u)
  }, [file])
  if (!url) return null
  return <model-viewer class="reg-3d-view" camera-controls auto-rotate alt={file.name} src={url} />
}

// 업로드 전 로컬 영상 파일의 썸네일. 서버에 아직 없으니 브라우저에서 한 프레임을 그려 쓴다.
// preload='metadata' + loadeddata 조합은 브라우저에 따라 이벤트가 오지 않는다(썸네일이 안 뜨던 원인).
// 메타데이터가 오면 곧바로 탐색하고, seeked·canplay 어느 쪽이 먼저 오든 캡처한다.
function LocalThumb({ file }) {
  const [url, setUrl] = useState(null)
  useEffect(() => {
    let dead = false, done = false
    const src = URL.createObjectURL(file)
    const v = document.createElement('video')
    v.muted = true
    v.playsInline = true
    v.preload = 'auto'
    const grab = () => {
      if (done || dead || !v.videoWidth) return
      done = true
      try {
        const c = document.createElement('canvas')
        c.width = 200
        c.height = Math.max(1, Math.round(200 * v.videoHeight / v.videoWidth))
        c.getContext('2d').drawImage(v, 0, 0, c.width, c.height)
        setUrl(c.toDataURL('image/jpeg', 0.7))
      } catch { /* 브라우저가 못 여는 코덱이면 아이콘으로 남는다 */ }
      URL.revokeObjectURL(src)
    }
    v.addEventListener('loadedmetadata', () => {
      const t0 = Math.min(0.4, (v.duration || 1) / 4)
      try { v.currentTime = t0 } catch { grab() }
    })
    v.addEventListener('seeked', grab)
    v.addEventListener('canplay', grab)          // 탐색 이벤트가 안 오는 브라우저 대비
    v.addEventListener('error', () => URL.revokeObjectURL(src))
    v.src = src
    return () => { dead = true; v.removeAttribute('src'); if (!done) URL.revokeObjectURL(src) }
  }, [file])
  return url ? <img className="vm-thumb" src={url} alt="" />
             : <div className="vm-thumb none"><IcVideo /></div>
}

function ServerThumb({ src, alt }) {
  const [fail, setFail] = useState(0)
  useEffect(() => {
    if (fail !== 1) return
    const t = setTimeout(() => setFail(2), 4000)      // 추출 대기 후 재시도
    return () => clearTimeout(t)
  }, [fail])
  if (fail === 1) return <div className="vm-thumb none"><IcVideo /></div>
  return <img className="vm-thumb" loading="lazy" alt={alt} onError={() => setFail(1)}
              src={`/api/autolabel/frame?src=${encodeURIComponent(src)}&idx=0&w=200&r=${fail}`} />
}

function PartsApp({ onPrep, active }) {
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
  // 이 탭에서 처음 여는가(복구할 화면이 없나). 첫 렌더에서만 읽는다 —
  // 아래 저장 effect 가 곧 xr_page 를 덮어써서 effect 안에서는 구분할 수 없다.
  const freshTab = useRef(!sessionStorage.getItem('xr_page'))
  const [cmpJob, setCmpJob] = useState(() => sessionStorage.getItem('xr_cmpJob') || null)
  const [cmp, setCmp] = useState(null)               // 신규↔기존 모델 비교(평가) 상태
  const [applied, setApplied] = useState(false)      // 신규 모델 서비스 적용 완료
  const [svcMsg, setSvcMsg] = useState(null)         // 모델 변경 결과(적용·유지·롤백 공통 한 줄)
  const svcTimer = useRef(null)
  const flashSvc = (text) => {                       // 2초 뒤 자동으로 사라진다(닫기 버튼 없음)
    clearTimeout(svcTimer.current)
    setSvcMsg(text)
    svcTimer.current = setTimeout(() => setSvcMsg(null), 2000)
  }
  useEffect(() => () => clearTimeout(svcTimer.current), [])
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
    return { video, part: f ? partOf(f.folder) : video,
             labels: labeledMap[video].labels, trained: trainedVideos.includes(video) }
  }).filter(it => it.labels > 0).sort((a, b) => a.part.localeCompare(b.part))
  const selected = items.filter(it => picked.includes(it.part))
  const allOn = items.length > 0 && selected.length === items.length
  const toggle = (p) => setPicked(c => c.includes(p) ? c.filter(x => x !== p) : [...c, p])
  const toggleAll = () => setPicked(allOn ? [] : items.map(it => it.part))
  // 이미 학습된(현재 서비스 모델 보유) 부품 = 학습됨. 배지 판정과 동일 기준. 망각 방지 위해 기본 선택 대상.
  const trainedPartList = () => items.filter(it => served ? (served.classes || []).includes(it.part) : it.trained).map(it => it.part)

  const openTrain = () => {   // 학습 페이지로 이동(부품목록/라벨/모델평가 어디서 오든)
    // 화면 이동만으로는 학습·평가 결과를 비우지 않는다(초기화는 '새 학습'에서만).
    // 단, 죽은 잡(백엔드 재시작으로 JOBS 가 비면 status 는 {"error":"unknown job"})은 여기서 정리한다.
    // 탭 keep-alive 라 마운트 복구 effect 가 다시 돌지 않아, 이걸 안 하면 '학습 중 0% · 로그 대기 중'
    // 유령 패널이 계속 남는다.
    loadTrain()
    setPage('training')
    if (job && job !== 'err' && !status?.running) {
      fetch(`/api/sam2/status?job=${job}`).then(r => r.json())
        .then(d => {
          if (d && d.error) { setJob(null); setStatus(null) }   // 백엔드가 '없는 잡'이라고 확인해준 경우만 정리
          else if (d) setStatus(d)
        })
        .catch(() => {})   // 일시적 통신 오류로 정상 결과를 지우지 않는다
    }
  }
  const backToLabel = () => setPage('label')               // 학습은 계속 진행(폴링 유지), 라벨 화면으로 복귀
  const newRun = () => { setJob(null); setStatus(null); setCmpJob(null); setCmp(null); setApplied(false); setSvcMsg(null); setPicked(trainedPartList()); setPage('training') }

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
    const r = await fetch('/api/sam2/multiclass', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session, epochs, classes, augment: true })   // 배경 합성 증강 항상 적용
    }).then(x => x.json())
    if (r.error) { setJob('err'); setStatus({ error: r.error, running: false, log: [] }); return }
    setCmpJob(null); setCmp(null); setApplied(false)   // 새로 학습하면 이전 평가 결과는 무효(재비교 대상)
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
  const notify = (message) =>   // 인앱 알림(네이티브 alert 대체)
    ask({ title: '알림', message, confirmLabel: '확인', alertOnly: true, onOk: () => {} })
  const askDeleteModel = (mid) =>   // 타임라인/히스토리에서 버전 삭제(현재 서비스 모델은 백엔드가 거부)
    ask({ title: '모델 삭제', message: `모델 #${mid} 을(를) 삭제할까요? 되돌릴 수 없습니다.`,
          confirmLabel: '삭제', danger: true, onOk: () => doDeleteModel(mid) })
  const doDeleteModel = async (mid) => {
    const r = await fetch('/api/sam2/delete_model', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: mid })
    }).then(x => x.json()).catch(() => ({ error: '삭제 실패' }))
    if (r.error) { notify(r.error); return }
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
  useEffect(() => {   // job이 비워지면(새 설정) 세션스토리지도 같이 비움 — 안 그러면 리마운트 때 죽은 job이 되살아남
    if (job && job !== 'err') sessionStorage.setItem('xr_job', job)
    else sessionStorage.removeItem('xr_job')
  }, [job])
  useEffect(() => {   // cmpJob도 비워지면 같이 지운다 — 메모리와 스토리지가 갈리면 F5 결과가 인앱 이동과 달라짐
    if (cmpJob && cmpJob !== 'err') sessionStorage.setItem('xr_cmpJob', cmpJob)
    else sessionStorage.removeItem('xr_cmpJob')
  }, [cmpJob])
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
      // 다른 곳(폰 등)에서 실행 중인 잡이 있으면 그 잡에 다시 붙는다.
      // 화면 이동은 '이 탭에서 처음 여는 경우'에만 한다. 예전에는 항상 옮겼는데,
      // 라벨 화면에서 새로고침하거나 폰에서 탭을 다시 열면 화면이 저절로 학습/평가로
      // 튀어 '오락가락'하는 것처럼 보였다(QA 2026-08-20). 진행 중 표시는 헤더 칩으로 알린다.
      fetch('/api/sam2/active').then(r => r.json()).then(a => {
        if (!a || !a.job || !a.running) return
        if (a.session) setSession(a.session)
        if (a.kind === 'multiclass') {
          setJob(a.job); setStatus(a); loadTrain()
          if (freshTab.current) setPage('training')
        } else if (a.kind === 'compare') {
          setCmpJob(a.job); setCmp(a)
          if (freshTab.current) setPage('evaluate')
        }
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

  // 모델 변경 3가지(신규 적용 · 기존 유지 · 과거 롤백)는 동작을 통일한다.
  //   화면 이동 없음 / 결과는 같은 자리 한 줄 / 버튼은 사라지지 않고 상태만 바뀐다.
  const refreshServed = async () => {
    const [s, m] = await Promise.all([
      fetch('/api/sam2/served').then(x => x.json()).catch(() => null),
      fetch('/api/sam2/models').then(x => x.json()).catch(() => null),
    ])
    if (s) setServed(s && !s.none ? s : null)
    if (m) setModels(m.models || [])
    return s && !s.none ? s : null
  }

  const doRollbackTo = async (mid) => {   // 과거 버전을 서비스 모델로
    if (!mid) return
    const r = await fetch('/api/sam2/rollback_to', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: mid })
    }).then(x => x.json()).catch(() => ({ error: '롤백 실패' }))
    if (r.error) { notify(r.error); return }
    setRolledTo(mid); setApplied(false)
    const s = await refreshServed()
    flashSvc(`과거 버전으로 롤백했습니다 · 현재 서비스 모델 ${s?.label || mid}`)
  }

  const doRollback = async () => {   // 신규 폐기(기존 모델 유지) — 화면은 그대로 둔다
    await fetch('/api/sam2/rollback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {})
    setApplied(false)
    const s = await refreshServed()
    flashSvc(`기존 모델을 유지했습니다 · 현재 서비스 모델 ${s?.label || '없음'}`)
  }

  const doApply = async () => {   // 신규 모델을 서비스에 적용(추론 서버 배포는 백엔드가 함께 처리)
    // 화면에서 평가한 그 run 을 명시해서 보낸다. 안 보내면 백엔드가 '가장 최신 run' 을
    // 적용해서, 평가 중에 다른 학습이 끝나면 엉뚱한 모델이 서비스로 올라간다.
    const target = status?.model_id || cmp?.session || status?.session || session
    const r = await fetch('/api/sam2/apply_model', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: target })
    }).then(x => x.json()).catch(() => ({ error: '적용 실패' }))
    if (r.error) { notify(r.error); return }
    setApplied(true)
    const s = await refreshServed()
    flashSvc(`신규 모델을 적용했습니다 · 현재 서비스 모델 ${s?.label || r.session || target}`)
  }
  const doCancel = async () => {   // 학습 중단
    // job id 를 잃은 상태(화면 재진입·탭 이동)에서도 보낸다. 서버가 진행 중인 잡을 찾아 멈춘다.
    const r = await fetch('/api/sam2/cancel', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job: job && job !== 'err' ? job : null }),
    }).then(x => x.json()).catch(() => ({ error: '중단 요청 실패' }))
    if (r.error) { notify(r.error); return }
    // 중단은 배치 콜백에서 잡히므로 1~2초 안에 stage 가 cancelled 로 바뀐다. 폴링이 받아 화면을 정리한다.
    fetch('/api/sam2/active').then(x => x.json()).then(a => { if (!a?.running) setJob(null) }).catch(() => {})
  }

  const ep = status?.epoch || 0
  const tot = status?.total_epochs || epochs
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
  // 학습이 끝나면 비교를 자동으로 시작한다(버튼을 누르지 않아도 되게).
  // cmpJob 이 없을 때만 1회 — 재실행은 평가 화면의 다시 비교 버튼이 맡는다.
  useEffect(() => {
    if (trainDone && !cmpJob) runCompare(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainDone, cmpJob])
  const cmpDone = cmp?.stage === 'done'
  const cmpPct = cmpDone ? 100 : Math.round((cmp?.compare_frac || 0) * 100)
  const cmpTitle = cmpDone ? '모델 평가 완료'
    : (cmp?.error ? '평가 오류' : `모델 평가 중... (${cmp?.compare_done || 0}/${cmp?.compare_total || 0})`)
  const pctv = (x) => x == null ? '—' : `${Math.round(x * 100)}%`
  const deltaEl = (before, after) => {
    if (before == null || after == null) return <span className="delta flat">—</span>
    const d = Math.round((after - before) * 100)
    if (d > 0) return <span className="delta up">+{d}%p</span>       // 상승 = 빨강
    if (d < 0) return <span className="delta down">{d}%p</span>      // 하락 = 파랑(d 에 - 포함)
    return <span className="delta flat">0%p</span>
  }
  // 평가 화면용 파생값
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
      g.frames.push({ img: s.img || null, iw: s.iw || 640, ih: s.ih || 480,
                      before: s.before || null, after: s.after || null })
    }
    return [...byPart.values()]
  })()

  return (
    // 3개 하위 화면(라벨·학습·평가)이 카드 높이를 채우도록 flex 체인의 시작점
    <div style={{ display: 'flex', flexDirection: 'column', flex: '1 1 0', minHeight: 0 }}>
      {page === 'label' ? (
        <>
          <PageHead title="부품 학습 데이터 생성" flat
                    right={<>
                      {/* 다른 곳에서 학습/평가가 돌고 있으면 알려주고, 누르면 그 화면으로 간다
                          (화면을 자동으로 바꾸지 않는다 — 사용자가 있던 자리를 지킨다) */}
                      {status?.running && (
                        <button className="prep-chip" onClick={openTrain} type="button">
                          <span className="prep-spin" aria-hidden="true" />학습 진행 중
                        </button>)}
                      {!status?.running && cmp?.running && (
                        <button className="prep-chip" onClick={() => setPage('evaluate')} type="button">
                          <span className="prep-spin" aria-hidden="true" />모델 평가 중
                        </button>)}
                      <button className="ph-next" onClick={openTrain} title="학습 설정으로 이동" aria-label="학습 설정">
                        부품 학습<IcChevronRight />
                      </button>
                    </>} />
          <AutoLabelView onPrep={onPrep} active={active} />
        </>
      ) : page === 'training' ? (
        // ===== 2단계: 학습 (설정 → 진행 → 결과 요약) =====
        <div className="train-page">
          <PageHead
            title="부품 학습"
            back={onBackFromTrain}
            right={<>
              {!job && <button className="icon-back next" onClick={goManage} title="모델 관리" aria-label="모델 관리"><IcChevronRight /></button>}
              {trainDone && <button className="icon-back next" onClick={goEvaluate} title="모델 평가 · 적용" aria-label="모델 평가·적용"><IcChevronRight /></button>}
            </>}
          />

          {/* running 클래스: 폰에서 학습 중일 때 진행률·로그를 목록 위로 올리기 위한 표식(app.css) */}
          <div className={`train-split${running ? ' running' : ''}`}>
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
              {job && status && (
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
              {(!job || !status) && (
                // 학습 시작 전(또는 잡 상태를 확인 못한 경우): 에포크 입력창 + 시작 버튼
                <div className="train-setup-bar">
                  <span className="al-hint">Epoch</span>
                  <input className="ep-in" type="number" min={1} value={epochs}
                         onChange={(e) => setEpochs(Math.max(1, Math.floor(+e.target.value) || 1))} />
                  <button className="act-btn train" onClick={runTrain} disabled={selected.length === 0}>학습 시작</button>
                </div>
              )}
              {/* 로그창 크기는 고정한다. 예전에는 학습 중 compact(120px) -> 완료 200px 로 바뀌어
                  화면이 커졌다 작아졌다 했다. */}
              {job && status && <LogConsole log={status?.log} />}
              {status?.error && <div className="reco-banner rollback"><IcWarn /><span>학습 오류: {status.error}</span></div>}
              {status?.stage === 'cancelled' && <div className="reco-banner review"><IcWarn /><span>학습이 중단되었습니다.</span></div>}
              {/* 곡선을 결과 요약보다 먼저 보여준다(학습 경과 -> 최종 요약 순서) */}
              {job && status?.curve?.length > 0 && (
                <div className="ev2-card">
                  <h4 className="ev2-h">학습 곡선</h4>
                  <div className="charts">
                    <MiniLineChart title="Loss" data={status.curve}
                      series={[{ key: 'box', name: 'box_loss', color: '#ef4444' }, { key: 'cls', name: 'cls_loss', color: '#f59e0b' }, { key: 'dfl', name: 'dfl_loss', color: '#8b5cf6' }]} />
                    {/* mAP 는 이 학습에서 train=val 이라 0.99 로 포화돼 판단에 못 쓴다.
                        대신 '인지했다/못했다'를 그대로 보여주는 두 지표를 그린다. */}
                    <MiniLineChart title="Recall · Precision · F1" data={status.curve}
                      series={[{ key: 'r', name: 'Recall', color: '#10b981' },
                               { key: 'p', name: 'Precision', color: '#06b6d4' },
                               { key: 'f1', name: 'F1', color: '#8b5cf6' }]} />
                  </div>
                </div>
              )}
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
          {svcMsg && (   // 세 액션의 결과가 항상 같은 자리·같은 모양으로 나오고 2초 뒤 사라진다
            <div className="svc-msg" role="status"><IcCheck /><span>{svcMsg}</span></div>
          )}
          {/* 제목이 평가/평가 중/모델관리 세 가지로 바뀌어 같은 화면인지 알기 어려웠다.
              비교를 하러 온 경우는 계속 '모델 평가', 학습 없이 관리만 하러 온 경우는 '모델관리' 로 고정한다. */}
          <PageHead title={cmp ? '모델 평가' : '모델관리'} back={openTrain}
                    right={(!cmp?.running && !cmp?.error) ? (
                      <div className="ev2-head-actions">
                        <RollbackMenu models={models} servedId={served?.model_id}
                                      onRollbackTo={askRollbackTo} onDeleteModel={askDeleteModel}
                                      onKeep={null} onApply={null} applied={applied} />
                        {cmpDone && cmp.recommend && (   // 과거 조회 | 기존 유지 | 신규 적용 나란히(적용 후에도 남는다)
                          <>
                            <button className={`act-btn ${cmp.recommend.level === 'rollback' ? 'danger' : 'ghost'}`}
                                    onClick={doRollback}>기존 모델 유지</button>
                            <button className={`act-btn ${cmp.recommend.level === 'rollback' ? 'ghost' : 'train'}`}
                                    onClick={doApply} disabled={applied}>
                              {applied ? '✓ 적용됨' : '신규 모델 적용'}
                            </button>
                          </>
                        )}
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
                    {/* 1) 판정 + 근거 인식률을 한 줄에 — 세로로 쌓으면 아래 비교 뷰어가 눌려 이미지가 안 보인다 */}
                    <div className="ev2-top">
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
                      <section className="scoreboard">
                        <ScoreTile label="기존 부품 인식" n={cmp.gen?.n ?? 0}
                                   before={cmp.gen?.before} after={cmp.gen?.after} pctv={pctv} deltaEl={deltaEl} warnDown />
                        <ScoreTile label="신규 부품 인식" n={cmp.newp?.n ?? 0}
                                   before={cmp.newp?.before} after={cmp.newp?.after} pctv={pctv} deltaEl={deltaEl} />
                      </section>
                    </div>

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

// ===== 탭1: 부품 등록 (틀만 — 내부 로직 미구현) =====
// 부품 카테고리 선택 — 드롭다운에서 항목 선택·추가·편집·삭제(세션 내, 로직 틀)
// 카테고리 선택 + 추가·편집·삭제. 서버(categories 테이블)에 즉시 반영된다.
// value = category_id (숫자) 또는 null
function CategorySelect({ value, onChange }) {
  const [cats, setCats] = useState([])                // [{id, name}]
  const [open, setOpen] = useState(false)
  const [editId, setEditId] = useState(null)          // 편집 중인 항목 id
  const [draft, setDraft] = useState('')              // 편집 입력값
  const [adding, setAdding] = useState('')            // 새 항목 입력값
  const [err, setErr] = useState('')
  const ref = useRef(null)

  const load = useCallback(() => {
    fetch('/api/categories').then(r => r.json())
      .then(d => setCats(d.categories || [])).catch(() => {})
  }, [])
  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (!open) return
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) { setOpen(false); setEditId(null); setErr('') } }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  const cur = cats.find(c => c.id === value)
  const call = async (url, opts) => {
    const r = await fetch(url, opts).then(x => x.json()).catch(() => ({ error: '요청 실패' }))
    if (r.error) { setErr(r.error); return null }
    setErr(''); load(); return r
  }
  const saveEdit = async (c) => {
    const v = draft.trim()
    if (v && v !== c.name) {
      await call(`/api/categories/${c.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' },
                                              body: JSON.stringify({ name: v }) })
    }
    setEditId(null)
  }
  const del = async (c) => {
    const r = await call(`/api/categories/${c.id}`, { method: 'DELETE' })
    if (r && value === c.id) onChange(null)     // 선택 중이던 분류가 사라지면 선택 해제
  }
  const add = async () => {
    const v = adding.trim(); if (!v) return
    const r = await call('/api/categories', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                              body: JSON.stringify({ name: v }) })
    if (r) setAdding('')
  }
  return (
    <div className="csel" ref={ref}>
      <button type="button" className="csel-btn" onClick={() => setOpen(o => !o)}>
        <span className={cur ? '' : 'csel-ph'}>{cur ? cur.name : '카테고리 선택'}</span>
        <IcChevronDown />
      </button>
      {open && (
        <div className="csel-pop">
          <div className="csel-list">
            {cats.length === 0 && <div className="csel-empty">카테고리가 없습니다</div>}
            {cats.map(c => editId === c.id ? (
              <div className="csel-edit" key={c.id}>
                <input autoFocus value={draft} onChange={e => setDraft(e.target.value)}
                       onKeyDown={e => { if (e.key === 'Enter') saveEdit(c); if (e.key === 'Escape') setEditId(null) }} />
                <button type="button" className="csel-ic ok" onClick={() => saveEdit(c)} title="저장"><IcCheck /></button>
              </div>
            ) : (
              <div className={`csel-item${value === c.id ? ' on' : ''}`} key={c.id}>
                <button type="button" className="csel-pick" onClick={() => { onChange(c.id); setOpen(false) }}>{c.name}</button>
                <button type="button" className="csel-ic" onClick={() => { setEditId(c.id); setDraft(c.name) }} title="편집"><IcPencil /></button>
                <button type="button" className="csel-ic del" onClick={() => del(c)} title="삭제"><IcTrash /></button>
              </div>
            ))}
          </div>
          {err && <div className="csel-err">{err}</div>}
          <div className="csel-add">
            <input placeholder="새 카테고리 추가" value={adding} onChange={e => setAdding(e.target.value)}
                   onKeyDown={e => { if (e.key === 'Enter') add() }} />
            <button type="button" className="csel-addbtn" onClick={add} title="추가"><IcPlus /></button>
          </div>
        </div>
      )}
    </div>
  )
}

// 영상 여러 개를 순차 업로드하고, 각 영상의 프레임 사전 추출이 끝날 때까지 기다린다.
// 순차인 이유: 프레임 추출이 CPU(OpenCV)를 쓰므로 동시에 돌리면 서로 느려지고 진행 표시도 뒤섞인다.
// 반환: [{name, ok, count, error}]
async function uploadVideos(pid, files, onProgress = () => {}, onFile = () => {}) {
  const out = []
  for (let i = 0; i < files.length; i++) {
    const f = files[i]
    const tag = files.length > 1 ? ` (${i + 1}/${files.length})` : ''
    onProgress(`영상 업로드 중${tag}: ${f.name}`)
    const fd = new FormData(); fd.append('file', f)
    const v = await fetch(`/api/parts/${pid}/video`, { method: 'POST', body: fd })
      .then(x => x.json()).catch(() => ({ error: '업로드 실패' }))
    if (v.error) { out.push({ name: f.name, ok: false, error: v.error }); continue }
    onProgress(`프레임 추출 중${tag}: ${f.name}`)
    onFile(f.name, { stage: 'extract', count: 0, total: 0 })
    let done = null
    for (let k = 0; k < 900; k++) {                       // 최대 15분(긴 영상 대비)
      const s = await fetch(`/api/parts/job?job=${v.job}`).then(x => x.json()).catch(() => null)
      if (!s || s.error) break
      onFile(f.name, { stage: s.stage, count: s.count || 0, total: s.total || 0 })
      if (!s.running) { done = s; break }
      await new Promise(res => setTimeout(res, 700))
    }
    onFile(f.name, { stage: 'done', count: done?.count || 0, total: done?.count || 0 })
    if (done && !done.error) out.push({ name: f.name, ok: true, count: done.count })
    else out.push({ name: f.name, ok: !done, count: 0, error: done?.error || '추출 진행 중(백그라운드 계속)' })
  }
  return out
}

// 등록 화면 = 신규 등록 + 기존 부품 수정 겸용.
// editPart 가 있으면 수정 모드: 정보 저장 + 등록된 영상 목록 확인·삭제·추가를 한 화면에서 한다.
// 업로드 전 로컬 파일 미리보기. src 에 URL.createObjectURL 을 직접 쓰면 렌더마다
// 새 blob URL 이 생겨 누수되므로, 파일당 한 번만 만들고 닫힐 때 해제한다.
function LocalVideoPreview({ file }) {
  const url = useMemo(() => URL.createObjectURL(file), [file])
  useEffect(() => () => URL.revokeObjectURL(url), [url])
  return <video src={url} controls autoPlay muted playsInline />
}

function RegisterPart({ editPart, onExitEdit, onSaved, onDeleted }) {
  const editing = !!editPart
  const [name, setName] = useState('')
  const [cat, setCat] = useState(null)                 // category_id
  const [desc, setDesc] = useState('')
  const [vidMenu, setVidMenu] = useState(false)        // 동영상 프레임 클릭 시 열리는 선택 팝오버
  const [vidFiles, setVidFiles] = useState([])         // (신규 등록 시) 함께 업로드할 영상
  const [m3dFile, setM3dFile] = useState(null)         // 3D 모델 파일
  const [vids, setVids] = useState([])                 // (수정 시) 이미 등록된 영상 목록
  const [busy, setBusy] = useState('')                 // 진행 문구(등록·업로드·프레임 추출)
  const [msg, setMsg] = useState(null)                 // {ok|err, text}
  const [confirmDel, setConfirmDel] = useState(null)   // 삭제할 영상
  const [openVid, setOpenVid] = useState(null)
  const [vidProg, setVidProg] = useState({})       // 파일별 프레임 추출 진행률
  const [m3dGone, setM3dGone] = useState(false)    // 등록된 3D 를 지웠는지(화면 즉시 반영)
  const [confirm3d, setConfirm3d] = useState(false)
  const has3d = !!editPart?.has_model3d && !m3dGone          // 등록된 3D 유무(지운 직후 즉시 반영)

  const del3d = async () => {
    setConfirm3d(false)
    const r = await fetch(`/api/parts/${editPart.id}/model3d`, { method: 'DELETE' })
      .then(x => x.json()).catch(() => ({ error: '요청 실패' }))
    if (r.error) { setMsg({ ok: false, text: r.error }); return }
    setM3dGone(true)
    setMsg({ ok: true, text: '3D 모델을 제거했습니다' })
  }         // 미리보기 펼친 영상 stem
  const [confirmDelPart, setConfirmDelPart] = useState(false)   // 이 부품 자체를 삭제
  const vidRef = useRef(null)
  const vidInput = useRef(null)
  const camInput = useRef(null)     // 직접 촬영용(capture 속성이 붙은 별도 input)
  const m3dInput = useRef(null)

  useEffect(() => {                                    // 바깥 클릭 시 팝오버 닫기
    if (!vidMenu) return
    const onDoc = (e) => { if (vidRef.current && !vidRef.current.contains(e.target)) setVidMenu(false) }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [vidMenu])

  const loadVids = useCallback((pid) => {
    fetch(`/api/parts/${pid}/videos`).then(r => r.json())
      .then(d => setVids(d.videos || [])).catch(() => setVids([]))
  }, [])

  // 수정 대상이 바뀌면 그 부품 값으로 폼을 채운다(탭이 언마운트되지 않으므로 명시적으로 동기화)
  useEffect(() => {
    setMsg(null); setBusy(''); setVidFiles([]); setM3dFile(null); setOpenVid(null)
    if (editPart) {
      setName(editPart.name); setCat(editPart.category_id); setDesc(editPart.description || '')
      setVids([]); loadVids(editPart.id)
    } else {
      setName(''); setCat(null); setDesc(''); setVids([])
    }
  }, [editPart, loadVids])

  const reset = () => { setName(''); setCat(null); setDesc(''); setVidFiles([]); setM3dFile(null) }

  // 신규 등록: 부품 행 생성 → (있으면) 3D·영상 업로드 → 영상마다 프레임 사전 추출 완료까지 대기
  const create = async () => {
    setMsg(null); setBusy('부품 등록 중...')
    try {
      const r = await fetch('/api/parts', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), category_id: cat, description: desc })
      }).then(x => x.json())
      if (r.error) { setMsg({ ok: false, text: r.error }); return }
      const pid = r.id
      if (m3dFile) {
        setBusy('3D 모델 업로드 중...')
        const fd = new FormData(); fd.append('file', m3dFile)
        const m = await fetch(`/api/parts/${pid}/model3d`, { method: 'POST', body: fd }).then(x => x.json())
        if (m.error) { setMsg({ ok: false, text: `부품은 등록됐지만 3D 모델 실패: ${m.error}` }); return }
      }
      if (vidFiles.length) {
        const res = await uploadVideos(pid, vidFiles, setBusy,
          (name, p) => setVidProg(m => ({ ...m, [name]: p })))
        const okN = res.filter(x => x.ok).length
        const frames = res.reduce((a, x) => a + (x.count || 0), 0)
        const fails = res.filter(x => !x.ok)
        if (fails.length) {   // 일부 영상이 실패하면 이유를 봐야 하니 이 화면에 남긴다
          setMsg({ ok: false, text: `${r.name} 등록됨 · 영상 ${okN}/${vidFiles.length} 성공(프레임 ${frames}장). 실패: ${fails.map(f => `${f.name}(${f.error})`).join(', ')}` })
          reset(); return
        }
        reset(); onSaved?.(r.name); return   // 성공 = 목록에서 방금 등록한 줄을 보여준다
      }
      reset(); onSaved?.(r.name)
    } catch (e) {
      setMsg({ ok: false, text: `요청 실패: ${e}` })
    } finally { setBusy('') }
  }

  // 수정: 정보 저장(이름·카테고리·설명) + 3D 교체 → 성공하면 목록 화면으로 복귀
  const save = async () => {
    setMsg(null); setBusy('저장 중...')
    try {
      const r = await fetch(`/api/parts/${editPart.id}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), category_id: cat, description: desc })
      }).then(x => x.json())
      if (r.error) { setMsg({ ok: false, text: r.error }); return }
      if (m3dFile) {
        setBusy('3D 모델 업로드 중...')
        const fd = new FormData(); fd.append('file', m3dFile)
        const m = await fetch(`/api/parts/${editPart.id}/model3d`, { method: 'POST', body: fd }).then(x => x.json())
        if (m.error) { setMsg({ ok: false, text: `정보는 저장됐지만 3D 모델 실패: ${m.error}` }); return }
        setM3dFile(null)
      }
      onSaved?.(name.trim())   // 저장 성공 = 목록으로 복귀(방금 저장한 줄을 강조해서 보여준다)
    } catch (e) {
      setMsg({ ok: false, text: `요청 실패: ${e}` })
    } finally { setBusy('') }
  }

  // 수정 모드에서 영상 추가(여러 개) — 업로드 + 프레임 사전 추출까지
  // 신규 등록: 고른 파일을 목록에 누적한다(덮어쓰면 두 번째 선택에서 앞 것이 사라진다).
  // 같은 파일이면 건너뛰고, input 값을 비워야 같은 파일을 다시 골라도 change 가 뜬다.
  const pickVideos = (e) => {
    const picked = Array.from(e.target.files || [])
    e.target.value = ''
    if (!picked.length) return
    const idOf = f => `${f.name}_${f.size}`
    setVidFiles(prev => {
      const has = new Set(prev.map(idOf))
      return [...prev, ...picked.filter(f => !has.has(idOf(f)))]
    })
  }

  const addVideos = async (e) => {
    const files = Array.from(e.target.files || [])
    e.target.value = ''
    if (!files.length) return
    setMsg(null)
    const res = await uploadVideos(editPart.id, files, setBusy,
      (name, p) => setVidProg(m => ({ ...m, [name]: p })))
    setBusy('')
    const okN = res.filter(x => x.ok).length
    const frames = res.reduce((a, x) => a + (x.count || 0), 0)
    const fails = res.filter(x => !x.ok)
    setMsg(fails.length
      ? { ok: false, text: `영상 ${okN}/${files.length} 추가(프레임 ${frames}장). 실패: ${fails.map(f => `${f.name}(${f.error})`).join(', ')}` }
      : { ok: true, text: `영상 ${okN}개 추가 · 프레임 ${frames}장 추출` })
    loadVids(editPart.id)
  }

  const delVideo = async (v) => {
    setConfirmDel(null); setBusy(`${v.stem} 삭제 중...`); setMsg(null)
    const r = await fetch(`/api/parts/${editPart.id}/video/${encodeURIComponent(v.stem)}`, { method: 'DELETE' })
      .then(x => x.json()).catch(() => ({ error: '요청 실패' }))
    setBusy('')
    setMsg(r.error ? { ok: false, text: r.error }
                   : { ok: true, text: `${v.stem} 삭제됨` })
    loadVids(editPart.id)
  }

  // 수정 화면에서 이 부품 자체를 삭제. 목록으로 돌아가지 않고 여기서 끝낼 수 있게 한다.
  const delPart = async () => {
    setConfirmDelPart(false)
    const r = await fetch(`/api/parts/${editPart.id}`, { method: 'DELETE' })
      .then(x => x.json()).catch(() => ({ error: '요청 실패' }))
    if (r.error) { setMsg({ ok: false, text: r.error }); return }
    onDeleted?.(editPart.id)          // 목록 갱신 + 편집 상태 해제(사라진 부품 화면이 남지 않게)
  }

  return (
    <div className="tab-body">
      <div className="tab-head">
        <div>
          <h3 className="tab-h">{editing ? '부품 수정' : '부품 등록'}</h3>
          <p className="tab-sub">{editing
            ? '부품 정보를 수정하고 등록된 영상을 확인·삭제·추가합니다.'
            : '부품 정보와 3D 모델·동영상을 등록합니다.'}</p>
        </div>
      </div>
      <div className="reg-form">
        {/* 2열: 왼쪽 = 식별 정보(이름·카테고리·설명), 오른쪽 = 첨부(3D·동영상) */}
        <div className="reg-scroll">
      <div className="reg-grid">
          {/* 왼쪽: 이름 / 카테고리 / 설명(남는 높이 전체) 세로 배열 */}
          <div className="reg-col">
            <div className="reg-field">
              <span className="reg-label">부품 이름</span>
              <input className="reg-input" value={name} onChange={e => setName(e.target.value)} placeholder="예: gearbox" />
            </div>
            <div className="reg-field">
              <span className="reg-label">부품 카테고리</span>
              <CategorySelect value={cat} onChange={setCat} />
            </div>
            <div className="reg-field reg-desc-field">
              <span className="reg-label">부품 설명</span>
              <textarea className="reg-textarea" value={desc} onChange={e => setDesc(e.target.value)}
                        placeholder="부품에 대한 설명 (선택)" />
            </div>
          </div>

          <div className="reg-col">
            <div className="reg-field">
              <span className="reg-label">3D 모델</span>
              <input ref={m3dInput} type="file" accept=".glb,.gltf,.obj,.stl,.ply,.fbx" style={{ display: 'none' }}
                     onChange={e => setM3dFile(e.target.files?.[0] || null)} />
              {/* 등록된 3D 는 칸 아래에 작게 띄운다. glb 는 이미지가 아니라 3D 라 뷰어가 필요하다 */}
              <div className="reg-3d-wrap">
                <button type="button" className={`reg-drop${m3dFile || (editing && has3d) ? ' has-del' : ''}`}
                        onClick={() => m3dInput.current?.click()}>
                  <IcCube />
                  <div className="reg-drop-txt">
                    <b>{m3dFile ? m3dFile.name : (editing && has3d ? '3D 모델 등록됨' : '3D 모델 불러오기')}</b>
                  </div>
                  <span className="vm-size">{m3dFile ? `${(m3dFile.size / 1048576).toFixed(1)} MB`
                                                     : '.glb · .obj · .stl'}</span>
                </button>
                {(m3dFile || (editing && has3d)) && (
                  <button type="button" className="csel-ic del reg-3d-del" title="3D 모델 제거"
                          onClick={e => { e.stopPropagation(); m3dFile ? setM3dFile(null) : setConfirm3d(true) }}>
                    <IcTrash />
                  </button>
                )}
              </div>

              {/* 등록된 3D 를 칸 아래에 그림만 띄운다. glb 는 이미지가 아니라 3D 라 뷰어가 필요하다 */}
              {m3dFile
                ? <LocalModel file={m3dFile} />                       /* 고른 파일을 저장 전에 미리 본다 */
                : editing && has3d && (
                    <model-viewer class="reg-3d-view" camera-controls auto-rotate
                                  alt={`${editPart.name} 3D 모델`}
                                  src={`/api/xr/parts/${encodeURIComponent(editPart.name)}/model3d`} />
                  )}
            </div>

            <div className="reg-field reg-vid-field">
              <span className="reg-label">부품 동영상{editing ? ` (${vids.length}개)` : ''}</span>
              {/* 수정 모드: 파일 선택 즉시 업로드 / 신규: 등록 시 함께 업로드 */}
              <input ref={vidInput} type="file" multiple accept="video/mp4,video/quicktime,.mp4,.mov,.avi,.mkv"
                     style={{ display: 'none' }}
                     onChange={editing ? addVideos : pickVideos} />
              {/* 직접 촬영: capture 속성이 폰 기본 카메라(동영상 모드)를 띄운다. 찍으면 파일로 돌아와
                  업로드 경로를 그대로 탄다(HTTPS 불필요 — getUserMedia 를 쓰지 않으므로). */}
              <input ref={camInput} type="file" accept="video/*" capture="environment"
                     style={{ display: 'none' }}
                     onChange={editing ? addVideos : pickVideos} />
              <div className="reg-vidwrap" ref={vidRef}>
                <button type="button" className="reg-drop"
                        onClick={() => (IS_TOUCH ? setVidMenu(v => !v) : vidInput.current?.click())}
                        aria-expanded={IS_TOUCH ? vidMenu : undefined}>
                  <IcVideo />
                  <div className="reg-drop-txt">
                    <b>{!editing && vidFiles.length ? `동영상 ${vidFiles.length}개 선택됨` : '부품 동영상 추가'}</b>
                  </div>
                  <span className="vm-size">{!editing && vidFiles.length
                    ? `${(vidFiles.reduce((a, f) => a + f.size, 0) / 1048576).toFixed(1)} MB`
                    : '.mp4 · .mov'}</span>
                  {IS_TOUCH && <IcChevronDown />}
                </button>
                {IS_TOUCH && vidMenu && (
                  <div className="reg-vidpop">
                    <button className="reg-vopt" type="button" onClick={() => { setVidMenu(false); vidInput.current?.click() }}>
                      <IcVideo /><div><b>동영상 업로드</b><span>.mp4 · .mov 파일 선택(여러 개)</span></div>
                    </button>
                    <button className="reg-vopt" type="button"
                            onClick={() => { setVidMenu(false); camInput.current?.click() }}>
                      <IcCamera /><div><b>직접 촬영</b><span>{IS_TOUCH
                        ? '폰 카메라로 바로 촬영'
                        : '폰에서 촬영 (PC 는 파일 선택창이 열립니다)'}</span></div>
                    </button>
                  </div>
                )}
              </div>

              {/* 신규 등록: 올릴 파일 목록(개별 제거). 비어 있으면 자리표시로 열 높이 유지 */}
              {!editing && vidFiles.length === 0 && (
                <div className="vm-empty grow">선택한 영상이 여기에 표시됩니다</div>
              )}
              {!editing && vidFiles.length > 0 && (
                <ul className="vm-list grow">
                  {vidFiles.map((f, i) => {
                    const key = `${f.name}_${f.size}`
                    const open = openVid === key
                    return (
                    <li key={key} className={open ? 'on' : ''}>
                      {/* 업로드 전이라 서버에 없다 -> 로컬 파일을 objectURL 로 바로 재생 */}
                      <div className="vm-row" role="button" tabIndex={0}
                           onClick={() => setOpenVid(open ? null : key)}
                           onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpenVid(open ? null : key) } }}>
                        <LocalThumb file={f} />
                        <div className="vm-meta one">
                          <b>{f.name}</b>
                        </div>
                        <span className="vm-size">{(f.size / 1048576).toFixed(1)} MB</span>
                        <span className={`vm-caret${open ? ' on' : ''}`}><IcChevronDown /></span>
                        <button type="button" className="csel-ic del" title="목록에서 제거"
                                onClick={e => { e.stopPropagation(); setOpenVid(o => o === key ? null : o); setVidFiles(vidFiles.filter((_, k) => k !== i)) }}><IcTrash /></button>
                      </div>
                      {open && (
                        <div className="vm-player">
                          <LocalVideoPreview file={f} />
                        </div>
                      )}
                    </li>
                  )})}
                </ul>
              )}

              {/* 수정: 등록된 영상 목록(썸네일·역할·프레임수·삭제) */}
              {editing && (
                vids.length === 0
                  ? <div className="vm-empty grow">등록된 영상이 없습니다. 위에서 추가하세요.</div>
                  : <ul className="vm-list grow">
                      {vids.map(v => {
                        const open = openVid === v.stem
                        return (
                        <li key={v.stem} className={open ? 'on' : ''}>
                          {/* 항목을 누르면 아래로 밀리면서 실제 영상 미리보기가 펼쳐진다 */}
                          <div className="vm-row" role="button" tabIndex={0}
                               onClick={() => setOpenVid(open ? null : v.stem)}
                               onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpenVid(open ? null : v.stem) } }}>
                            <ServerThumb src={v.src} alt={v.stem} />
                            <div className="vm-meta one">
                              <b>{v.stem}</b>
                            </div>
                            <span className="vm-size">{v.frames}장{v.size_mb != null ? ` · ${v.size_mb} MB` : ''}</span>
                            <span className={`vm-caret${open ? ' on' : ''}`}><IcChevronDown /></span>
                            <button className="csel-ic del" type="button" disabled={!!busy} title="이 영상 삭제"
                                    onClick={e => { e.stopPropagation(); setConfirmDel(v) }}><IcTrash /></button>
                          </div>
                          {open && (
                            <div className="vm-player">
                              {/* 펼칠 때만 로드해서 목록 열자마자 여러 영상을 내려받지 않게 한다 */}
                              <video src={`/api/parts/${editPart.id}/video/${encodeURIComponent(v.stem)}/file`}
                                     controls autoPlay muted playsInline preload="metadata" />
                            </div>
                          )}
                        </li>
                      )})}
                    </ul>
              )}
            </div>
          </div>
        </div>

        {msg && <div className={`reg-msg ${msg.ok ? 'ok' : 'err'}`}>{msg.text}</div>}
        </div>
      <div className="reg-actions">
          {(() => {   // 진행 표시는 등록 버튼 왼쪽 한 곳. 문구는 버튼과 같은 말로 통일하고 % 만 덧붙인다
            const act = Object.values(vidProg).filter(p => p && p.stage !== 'done' && p.total)
            if (!busy && !act.length) return null
            const pct = act.length
              ? Math.min(99, Math.round(act.reduce((a, p) => a + p.count / p.total, 0) / act.length * 100))
              : null
            return (
              <span className="reg-busy">
                <span className="spinner" />
                {editing ? '저장 중' : '부품 등록 중'}{pct == null ? '' : ` ${pct}%`}
              </span>
            )
          })()}
          {editing && (
            <button className="act-btn dline" type="button" disabled={!!busy}
                    onClick={() => setConfirmDelPart(true)}><IcTrash /> 부품 삭제</button>
          )}
          {editing && (
            <button className="act-btn ghost" type="button" disabled={!!busy} onClick={onExitEdit}>
              취소
            </button>
          )}
          <button className="act-btn train" type="button" disabled={!name.trim() || !!busy}
                  onClick={editing ? save : create}>
            {editing ? '저장' : '부품 등록'}
          </button>
        </div>
      </div>
      <ConfirmModal open={confirm3d} title="3D 모델 제거"
                    message="등록된 3D 모델을 제거할까요? XR 이 내려받는 파일도 함께 사라집니다."
                    confirmLabel="제거" danger
                    onConfirm={del3d} onCancel={() => setConfirm3d(false)} />
      <ConfirmModal open={!!confirmDel} title="영상 삭제"
                    message={confirmDel ? `${confirmDel.stem} 을(를) 삭제할까요? 이 영상으로 만든 프레임·라벨·참조샷도 함께 사라집니다.` : ''}
                    confirmLabel="삭제" danger
                    onCancel={() => setConfirmDel(null)} onConfirm={() => delVideo(confirmDel)} />
      <ConfirmModal open={confirmDelPart} title="부품 삭제"
                    message={editing ? `${editPart.name} 을(를) 삭제할까요? 등록한 영상과 생성된 라벨이 함께 사라집니다.` : ''}
                    confirmLabel="삭제" danger
                    onCancel={() => setConfirmDelPart(false)} onConfirm={delPart} />
    </div>
  )
}


// ===== 탭2: 부품 목록 (DB 실데이터. 수정은 등록 화면으로 이동, 삭제는 여기서) =====
function PartList({ onEdit, onDeleted, active, flash, onFlashDone }) {
  const [rows, setRows] = useState([])
  const [loaded, setLoaded] = useState(false)
  const [msg, setMsg] = useState(null)
  const [confirmDel, setConfirmDel] = useState(null)

  const load = useCallback(() => {
    fetch('/api/parts').then(r => r.json())
      .then(d => { setRows(d.parts || []); setLoaded(true) })
      .catch(() => setLoaded(true))
  }, [])
  useEffect(() => { load() }, [load])
  // 이 탭이 활성될 때마다 갱신. 탭은 언마운트되지 않으므로(keep-alive) 명시적으로 다시 읽어야
  // 수정 화면에서 저장하고 돌아온 결과가 반영된다.
  useEffect(() => { if (active) load() }, [active, load])
  // 방금 등록·저장한 부품 줄로 스크롤하고 2초간 강조한다(토스트 대신 결과를 그 자리에서 보여준다)
  useEffect(() => {
    if (!flash || !active || !rows.length) return
    const el = document.querySelector(`[data-part="${CSS.escape(flash)}"]`)
    el?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    const t = setTimeout(() => onFlashDone?.(), 2000)
    return () => clearTimeout(t)
  }, [flash, active, rows, onFlashDone])

  const doDelete = async (p) => {
    setConfirmDel(null)
    const r = await fetch(`/api/parts/${p.id}`, { method: 'DELETE' }).then(x => x.json()).catch(() => ({ error: '요청 실패' }))
    if (r.error) setMsg({ ok: false, text: r.error })
    else { setMsg({ ok: true, text: `${p.name} 삭제됨` }); onDeleted?.(p.id); load() }
  }

  return (
    <div className="tab-body">
      <h3 className="tab-h">부품 목록</h3>
      <p className="tab-sub">등록된 부품입니다. 수정을 누르면 등록 화면에서 정보·영상을 관리합니다. (총 {rows.length}개)</p>
      {msg && <div className={`reg-msg ${msg.ok ? 'ok' : 'err'}`}>{msg.text}</div>}
      {!loaded ? (
        <div className="list-empty">불러오는 중...</div>
      ) : rows.length === 0 ? (
        <div className="list-empty">등록된 부품이 없습니다. ‘부품 등록’ 탭에서 추가하세요.</div>
      ) : (
        /* 표는 내부에서만 스크롤 — 페이지 전체가 내려가면 상단 탭이 화면 밖으로 밀린다 */
        <div className="pt-scroll">
        <table className="part-table">
          <thead>
            <tr>
              <th className="pt-w-name">이름</th>
              <th className="pt-w-cat">카테고리</th>
              <th>설명</th>
              <th className="pt-w-stat pt-mid">3D 모델</th>
              <th className="pt-w-manage2">관리</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(p => (
              <tr key={p.id} data-part={p.name} className={flash === p.name ? 'pt-flash' : undefined}>
                <td className="pt-name">
                  <span className="pt-name-txt">{p.name}</span>
                  {/* 호버 미리보기: 미리 추출해둔 첫 프레임 */}
                  <div className="pt-preview">
                    {p.videos?.[0]?.frames ? (
                      <img className="pt-preview-img" loading="lazy" alt={p.name}
                           src={`/api/autolabel/frame?src=${encodeURIComponent(`bell412/${p.name}/videos/${p.videos[0].stem}`)}&idx=0&w=240`} />
                    ) : (
                      <div className="pt-preview-thumb">{p.has_model3d ? <IcCube /> : <IcX />}</div>
                    )}
                    <b>{p.name}</b>
                    <p>{p.description || '설명 없음'}</p>
                    <p className="pt-preview-meta">영상 {p.n_videos}개{p.frames ? ` · 프레임 ${p.frames}장` : ''}</p>
                  </div>
                </td>
                <td className="pt-cat">{p.category || <span className="pt-no">—</span>}</td>
                <td className="pt-desc" title={p.description || ''}>{p.description}</td>
                <td className="pt-mid">{p.has_model3d ? <span className="pt-ok"><IcCheck /></span> : <span className="pt-no">—</span>}</td>
                <td>
                  <div className="pt-actions">
                    {/* 글자는 span 으로 분리 — 폰(<=560px)에서는 폭이 모자라 아이콘만 남긴다(app.css .btn-txt) */}
                    <button className="act-btn ghost sm" type="button" aria-label={`${p.name} 수정`}
                            onClick={() => onEdit?.(p)}><IcPencil /> <span className="btn-txt">수정</span></button>
                    <button className="act-btn dline sm" type="button" aria-label={`${p.name} 삭제`}
                            onClick={() => setConfirmDel(p)}><IcTrash /> <span className="btn-txt">삭제</span></button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
      <ConfirmModal open={!!confirmDel} title="부품 삭제"
                    message={confirmDel ? `${confirmDel.name} 을(를) 삭제할까요? 등록한 영상과 생성된 라벨이 함께 사라집니다.` : ''}
                    confirmLabel="삭제" danger
                    onCancel={() => setConfirmDel(null)} onConfirm={() => doDelete(confirmDel)} />
    </div>
  )
}


// 손가락 입력 기기인지(폰·태블릿) — '직접 촬영' 설명 문구를 기기에 맞게 쓰기 위한 판별
const IS_TOUCH = typeof window !== 'undefined' && !!window.matchMedia?.('(pointer: coarse)').matches

const APP_TABS = [
  { id: 'register', label: '부품 등록', Icon: IcPlus },
  { id: 'list', label: '부품 목록', Icon: IcList },
  { id: 'train', label: '부품 학습', Icon: IcLayers },
]

export default function App() {
  const [tab, setTab] = useState(() => localStorage.getItem('xr_tab') || 'register')
  const [prep, setPrep] = useState(null)   // 프레임 미리 컷 진행 {done,total,name} — 헤더 오른쪽에 표시
  // 한 번 열어본 탭은 언마운트하지 않고 숨기기만 한다(display:none).
  // 언마운트하면 진행 중 학습·찍어둔 참조샷·비교 결과·입력값이 전부 날아가고 재진입 때 전부 재조회(=재로드)됨.
  const [seen, setSeen] = useState(() => ({ [tab]: true }))   // 안 가본 탭은 마운트 안 해 초기 로딩 비용 절약
  const [editPart, setEditPart] = useState(null)              // 부품 목록의 '수정' -> 등록 화면 편집 모드
  const go = (t, keepEdit = false) => {
    setTab(t)
    setSeen(s => s[t] ? s : { ...s, [t]: true })
    localStorage.setItem('xr_tab', t)
    // 등록 탭을 떠나면 수정 상태를 푼다. 탭은 unmount 되지 않아서, 그냥 두면 다음에 등록 탭에
    // 들어올 때 '부품 수정' 화면이 그대로 남는다(사용자 지적).
    if (t !== 'register' && !keepEdit) setEditPart(null)
  }
  const openEdit = (p) => { setEditPart(p); go('register', true) }  // 수정은 등록 화면에서(영상 목록까지 한곳에서)
  const [flash, setFlash] = useState(null)                    // 목록에서 잠깐 강조할 부품 이름
  const doneEdit = (name) => { setEditPart(null); setFlash(name || null); go('list') }  // 결과를 볼 수 있는 목록으로
  // 삭제한 부품을 수정 중이었으면 편집 상태를 비운다(탭 keep-alive 라 그냥 두면
  // 등록 탭에 사라진 부품의 수정 화면이 계속 남는다)
  const dropEdit = (id) => setEditPart(p => (p && p.id === id ? null : p))
  const view = (id) => id === 'register' ? <RegisterPart editPart={editPart} onExitEdit={() => setEditPart(null)} onSaved={doneEdit} onDeleted={doneEdit} />
                     : id === 'list' ? <PartList onEdit={openEdit} onDeleted={dropEdit} active={tab === 'list'} flash={flash} onFlashDone={() => setFlash(null)} />
                     : <PartsApp onPrep={setPrep} active={tab === 'train'} />
  return (
    <main className="solo">
      <div className="apptabs">
        <div className="seg">
          {APP_TABS.map(({ id, label, Icon }) => (
            <button key={id} className={`apptab${tab === id ? ' on' : ''}`} onClick={() => go(id)} type="button">
              <Icon /><span>{label}</span>
            </button>
          ))}
        </div>
        {/* 배경 작업(영상 -> 프레임 미리 컷) 진행. 앱 헤더에 두면 어느 탭에 있어도 계속 보인다 */}
        {prep && (
          <span className="prep-chip" title={`영상에서 프레임을 미리 잘라 두는 중: ${prep.name}`}>
            <span className="prep-spin" aria-hidden="true" />
            프레임 준비 {prep.done + 1}/{prep.total}
          </span>
        )}
      </div>
      <div className="card">
        {APP_TABS.map(({ id }) => seen[id] && (
          <div key={id} className="tab-pane" style={tab === id ? undefined : { display: 'none' }}>{view(id)}</div>
        ))}
      </div>
    </main>
  )
}
