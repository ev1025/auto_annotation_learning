// 부품 인식(라이브) 화면. public/recog.html 을 React 로 편입한 것.
//
// 단독 html 을 유지했던 이유는 "빌드 없이 파일 하나로 뜬다"였고, 대신 스타일·상태 코드가
// 대시보드와 갈라졌다. 편입으로 그 중복을 없애고, 옛 파일은 동결 사본으로 남겨 둔다
// (심사 자리에서 번들이 깨져도 /recog.html 로 띄울 수 있게).
//
// 카메라(getUserMedia)는 보안 컨텍스트에서만 열린다. HTTP(:7862)로 들어오면 안내만 띄우고,
// HTTPS(:8443 프록시)로 들어오면 자동으로 시작한다.
// 추론은 대시보드의 /api/detect 프록시를 지난다. 브라우저가 :9412 를 직접 부르면 CORS 에 막힌다.
import { useCallback, useEffect, useRef, useState } from 'react'

const CAMERA_ID = 'PHONE_TEST'
const INTERVAL = 333          // 3fps (요청 규격)
const TARGET = 'xr'           // 추론 대상 서버(연동 고정)
const MAX_DRAW = 12           // 덜 학습된 모델은 한 장에 수백 개를 뱉는다. 전부 그리면 사진이 안 보인다

export default function RecogPage({ active }) {
  const stageRef = useRef(null)
  const vidRef = useRef(null)
  const ovRef = useRef(null)
  const workRef = useRef(null)          // 프레임을 JPEG 로 굽는 작업용 캔버스
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const busyRef = useRef(false)
  const sentRef = useRef(0)
  const t0Ref = useRef(0)

  const [model, setModel] = useState('모델 확인 중...')
  const [note, setNote] = useState('')
  const [live, setLive] = useState(false)
  const [fps, setFps] = useState('')
  const [msg, setMsg] = useState('결과 없음')

  // ---- 현재 서비스 모델 ----
  useEffect(() => {
    if (!active) return
    fetch('/api/detect/health?target=' + TARGET).then(r => r.json()).then(d => {
      if (d && d.ok) {
        const n = Object.values(d.classes || {}).length
        setModel((d.model || 'model.pt') + ' · ' + n + '종')
      } else {
        setModel('추론서버 응답 없음 ' + ((d && d.error) || ''))
      }
    }).catch(() => setModel('추론서버 확인 실패'))
  }, [active])

  // 그려진 사진 영역에 캔버스를 정확히 겹친다.
  // object-fit:contain 이라 요소 박스(=스테이지) 안에서 사진은 여백을 두고 그려진다.
  const fitOverlay = useCallback((el) => {
    const ov = ovRef.current, stage = stageRef.current
    if (!el || !ov || !stage) return
    const natW = el.naturalWidth || el.videoWidth
    const natH = el.naturalHeight || el.videoHeight
    if (!natW || !natH) return
    const r = el.getBoundingClientRect(), s = stage.getBoundingClientRect()
    const k = Math.min(r.width / natW, r.height / natH)
    const w = natW * k, h = natH * k
    ov.style.left = (r.left - s.left + (r.width - w) / 2) + 'px'
    ov.style.top = (r.top - s.top + (r.height - h) / 2) + 'px'
    ov.style.width = w + 'px'
    ov.style.height = h + 'px'
  }, [])

  const drawBoxes = useCallback((dets, w, h, el) => {
    const ov = ovRef.current
    if (!ov) return
    if (el) fitOverlay(el)
    ov.width = w; ov.height = h; ov.hidden = false
    const g = ov.getContext('2d')
    g.clearRect(0, 0, w, h)
    g.lineWidth = Math.max(2, Math.round(w / 240))
    g.font = '700 ' + Math.max(13, Math.round(w / 32)) + 'px sans-serif'
    g.textBaseline = 'top'
    dets.forEach(d => {
      const b = d.bbox
      g.strokeStyle = '#22c55e'
      g.strokeRect(b.x, b.y, b.w, b.h)
      const label = d.detection_class + ' ' + Math.round(d.confidence * 100) + '%'
      const tw = g.measureText(label).width + 10
      const th = Math.max(18, Math.round(w / 26))
      g.fillStyle = 'rgba(34,197,94,.92)'
      g.fillRect(b.x, Math.max(0, b.y - th), tw, th)
      g.fillStyle = '#04250f'
      g.fillText(label, b.x + 5, Math.max(0, b.y - th) + 2)
    })
  }, [fitOverlay])

  const send = useCallback(async (blob, w, h, el) => {
    const fd = new FormData()
    fd.append('image', blob, 'frame.jpg')
    fd.append('camera_id', CAMERA_ID)
    fd.append('target', TARGET)
    const t = performance.now()
    try {
      const d = await fetch('/api/detect', { method: 'POST', body: fd }).then(r => r.json())
      const ms = Math.round(performance.now() - t)
      if (d.error) { setMsg(d.detail || d.error); return }
      const all = d.detections || []
      const dets = all.slice().sort((a, b) => b.confidence - a.confidence).slice(0, MAX_DRAW)
      if (dets.length) {
        // 이름·신뢰도는 사진 위 박스에 그린다. 아래 줄은 같은 정보 반복이라 요약만 둔다
        setMsg('응답 ' + ms + 'ms · 검출 ' + all.length + '개'
               + (all.length > dets.length ? ' (상위 ' + dets.length + '개 표시)' : ''))
        drawBoxes(dets, w, h, el)
      } else {
        setMsg('검출 없음 · 응답 ' + ms + 'ms')
        if (ovRef.current) ovRef.current.hidden = true
      }
    } catch (e) {
      setMsg(String(e))
    }
  }, [drawBoxes])

  const stopLive = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    setLive(false); setFps('')
    if (ovRef.current) ovRef.current.hidden = true
  }, [])

  const startLive = useCallback(async () => {
    if (streamRef.current) return
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }, audio: false,
      })
    } catch (e) {
      setNote('카메라를 열지 못했습니다: ' + e.name)
      return
    }
    const vid = vidRef.current
    vid.srcObject = streamRef.current
    await vid.play()
    setLive(true)
    sentRef.current = 0; t0Ref.current = performance.now()
    timerRef.current = setInterval(async () => {
      if (busyRef.current || !streamRef.current) return
      const w = vid.videoWidth, h = vid.videoHeight
      if (!w || !h) return
      busyRef.current = true
      const work = workRef.current || (workRef.current = document.createElement('canvas'))
      work.width = w; work.height = h
      work.getContext('2d').drawImage(vid, 0, 0, w, h)
      const blob = await new Promise(res => work.toBlob(res, 'image/jpeg', 0.8))
      await send(blob, w, h, vid)
      sentRef.current += 1
      setFps(sentRef.current + '장 · '
             + (sentRef.current / Math.max(1, (performance.now() - t0Ref.current) / 1000)).toFixed(1) + ' fps')
      busyRef.current = false
    }, INTERVAL)
  }, [send])

  // 탭에 들어오면 시작, 나가면 멈춘다. 카메라를 켜 둔 채 다른 탭으로 가면 폰이 계속 뜨겁다.
  useEffect(() => {
    if (!active) { stopLive(); return }
    if (!window.isSecureContext || !navigator.mediaDevices) {
      setNote('HTTP 접속이라 카메라를 브라우저가 막습니다. HTTPS(8443 포트)로 열어 주세요.')
      return
    }
    startLive()
    return stopLive
  }, [active, startLive, stopLive])

  useEffect(() => {
    const onHide = () => stopLive()
    const onResize = () => { if (live && ovRef.current && !ovRef.current.hidden) fitOverlay(vidRef.current) }
    window.addEventListener('pagehide', onHide)
    window.addEventListener('resize', onResize)
    return () => { window.removeEventListener('pagehide', onHide); window.removeEventListener('resize', onResize) }
  }, [live, fitOverlay, stopLive])

  return (
    <section className="recog">
      <div className="recog-head">
        <h2 className="page-title">부품 인식</h2>
        <span className="recog-model">{model}</span>
      </div>

      <div className="recog-stage" ref={stageRef}>
        {!live && <div className="recog-ph">{note || '카메라 준비 중...'}</div>}
        <video ref={vidRef} playsInline muted hidden={!live} />
        <canvas ref={ovRef} hidden />
        {fps && <span className="recog-fps">{fps}</span>}
      </div>

      <div className="recog-bar">
        <span className="recog-msg">{msg}</span>
        <span className="recog-acts">
          {live
            ? <button type="button" className="act-btn ghost" onClick={stopLive}>정지</button>
            : <button type="button" className="act-btn train" onClick={startLive}>카메라 시작</button>}
        </span>
      </div>
      {note && live && <p className="recog-note">{note}</p>}
    </section>
  )
}
