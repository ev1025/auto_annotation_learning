// /recog.html 진입점. 대시보드 탭이 아니라 주소로 바로 들어오는 화면이다.
import React from 'react'
import ReactDOM from 'react-dom/client'
import RecogPage from './RecogPage.jsx'
import './app.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* 단독 화면이므로 항상 활성(카메라 자동 시작) */}
    <main className="solo"><div className="card"><div className="tab-pane">
      <RecogPage active={true} />
    </div></div></main>
  </React.StrictMode>,
)
