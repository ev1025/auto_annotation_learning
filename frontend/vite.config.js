import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 화면마다 진입점을 둔다(다중 페이지). 대시보드 탭에 넣지 않고 주소로 바로 들어가는 화면들:
//   /recog.html      부품 인식(라이브)
//   /mAP_Test.html   공인시험 채점
//   /gt_label.html   GT 만들기
// 단독 html 이던 시절의 주소를 그대로 유지한다(문서·심사 자료에 이 주소가 적혀 있다).
// 옛 단독 파일은 public/legacy/ 에 동결 사본으로 남긴다.
import { resolve } from 'path'

export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        recog: resolve(__dirname, 'recog.html'),
      },
    },
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:7862',
      '/previews': 'http://127.0.0.1:7862',
    },
  },
})
