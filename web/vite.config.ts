import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend is `python -m agent.server` on 49010; /api (including the SSE
// stream) is proxied to it, unbuffered.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:49010',
        changeOrigin: true,
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
})
