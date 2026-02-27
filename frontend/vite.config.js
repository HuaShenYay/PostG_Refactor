import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { visualizer } from 'rollup-plugin-visualizer'
import compression from 'vite-plugin-compression'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    compression(),
    compression({ algorithm: 'brotliCompress', ext: '.br' }),
    visualizer({
      filename: 'dist/bundle-report.html',
      gzipSize: true,
      brotliSize: true,
      open: false
    })
  ],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true
      }
    }
  },
  build: {
    sourcemap: true,
    minify: 'esbuild',
    cssCodeSplit: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vue: ['vue', 'vue-router', 'naive-ui'],
          echarts: ['echarts', 'echarts-wordcloud']
        }
      }
    }
  },
  esbuild: {
    drop: ['console', 'debugger']
  }
})
