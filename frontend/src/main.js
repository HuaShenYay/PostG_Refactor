import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
// Naive UI 配置
import naive from 'naive-ui'
import './utils/responsive.js'
import './style.css'

const app = createApp(App)

app.use(naive)
app.use(router)
app.mount('#app')
