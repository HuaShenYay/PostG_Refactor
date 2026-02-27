<template>
  <div class="login-container">
    <div class="background-animation">
      <div class="floating-circle" v-for="i in 6" :key="i" :style="{ 
        '--delay': i * 0.5 + 's',
        '--size': (Math.random() * 200 + 100) + 'px',
        '--x': Math.random() * 100 + '%',
        '--y': Math.random() * 100 + '%'
      }"></div>
    </div>
    
    <div class="login-content fade-in">
      <n-card class="login-card glass-card" bordered="false">
        <div class="editorial-header">
          <span class="date-stamp">公元二零二五</span>
          <h1 class="title">诗云</h1>
          <div class="decorative-line"></div>
          <span class="tagline">诗云 · 重温古典社会下的民族语言</span>
        </div>
        
        <n-form :model="form" @submit.prevent="handleLogin" class="login-form">
          <n-form-item>
            <n-input 
              v-model:value="form.username" 
              placeholder="称 谓" 
              size="large"
              :prefix="() => h(Person, { size: 20 })"
              class="modern-input"
            />
          </n-form-item>
          <n-form-item>
            <n-input 
              v-model:value="form.password" 
              type="password" 
              placeholder="口 令" 
              size="large"
              :prefix="() => h(LockClosed, { size: 20 })"
              show-password-on="mousedown"
              class="modern-input"
            />
          </n-form-item>
          
          <n-form-item>
            <n-button 
              type="primary" 
              size="large" 
              :loading="loading"
              @click="handleLogin"
              class="login-btn modern-btn"
            >
              {{ loading ? '登录中...' : '登 录' }}
            </n-button>
          </n-form-item>
        </n-form>

        <div class="actions">
          <router-link to="/register" class="register-link">尚无称谓？前往注册</router-link>
        </div>

        <div class="footer-note">
          <p>诗云 · 现代语境下的古典回归</p>
        </div>
      </n-card>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, h } from 'vue'
import { useRouter } from 'vue-router'
import { useMessage } from 'naive-ui'
import { Person, LockClosed } from '@vicons/ionicons5'
import axios from 'axios'

const router = useRouter()
const message = useMessage()
const loading = ref(false)
const form = reactive({
  username: '',
  password: ''
})

const handleLogin = async () => {
  if (!form.username || !form.password) {
    message.warning('请完整填写')
    return
  }
  
  loading.value = true
  try {
    const res = await axios.post('/api/login', {
      username: form.username,
      password: form.password
    })
    
    if (res.data.status === 'success') {
      localStorage.setItem('user', form.username)
      message.success('登录成功')
      router.push('/')
    } else {
      message.error(res.data.message)
    }
  } catch (e) {
    message.error(e.response?.data?.message || '连接失败')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  overflow: hidden;
}

.background-animation {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  overflow: hidden;
}

.floating-circle {
  position: absolute;
  width: var(--size);
  height: var(--size);
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, rgba(166, 27, 27, 0.1), transparent);
  filter: blur(40px);
  animation: float-circle 20s ease-in-out infinite;
  animation-delay: var(--delay);
  left: var(--x);
  top: var(--y);
  transform: translate(-50%, -50%);
}

@keyframes float-circle {
  0%, 100% {
    transform: translate(-50%, -50%) translate(0, 0) scale(1);
  }
  25% {
    transform: translate(-50%, -50%) translate(30px, -30px) scale(1.1);
  }
  50% {
    transform: translate(-50%, -50%) translate(-20px, 20px) scale(0.9);
  }
  75% {
    transform: translate(-50%, -50%) translate(-30px, -20px) scale(1.05);
  }
}

.login-content {
  position: relative;
  z-index: 1;
  width: 100%;
  padding: 40px;
  animation: fade-in-up 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes fade-in-up {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.login-card {
  width: 100%;
  max-width: 440px;
  margin: 0 auto;
  border: none !important;
  background: rgba(255, 255, 255, 0.8) !important;
  backdrop-filter: blur(30px);
  -webkit-backdrop-filter: blur(30px);
  border-radius: 24px !important;
  animation: card-appear 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.2s both;
}

@keyframes card-appear {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(20px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

.login-card :deep(.n-card-body) {
  padding: 60px 40px;
}

.editorial-header {
  text-align: center;
  margin-bottom: 60px;
  position: relative;
}

.date-stamp {
  font-size: 14PX;
  letter-spacing: 0.5em;
  color: var(--accent-red);
  display: block;
  margin-bottom: 20px;
  font-weight: 300;
  opacity: 0.8;
  animation: fade-in 1s ease-out 0.4s both;
}

@keyframes fade-in {
  from {
    opacity: 0;
  }
  to {
    opacity: 0.8;
  }
}

.title {
  font-family: "Noto Serif SC", serif;
  font-size: 64PX;
  font-weight: 300;
  margin: 0;
  letter-spacing: 0.15em;
  color: var(--modern-black);
  background: linear-gradient(135deg, var(--modern-black) 0%, var(--accent-red) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: title-appear 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.5s both;
}

@keyframes title-appear {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.decorative-line {
  width: 60px;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent-red), transparent);
  margin: 20px auto 0;
  border-radius: 2px;
  animation: line-expand 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.7s both;
}

@keyframes line-expand {
  from {
    width: 0;
  }
  to {
    width: 60px;
  }
}

.tagline {
  font-size: 12PX;
  letter-spacing: 0.2em;
  color: #999;
  margin-top: 15px;
  display: block;
  text-transform: uppercase;
  animation: fade-in 0.8s ease-out 0.9s both;
}

@media (max-width: 768px) {
  .title { font-size: 48PX; }
  .login-card { max-width: 100%; }
  .login-card :deep(.n-card-body) { padding: 40px 30px; }
  .login-content { padding: 20px; }
}

@media (max-width: 480px) {
  .title { font-size: 36PX; }
  .date-stamp { font-size: 12PX; }
  .login-card :deep(.n-card-body) { padding: 30px 20px; }
  .login-form :deep(.n-input-input) { font-size: 14PX; }
  .login-btn { height: 48px; font-size: 13PX; }
  .editorial-header { margin-bottom: 40px; }
}

.login-form {
  margin-top: 40px;
}

.login-form :deep(.n-input-wrapper) {
  background: rgba(255, 255, 255, 0.5) !important;
  border: none !important;
  padding: 14px 20px !important;
  border-radius: 16px !important;
  transition: var(--transition-smooth);
  outline: none !important;
  box-shadow: none !important;
}

.login-form :deep(.n-input-wrapper:hover) {
  background: rgba(255, 255, 255, 0.8) !important;
  border: none !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  outline: none !important;
}

.login-form :deep(.n-input-wrapper.n-input-wrapper--focus) {
  background: white !important;
  border: none !important;
  box-shadow: none !important;
  transform: translateY(-2px);
  outline: none !important;
}

.login-form :deep(.n-input-input) {
  text-align: center;
  font-size: 16PX;
  font-weight: 300;
  letter-spacing: 0.1em;
  color: var(--modern-black);
}

.login-form :deep(.n-input-input::placeholder) {
  color: rgba(0, 0, 0, 0.3);
}

.login-btn {
  width: 100%;
  margin-top: 30px;
  font-size: 14PX;
  letter-spacing: 0.4em;
  height: 52px;
  font-weight: 300;
  text-indent: 0.4em;
  border-radius: 16px !important;
  background: linear-gradient(135deg, var(--accent-red) 0%, var(--accent-red-dark) 100%) !important;
  border: none !important;
  box-shadow: 0 4px 16px rgba(166, 27, 27, 0.3) !important;
  transition: var(--transition-bounce);
}

.login-btn:hover {
  background: linear-gradient(135deg, var(--accent-red-light) 0%, var(--accent-red) 100%) !important;
  box-shadow: 0 8px 24px rgba(166, 27, 27, 0.4) !important;
  transform: translateY(-3px);
}

.login-btn:active {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(166, 27, 27, 0.3) !important;
}

.actions {
  text-align: center;
  margin-top: 25px;
  animation: fade-in 0.8s ease-out 1s both;
}

.register-link {
  font-size: 13PX;
  color: #888;
  text-decoration: none;
  letter-spacing: 0.1em;
  transition: all 0.3s;
  position: relative;
  padding: 5px 10px;
}

.register-link::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 50%;
  width: 0;
  height: 1px;
  background: var(--accent-red);
  transition: all 0.3s;
  transform: translateX(-50%);
}

.register-link:hover {
  color: var(--accent-red);
}

.register-link:hover::after {
  width: 100%;
}

.footer-note {
  text-align: center;
  margin-top: 50px;
  opacity: 0.3;
  animation: fade-in 0.8s ease-out 1.2s both;
}

.footer-note p {
  font-size: 11PX;
  letter-spacing: 0.4em;
  color: var(--modern-black);
  font-weight: 200;
}
</style>
