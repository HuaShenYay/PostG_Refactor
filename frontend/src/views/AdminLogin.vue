<template>
  <div class="admin-login-container">
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
          <span class="date-stamp">管理后台</span>
          <h1 class="title">诗云</h1>
          <div class="decorative-line"></div>
          <span class="tagline">诗云 · 内容管理平台</span>
        </div>

        <n-form :model="form" @submit.prevent="handleLogin" class="login-form">
          <n-form-item>
            <n-input
              v-model:value="form.username"
              placeholder="管理员账号"
              size="large"
              :prefix="() => h(Person, { size: 20 })"
              class="modern-input"
            />
          </n-form-item>
          <n-form-item>
            <n-input
              v-model:value="form.password"
              type="password"
              placeholder="管理员密码"
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
              {{ loading ? '验证中...' : '进入管理台' }}
            </n-button>
          </n-form-item>
        </n-form>

        <div class="actions">
          <router-link to="/login" class="register-link">返回用户登录</router-link>
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
    const { data } = await axios.post('/api/admin/login', {
      username: form.username,
      password: form.password,
    })

    localStorage.setItem('admin_token', data.token)
    localStorage.setItem('admin_name', data.admin.username)
    message.success('欢迎进入管理台')
    router.push('/admin')
  } catch (error) {
    message.error(error.response?.data?.message || '管理员登录失败')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.admin-login-container {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #f7f9fc 0%, #f3f6f9 50%, #e8ecf1 100%);
}

.background-animation {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
}

.floating-circle {
  position: absolute;
  width: var(--size);
  height: var(--size);
  left: var(--x);
  top: var(--y);
  background: radial-gradient(circle, rgba(207, 63, 53, 0.06) 0%, transparent 70%);
  border-radius: 50%;
  animation: float 20s ease-in-out infinite;
  animation-delay: var(--delay);
}

@keyframes float {
  0%, 100% {
    transform: translate(0, 0) scale(1);
    opacity: 0.6;
  }
  33% {
    transform: translate(30px, -30px) scale(1.05);
    opacity: 0.8;
  }
  66% {
    transform: translate(-20px, 20px) scale(0.95);
    opacity: 0.5;
  }
}

.login-content {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 480px;
  padding: 20px;
}

.login-card {
  padding: 48px 40px;
  border-radius: 28px !important;
}

.editorial-header {
  text-align: center;
  margin-bottom: 40px;
}

.date-stamp {
  font-size: 11px;
  letter-spacing: 0.3em;
  color: var(--cinnabar-red);
  text-transform: uppercase;
  font-weight: 500;
  display: block;
  margin-bottom: 16px;
}

.title {
  font-family: 'Playfair Display', 'Noto Serif SC', serif;
  font-size: 56px;
  font-weight: 700;
  color: var(--ink-black);
  margin: 0;
  line-height: 1.1;
  letter-spacing: 0.1em;
}

.decorative-line {
  width: 60px;
  height: 2px;
  background: linear-gradient(90deg, var(--cinnabar-red), var(--antique-gold));
  margin: 20px auto;
  border-radius: 2px;
}

.tagline {
  font-size: 13px;
  color: var(--text-secondary);
  letter-spacing: 0.15em;
}

.login-form {
  margin-top: 32px;
}

.modern-input {
  height: 52px;
  border-radius: 16px !important;
  background: rgba(0, 0, 0, 0.03) !important;
  border: 1px solid transparent !important;
  transition: all 0.2s ease;
}

.modern-input:hover {
  background: rgba(0, 0, 0, 0.05) !important;
  border: 1px solid rgba(0, 0, 0, 0.08) !important;
}

.modern-input:focus-within {
  background: rgba(255, 255, 255, 0.95) !important;
  border: 1px solid var(--cinnabar-red) !important;
  box-shadow: 0 0 0 3px rgba(207, 63, 53, 0.1) !important;
}

.login-btn {
  width: 100%;
  height: 52px;
  border-radius: 16px !important;
  font-size: 16px;
  font-weight: 500;
  letter-spacing: 0.1em;
  background: var(--cinnabar-red) !important;
  border: none !important;
  box-shadow: 0 4px 12px rgba(207, 63, 53, 0.25);
  transition: all 0.2s ease;
}

.login-btn:hover {
  background: var(--cinnabar-light) !important;
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(207, 63, 53, 0.35);
}

.login-btn:active {
  transform: translateY(0);
}

.actions {
  margin-top: 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.register-link {
  font-size: 13px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: color 0.2s ease;
  letter-spacing: 0.05em;
}

.register-link:hover {
  color: var(--cinnabar-red);
}

.admin-link {
  color: var(--cinnabar-red);
  opacity: 0.8;
}

.footer-note {
  margin-top: 40px;
  text-align: center;
}

.footer-note p {
  font-size: 12px;
  color: var(--text-tertiary);
  letter-spacing: 0.2em;
}

.fade-in {
  animation: fadeIn 0.5s ease forwards;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 640px) {
  .login-card {
    padding: 36px 28px;
  }

  .title {
    font-size: 44px;
  }
}
</style>
