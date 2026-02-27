<template>
  <div class="profile-container">
    <nav class="top-nav glass-card">
      <div class="nav-brand" @click="router.push('/')">
        <span class="logo-text">诗云</span>
        <span class="edition-badge">Zen Edition</span>
      </div>
      
      <div class="nav-actions">
        <div class="nav-btn-card" @click="router.push('/search')" title="Search">
          <n-icon><NSearch /></n-icon>
          <span>搜索</span>
        </div>
        
        <div class="nav-btn-card" @click="router.push('/personal-analysis')" title="Personal Analysis">
          <n-icon><NPersonOutline /></n-icon>
          <span>个人万象</span>
        </div>
        
        <div class="nav-btn-card" @click="router.push('/global-analysis')" title="Global Analysis">
          <n-icon><NGlobeOutline /></n-icon>
          <span>全站万象</span>
        </div>

        <div class="divider-vertical"></div>

        <div class="user-area">
          <div v-if="currentUser !== '访客'" class="user-greeting active" @click="router.push('/profile')" title="个人信息">
            <n-icon class="user-icon"><NPersonOutline /></n-icon>
            <span class="user-name">{{ currentUser }}</span>
          </div>
          <div v-else class="login-prompt" @click="router.push('/login')">
            Login
          </div>
        </div>
      </div>
    </nav>

    <main class="profile-main anim-enter">
      <div class="profile-layout">
        <!-- Sidebar: User Info & Quick Actions -->
        <aside class="profile-sidebar">
          <div class="user-hero-card glass-card">
            <div class="avatar-wrapper">
              <div class="avatar-large">{{ currentUser.charAt(0) }}</div>
              <div class="status-dot"></div>
            </div>
            <h1 class="user-name-display">{{ currentUser }}</h1>
            <p class="user-bio">此去经年，应是良辰好景虚设。</p>
            
            <div class="sidebar-actions">
              <n-button 
                block 
                secondary 
                strong 
                round 
                size="large" 
                type="error" 
                class="logout-btn"
                @click="handleLogout"
              >
                <template #icon><n-icon><NLogOut /></n-icon></template>
                安全登出
              </n-button>
            </div>
          </div>

          <!-- Quick Stats Dashboard -->
          <div class="mini-stats-grid">
            <div class="stat-card glass-card" v-for="(val, label) in { '阅览': userStats.totalReads, '雅评': userStats.reviewCount, '游历': userStats.activeDays }" :key="label">
              <span class="stat-value">{{ val }}</span>
              <span class="stat-label">{{ label }}</span>
            </div>
          </div>
        </aside>

        <!-- Main Content: Account Settings -->
        <section class="profile-content">
          <div class="settings-card glass-card">
            <div class="card-header-zen">
              <n-icon class="header-icon"><NSettings /></n-icon>
              <div class="header-texts">
                <h2 class="card-title-zen">账户修缮</h2>
                <p class="card-subtitle-zen">在此更新您的身份凭证与安全设置</p>
              </div>
            </div>

            <div class="settings-form">
              <div class="form-group">
                <label>雅称 (Username)</label>
                <n-input 
                  v-model:value="formData.username" 
                  placeholder="请输入您的新雅称" 
                  size="large"
                  round
                >
                  <template #prefix><n-icon><NPerson /></n-icon></template>
                </n-input>
              </div>

              <div class="form-group">
                <label>密令 (Password)</label>
                <n-input 
                  v-model:value="formData.password" 
                  type="password" 
                  show-password-on="click" 
                  placeholder="如需更改请输入新密令" 
                  size="large"
                  round
                >
                  <template #prefix><n-icon><NLock /></n-icon></template>
                </n-input>
              </div>

              <div class="form-footer">
                <n-button 
                  type="primary" 
                  round 
                  size="large" 
                  class="save-btn"
                  :loading="updating"
                  @click="handleUpdate"
                >
                  保存所有更改
                </n-button>
              </div>
            </div>
          </div>

          <!-- Account Security Tip -->
          <div class="security-tip glass-card">
            <n-icon class="tip-icon"><NShield /></n-icon>
            <div class="tip-content">
              <h4>安全小贴士</h4>
              <p>为了您的账户安全，建议定期更换密令。诗云不会以任何形式要求您提供个人隐私信息。</p>
            </div>
          </div>
          
        </section>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { 
  NInput, 
  NButton, 
  NIcon, 
  useMessage,
  NCard
} from 'naive-ui'
import { 
  PersonOutline as NPerson,
  SearchOutline as NSearch,
  GlobeOutline as NGlobeOutline,
  PersonCircleOutline as NPersonOutline,
  LogOutOutline as NLogOut,
  SettingsOutline as NSettings,
  LockClosedOutline as NLock,
  ShieldCheckmarkOutline as NShield
} from '@vicons/ionicons5'
import axios from 'axios'

const router = useRouter()
const message = useMessage()
const currentUser = ref(localStorage.getItem('user') || '访客')
const updating = ref(false)

const userStats = ref({
  totalReads: 0,
  reviewCount: 0,
  activeDays: 0
})

const formData = ref({
  username: currentUser.value,
  password: ''
})

const handleUpdate = async () => {
  if (!formData.value.username) {
    message.error('雅称不可为空')
    return
  }
  
  updating.value = true
  try {
    const res = await axios.post('/api/user/update', {
      old_username: currentUser.value,
      new_username: formData.value.username,
      new_password: formData.value.password || null
    })
    
    if (res.data.status === 'success') {
      message.success('修缮成功')
      localStorage.setItem('user', formData.value.username)
      currentUser.value = formData.value.username
      formData.value.password = ''
    } else {
      message.error(res.data.message)
    }
  } catch (e) {
    message.error(e.response?.data?.message || '修缮失败')
  } finally {
    updating.value = false
  }
}

const handleLogout = () => {
  localStorage.removeItem('user')
  message.info('已安全离开诗云')
  router.push('/login')
}

const fetchUserStats = async () => {
  if (currentUser.value === '访客') return
  try {
    const res = await axios.get(`/api/user/${currentUser.value}/stats`)
    userStats.value = {
      totalReads: res.data.totalReads || 0,
      reviewCount: res.data.reviewCount || 0,
      activeDays: res.data.activeDays || 1
    }
  } catch (e) {
    void e
  }
}

onMounted(() => {
  fetchUserStats()
})
</script>

<style scoped>
.profile-container {
  min-height: 100vh;
  background: var(--gradient-bg);
  display: flex;
  flex-direction: column;
}

.profile-main {
  flex: 1;
  max-width: 1200px;
  margin: 40px auto;
  padding: 0 40px;
  width: 100%;
}

.profile-layout {
  display: grid;
  grid-template-columns: 350px 1fr;
  gap: 40px;
  align-items: start;
}

/* Sidebar Styling */
.profile-sidebar {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.user-hero-card {
  padding: 40px 30px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.avatar-wrapper {
  position: relative;
  margin-bottom: 20px;
}

.avatar-large {
  width: 100px;
  height: 100px;
  background: linear-gradient(135deg, var(--cinnabar-red), #d44c4c);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 40px;
  color: white;
  font-family: "Noto Serif SC", serif;
  box-shadow: 0 10px 30px rgba(207, 63, 53, 0.3);
}

.status-dot {
  position: absolute;
  bottom: 5px;
  right: 5px;
  width: 18px;
  height: 18px;
  background: #10b981;
  border: 3px solid rgba(255, 255, 255, 0.8);
  border-radius: 50%;
}

.user-name-display {
  font-family: "Noto Serif SC", serif;
  font-size: 28px;
  color: var(--ink-black);
  margin: 0 0 8px;
}

.user-bio {
  font-size: 14px;
  color: var(--text-tertiary);
  font-style: italic;
  margin-bottom: 30px;
}

.sidebar-actions {
  width: 100%;
  margin-top: 10px;
}

.logout-btn {
  font-weight: 600;
  letter-spacing: 0.1em;
}

/* Mini Stats */
.mini-stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.stat-card {
  padding: 15px 10px;
  text-align: center;
  transition: transform 0.3s var(--ease-smooth);
}

.stat-card:hover { transform: translateY(-5px); }

.stat-value {
  display: block;
  font-size: 22px;
  font-weight: 700;
  color: var(--cinnabar-red);
  font-family: "Playfair Display", serif;
}

.stat-label {
  font-size: 12px;
  color: var(--text-tertiary);
  font-weight: 500;
}

/* Content Area Styling */
.profile-content {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.settings-card {
  padding: 40px;
}

.card-header-zen {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 40px;
  padding-bottom: 20px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.header-icon {
  font-size: 36px;
  color: var(--cinnabar-red);
  background: rgba(207, 63, 53, 0.1);
  padding: 15px;
  border-radius: 12px;
}

.card-title-zen {
  font-family: "Noto Serif SC", serif;
  font-size: 24px;
  margin: 0;
  color: var(--ink-black);
}

.card-subtitle-zen {
  font-size: 14px;
  color: var(--text-tertiary);
  margin: 4px 0 0;
}

.settings-form {
  display: flex;
  flex-direction: column;
  gap: 25px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.form-group label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.form-footer {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.save-btn {
  min-width: 200px;
  font-weight: 600;
  background: linear-gradient(135deg, var(--cinnabar-red), #d44c4c);
  border: none;
}

/* Security Tip */
.security-tip {
  padding: 25px;
  display: flex;
  gap: 20px;
  align-items: start;
  background: linear-gradient(135deg, rgba(207, 63, 53, 0.03), rgba(207, 63, 53, 0.01));
}

.tip-icon {
  font-size: 28px;
  color: #10b981;
}

.tip-content h4 {
  margin: 0 0 8px;
  font-size: 16px;
  color: var(--ink-black);
}

.tip-content p {
  margin: 0;
  font-size: 14px;
  color: var(--text-tertiary);
  line-height: 1.6;
}

/* Animations */
.anim-enter {
  animation: slideUp 0.8s var(--ease-smooth);
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
  .profile-layout {
    grid-template-columns: 1fr;
  }
  
  .profile-sidebar {
    max-width: 100%;
  }
}

/* Lab Styles */
.lab-container {
    display: flex;
    gap: 40px;
    margin-top: 20px;
}
.lab-config {
    flex: 0 0 300px;
    padding-right: 30px;
    border-right: 1px solid rgba(0,0,0,0.05);
}
.config-val {
    margin-left: 10px;
    width: 30px;
    font-weight: bold;
}
.lab-actions {
    margin-top: 30px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.lab-charts {
    flex: 1;
    display: grid;
    grid-template-columns: repeat(2, minmax(420px, 1fr));
    gap: 20px;
}
.lab-chart {
    height: 420px;
    background: rgba(0,0,0,0.02);
    border-radius: 8px;
    padding: 10px;
}
.lab-placeholder {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0,0,0,0.02);
    border-radius: 8px;
    min-height: 300px;
}

@media (max-width: 1200px) {
    .lab-container { flex-direction: column; }
    .lab-config { border-right: none; border-bottom: 1px solid rgba(0,0,0,0.05); padding-bottom: 20px; }
}
@media (max-width: 900px) {
    .lab-charts { grid-template-columns: 1fr; }
    .lab-chart { height: 420px; }
}
</style>
