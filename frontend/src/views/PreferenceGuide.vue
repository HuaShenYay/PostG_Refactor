<template>
  <div class="guide-container">
    <div class="background-animation">
      <div class="floating-circle" v-for="i in 6" :key="i" :style="{ 
        '--delay': i * 0.5 + 's',
        '--size': (Math.random() * 200 + 100) + 'px',
        '--x': Math.random() * 100 + '%',
        '--y': Math.random() * 100 + '%'
      }"></div>
    </div>
    
    <n-card class="guide-card glass-card" :bordered="false">
      <div class="editorial-header">
        <span class="date-stamp">公元二零二五</span>
        <h1 class="title">诗心初探</h1>
        <div class="decorative-line"></div>
        <span class="tagline">请选择您偏好的诗歌主题 · 我们将为您量身推荐</span>
      </div>

      <n-steps :current="currentStep" :status="stepStatus" class="steps-container">
        <n-step title="主题选择" description="选择您感兴趣的诗歌主题" />
        <n-step title="完成设置" description="开始您的诗歌之旅" />
      </n-steps>

      <div class="steps-content">
        <n-transition-group name="fade-slide" tag="div" class="step-wrapper">
          <div v-if="currentStep === 0" key="step1" class="step-content">
            <div v-if="loading" class="loading-state">
              <n-spin size="large" />
              <p>正在加载主题...</p>
            </div>

            <div v-else>
              <div class="category-tabs">
                <n-tabs v-model:value="activeCategory" type="segment" animated>
                  <n-tab-pane name="all" tab="全部主题">
                    <div class="topics-grid">
                      <div 
                        v-for="(keywords, topicId) in topics" 
                        :key="topicId"
                        class="topic-card"
                        :class="{ 'selected': selectedTopics.includes(topicId) }"
                        @click="toggleTopic(topicId)"
                      >
                        <div class="topic-number">主题 {{ topicId + 1 }}</div>
                        <div class="topic-keywords">
                          <span 
                            v-for="(keyword, index) in keywords.slice(0, 5)" 
                            :key="index"
                            class="keyword-tag"
                          >
                            {{ keyword }}
                          </span>
                        </div>
                        <div class="select-indicator" v-if="selectedTopics.includes(topicId)">
                          <n-icon><CheckmarkCircle /></n-icon>
                        </div>
                      </div>
                    </div>
                  </n-tab-pane>
                  <n-tab-pane name="nature" tab="自然山水">
                    <div class="topics-grid">
                      <div 
                        v-for="(keywords, topicId) in filteredTopics('nature')" 
                        :key="topicId"
                        class="topic-card"
                        :class="{ 'selected': selectedTopics.includes(topicId) }"
                        @click="toggleTopic(topicId)"
                      >
                        <div class="topic-number">主题 {{ topicId + 1 }}</div>
                        <div class="topic-keywords">
                          <span 
                            v-for="(keyword, index) in keywords.slice(0, 5)" 
                            :key="index"
                            class="keyword-tag"
                          >
                            {{ keyword }}
                          </span>
                        </div>
                        <div class="select-indicator" v-if="selectedTopics.includes(topicId)">
                          <n-icon><CheckmarkCircle /></n-icon>
                        </div>
                      </div>
                    </div>
                  </n-tab-pane>
                  <n-tab-pane name="emotion" tab="情感抒发">
                    <div class="topics-grid">
                      <div 
                        v-for="(keywords, topicId) in filteredTopics('emotion')" 
                        :key="topicId"
                        class="topic-card"
                        :class="{ 'selected': selectedTopics.includes(topicId) }"
                        @click="toggleTopic(topicId)"
                      >
                        <div class="topic-number">主题 {{ topicId + 1 }}</div>
                        <div class="topic-keywords">
                          <span 
                            v-for="(keyword, index) in keywords.slice(0, 5)" 
                            :key="index"
                            class="keyword-tag"
                          >
                            {{ keyword }}
                          </span>
                        </div>
                        <div class="select-indicator" v-if="selectedTopics.includes(topicId)">
                          <n-icon><CheckmarkCircle /></n-icon>
                        </div>
                      </div>
                    </div>
                  </n-tab-pane>
                  <n-tab-pane name="life" tab="生活哲理">
                    <div class="topics-grid">
                      <div 
                        v-for="(keywords, topicId) in filteredTopics('life')" 
                        :key="topicId"
                        class="topic-card"
                        :class="{ 'selected': selectedTopics.includes(topicId) }"
                        @click="toggleTopic(topicId)"
                      >
                        <div class="topic-number">主题 {{ topicId + 1 }}</div>
                        <div class="topic-keywords">
                          <span 
                            v-for="(keyword, index) in keywords.slice(0, 5)" 
                            :key="index"
                            class="keyword-tag"
                          >
                            {{ keyword }}
                          </span>
                        </div>
                        <div class="select-indicator" v-if="selectedTopics.includes(topicId)">
                          <n-icon><CheckmarkCircle /></n-icon>
                        </div>
                      </div>
                    </div>
                  </n-tab-pane>
                </n-tabs>
              </div>

              <div class="selection-summary">
                <n-alert type="info" :bordered="false">
                  <template #header>
                    已选择 {{ selectedTopics.length }}/5 个主题
                  </template>
                  <div v-if="selectedTopics.length > 0" class="selected-tags">
                    <n-tag 
                      v-for="topicId in selectedTopics" 
                      :key="topicId"
                      closable
                      @close="toggleTopic(topicId)"
                      type="error"
                      round
                      class="selected-tag"
                    >
                      {{ topics[topicId]?.name || `主题 ${topicId + 1}` }}
                    </n-tag>
                  </div>
                </n-alert>
              </div>
            </div>
          </div>

          <div v-if="currentStep === 1" key="step3" class="step-content">
            <div class="completion-summary">
              <n-icon size="80" color="#18a058">
                <CheckmarkCircle />
              </n-icon>
              <h3>设置完成！</h3>
              <p>您的偏好已保存，即将为您开启诗歌之旅</p>
              
              <n-divider />

              <div class="summary-details">
                <n-descriptions :column="1" bordered>
                  <n-descriptions-item label="已选主题">
                    {{ selectedTopics.length }} 个
                  </n-descriptions-item>
                  <n-descriptions-item v-if="selectedTopics.length > 0" label="所选主题">
                    <n-space>
                      <n-tag 
                        v-for="topicId in selectedTopics" 
                        :key="topicId"
                        round
                        type="success"
                        size="small"
                      >
                        {{ topics[topicId]?.name || `主题 ${topicId + 1}` }}
                      </n-tag>
                    </n-space>
                  </n-descriptions-item>
                </n-descriptions>
              </div>
            </div>
          </div>
        </n-transition-group>
      </div>

      <div class="guide-actions">
        <n-space vertical size="large">
          <div v-if="currentStep === 0" class="action-buttons">
            <n-button 
              type="primary" 
              size="large" 
              :disabled="selectedTopics.length === 0"
              :loading="submitting"
              @click="handleSavePreferences"
              class="action-btn"
            >
              完成设置
            </n-button>
            <n-button 
              size="large" 
              @click="handleSkip"
              class="action-btn secondary"
            >
              跳过设置
            </n-button>
          </div>

          <div v-if="currentStep === 1" class="action-buttons">
            <n-button 
              type="primary" 
              size="large" 
              @click="goToHome"
              class="action-btn"
            >
              进入主页
            </n-button>
          </div>

          <p class="hint-text">
            <template v-if="currentStep === 0">
              至少选择一个主题，或点击跳过直接进入主页
            </template>
            <template v-if="currentStep === 1">
              调整您的推荐偏好，完成后点击完成设置
            </template>
            <template v-if="currentStep === 2">
              准备就绪，开始您的诗歌之旅
            </template>
          </p>
        </n-space>
      </div>

      <div class="footer-note">
        <p>诗云 · 现代语境下的古典回归</p>
      </div>
    </n-card>

    <n-modal v-model:show="showConfirmDialog" preset="dialog" title="确认跳过">
      <template #header>
        <div class="dialog-header">
          <n-icon size="24" color="#f0a020">
            <Warning />
          </n-icon>
          <span>确认跳过</span>
        </div>
      </template>
      <p>跳过偏好设置后，系统将为您推荐多样化的诗歌。您可以在后续的观象页面中调整偏好。</p>
      <p>确定要跳过吗？</p>
      <template #action>
        <n-space>
          <n-button @click="showConfirmDialog = false">取消</n-button>
          <n-button type="warning" @click="confirmSkip">确认跳过</n-button>
        </n-space>
      </template>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useMessage, useDialog } from 'naive-ui'
import { CheckmarkCircle, Warning } from '@vicons/ionicons5'
import axios from 'axios'

const router = useRouter()
const route = useRoute()
const message = useMessage()
const dialog = useDialog()

const loading = ref(true)
const submitting = ref(false)
const topics = ref({})
const selectedTopics = ref([])
const username = ref('')
const currentStep = ref(0)
const stepStatus = ref('process')
const activeCategory = ref('all')
const showConfirmDialog = ref(false)



const topicCategories = ref({
  nature: [0, 2, 4, 6, 8],
  emotion: [1, 3, 5, 7, 9],
  life: [10, 11, 12, 13, 14]
})

onMounted(async () => {
  username.value = route.query.username || localStorage.getItem('user')
  if (!username.value) {
    message.warning('请先登录')
    router.push('/login')
    return
  }
  
  await loadTopics()
})

const loadTopics = async () => {
  try {
    const res = await axios.get('/api/topics')
    topics.value = res.data
  } catch (e) {
    message.error('加载主题失败')
  } finally {
    loading.value = false
  }
}

const filteredTopics = (category) => {
  const categoryIds = topicCategories.value[category] || []
  const filtered = {}
  categoryIds.forEach(id => {
    if (topics.value[id]) {
      filtered[id] = topics.value[id]
    }
  })
  return filtered
}

const toggleTopic = (topicId) => {
  const index = selectedTopics.value.indexOf(topicId)
  if (index > -1) {
    selectedTopics.value.splice(index, 1)
    message.info('已取消选择主题 ' + (topicId + 1))
  } else {
    if (selectedTopics.value.length < 5) {
      selectedTopics.value.push(topicId)
      message.success('已选择主题 ' + (topicId + 1))
    } else {
      message.warning('最多选择5个主题')
    }
  }
}



const handleSavePreferences = async () => {
  submitting.value = true
  try {
    const res = await axios.post('/api/save_initial_preferences', {
      username: username.value,
      selected_topics: selectedTopics.value
    })
    
    if (res.data.status === 'success') {
      message.success('偏好设置成功')
      currentStep.value = 1
      stepStatus.value = 'finish'
    } else {
      message.error(res.data.message)
    }
  } catch (e) {
    message.error(e.response?.data?.message || '保存失败')
  } finally {
    submitting.value = false
  }
}

const handleSkip = () => {
  showConfirmDialog.value = true
}

const confirmSkip = async () => {
  showConfirmDialog.value = false
  message.info('已跳过偏好设置，系统将为您推荐多样化的诗歌')
  setTimeout(() => {
    router.push('/')
  }, 1000)
}

const goToHome = () => {
  router.push('/')
}
</script>

<style scoped>
.guide-container {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  overflow: hidden;
  background: var(--gradient-warm);
  background-attachment: fixed;
  padding: 40px 20px;
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
  background: radial-gradient(circle at 30% 30%, rgba(166, 27, 27, 0.08), transparent);
  filter: blur(40px);
  animation: float-circle 25s ease-in-out infinite;
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
    transform: translate(-50%, -50%) translate(20px, -20px) scale(1.1);
  }
  50% {
    transform: translate(-50%, -50%) translate(-15px, 15px) scale(0.9);
  }
  75% {
    transform: translate(-50%, -50%) translate(-20px, -15px) scale(1.05);
  }
}

.guide-card {
  width: 100%;
  max-width: 1000px;
  border: none !important;
  border-radius: 24px !important;
  background: rgba(255, 255, 255, 0.85) !important;
  backdrop-filter: blur(30px);
  -webkit-backdrop-filter: blur(30px);
  position: relative;
  z-index: 1;
  animation: card-appear 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.2s both;
}

@keyframes card-appear {
  from {
    opacity: 0;
    transform: translateY(30px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.editorial-header {
  text-align: center;
  margin-bottom: 50px;
  position: relative;
  animation: header-appear 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.4s both;
}

@keyframes header-appear {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.date-stamp {
  font-size: 14PX;
  letter-spacing: 0.5em;
  color: var(--accent-red);
  display: block;
  margin-bottom: 20px;
  font-weight: 300;
  opacity: 0.8;
}

.title {
  font-family: "Noto Serif SC", serif;
  font-size: 56PX;
  font-weight: 300;
  margin: 0;
  letter-spacing: 0.15em;
  color: var(--modern-black);
  background: linear-gradient(135deg, var(--modern-black) 0%, var(--accent-red) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.decorative-line {
  width: 60px;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent-red), transparent);
  margin: 20px auto 0;
  border-radius: 2px;
}

@media (max-width: 768px) {
  .title { font-size: 40PX; }
}

.tagline {
  font-size: 12PX;
  letter-spacing: 0.2em;
  color: #999;
  margin-top: 15px;
  display: block;
  text-transform: uppercase;
}

.steps-container {
  margin-bottom: 40px;
  animation: steps-appear 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.5s both;
}

@keyframes steps-appear {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.steps-content {
  min-height: 400px;
  margin-bottom: 30px;
}

.step-wrapper {
  position: relative;
}

.step-content {
  animation: fade-in-up 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes fade-in-up {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.loading-state {
  text-align: center;
  padding: 80px 0;
  color: #999;
}

.loading-state p {
  margin-top: 20px;
  font-size: 14PX;
}

.category-tabs {
  margin-bottom: 30px;
}

.topics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

@media (max-width: 600px) {
  .topics-grid {
    grid-template-columns: 1fr;
  }
}

.topic-card {
  border: 2px solid rgba(0, 0, 0, 0.08);
  border-radius: 24px;
  padding: 24px;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  animation: card-entry 0.4s cubic-bezier(0.4, 0, 0.2, 1) both;
}

@keyframes card-entry {
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.topic-card:hover {
  border-color: var(--accent-red);
  transform: translateY(-4px) scale(1.02);
  box-shadow: 0 8px 24px rgba(166, 27, 27, 0.15);
  background: rgba(255, 255, 255, 0.95);
}

.topic-card.selected {
  border-color: var(--accent-red);
  background: linear-gradient(135deg, rgba(166, 27, 27, 0.05) 0%, rgba(255, 255, 255, 0.95) 100%);
  box-shadow: 0 4px 16px rgba(166, 27, 27, 0.2);
  animation: card-selected 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes card-selected {
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
  100% {
    transform: scale(1);
  }
}

.topic-number {
  font-size: 14PX;
  color: #999;
  margin-bottom: 16px;
  font-weight: 300;
}

.topic-keywords {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.keyword-tag {
  font-size: 13PX;
  color: var(--modern-black);
  background: rgba(0, 0, 0, 0.04);
  padding: 6px 16px;
  border-radius: 24px;
  transition: all 0.3s ease;
}

.topic-card:hover .keyword-tag,
.topic-card.selected .keyword-tag {
  background: rgba(166, 27, 27, 0.1);
  color: var(--accent-red);
}

.select-indicator {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 28px;
  height: 28px;
  background: var(--accent-red);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  box-shadow: 0 4px 12px rgba(166, 27, 27, 0.3);
  animation: indicator-appear 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes indicator-appear {
  from {
    opacity: 0;
    transform: scale(0) rotate(-180deg);
  }
  to {
    opacity: 1;
    transform: scale(1) rotate(0deg);
  }
}

.selection-summary {
  margin-top: 30px;
}

.selected-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.selected-tag {
  border-radius: 24px !important;
}

.preferences-form {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  animation: form-appear 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes form-appear {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.completion-summary {
  text-align: center;
  padding: 40px 20px;
  animation: completion-appear 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes completion-appear {
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.completion-summary h3 {
  font-size: 24PX;
  margin: 20px 0 10px;
  color: var(--modern-black);
}

.completion-summary p {
  color: #666;
  font-size: 14PX;
}

.summary-details {
  max-width: 500px;
  margin: 0 auto;
  text-align: left;
}

.guide-actions {
  text-align: center;
  margin-top: 40px;
  animation: actions-appear 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.3s both;
}

@keyframes actions-appear {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
}

.action-btn {
  min-width: 160px;
  height: 52px;
  font-size: 14PX;
  letter-spacing: 0.2em;
  font-weight: 300;
  border-radius: 24px !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.action-btn:not(.secondary) {
  background: linear-gradient(135deg, var(--accent-red) 0%, var(--accent-red-dark) 100%) !important;
  border: none !important;
  box-shadow: 0 4px 16px rgba(166, 27, 27, 0.3) !important;
}

.action-btn:not(.secondary):hover:not(:disabled) {
  background: linear-gradient(135deg, var(--accent-red-light) 0%, var(--accent-red) 100%) !important;
  box-shadow: 0 8px 24px rgba(166, 27, 27, 0.4);
  transform: translateY(-3px);
}

.action-btn:not(.secondary):active:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(166, 27, 27, 0.3);
}

.action-btn.secondary {
  background: transparent !important;
  border: 1px solid rgba(0, 0, 0, 0.1) !important;
  color: #999;
}

.action-btn.secondary:hover {
  border-color: #999;
  color: var(--modern-black);
  background: rgba(0, 0, 0, 0.02) !important;
  transform: translateY(-2px);
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.hint-text {
  font-size: 12PX;
  color: #999;
  margin-top: 16px;
  letter-spacing: 0.1em;
}

.footer-note {
  text-align: center;
  margin-top: 60px;
  opacity: 0.3;
  animation: footer-appear 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.6s both;
}

@keyframes footer-appear {
  from {
    opacity: 0;
  }
  to {
    opacity: 0.3;
  }
}

.footer-note p {
  font-size: 11PX;
  letter-spacing: 0.4em;
  color: var(--modern-black);
  font-weight: 200;
}

.dialog-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.fade-slide-enter-from {
  opacity: 0;
  transform: translateX(30px);
}

.fade-slide-leave-to {
  opacity: 0;
  transform: translateX(-30px);
}

@media (max-width: 768px) {
  .guide-card {
    margin: 20px 0;
  }
  
  .action-buttons {
    flex-direction: column;
    align-items: stretch;
  }
  
  .action-btn {
    width: 100%;
  }
}
</style>
