<template>
  <div class="analysis-container">
    <!-- 顶部导航 (Consistent with Home) -->
    <nav class="top-nav glass-card">
      <div class="nav-brand">
        <span class="logo-text">诗云</span>
        <span class="edition-badge">Zen Edition</span>
      </div>
      
      <div class="nav-actions">
        <!-- 主页 -->
        <div class="nav-btn-card" @click="goHome" title="Home">
            <n-icon><NHome /></n-icon>
            <span>主页</span>
        </div>
        
        <!-- 个人万象 -->
        <div class="nav-btn-card" @click="goToPersonalAnalysis" title="Personal Analysis">
             <n-icon><NPersonOutline /></n-icon>
             <span>个人万象</span>
        </div>
        
        <!-- 全站万象 -->
        <div class="nav-btn-card" @click="goToGlobalAnalysis" title="Global Analysis">
             <n-icon><NGlobeOutline /></n-icon>
             <span>全站万象</span>
        </div>

        <div class="divider-vertical"></div>

        <!-- User Profile -->
        <div class="user-area">
             <div v-if="currentUser !== '访客'" class="user-greeting" @click="logout" title="Logout">
                <span class="user-name">{{ currentUser }}</span>
                <span class="logout-hint">离席</span>
             </div>
             <div v-else class="login-prompt" @click="$router.push('/login')">
                Login
             </div>
        </div>
      </div>
    </nav>

    <main class="analysis-main anim-enter">
        <div class="page-zen-header">
            <h1 class="zen-title">观象</h1>
            <p class="zen-subtitle">系统全览与深度数据洞察</p>
            <div class="zen-divider"></div>
        </div>

        <div class="dashboard-grid" v-if="loaded && stats.counts">
          <!-- 统计卡片 -->
          <div class="stat-zen-row">
            <div class="stat-zen-item">
              <span class="stat-label">雅士</span>
              <span class="stat-value">{{ stats.counts.users }}</span>
            </div>
            <div class="stat-zen-item border-left">
              <span class="stat-label">诗章</span>
              <span class="stat-value">{{ stats.counts.poems }}</span>
            </div>
            <div class="stat-zen-item border-left">
              <span class="stat-label">雅评</span>
              <span class="stat-value">{{ stats.counts.reviews }}</span>
            </div>
          </div>

          <!-- 图表区域 -->
          <div class="charts-area">
            <!-- 词云 -->
            <div class="glass-card viz-card large">
              <div class="section-zen-header">
                <h3><n-icon><NSparkles /></n-icon> 热门关键词</h3>
              </div>
              <div ref="wordCloudRef" class="chart-body"></div>
            </div>

            <!-- 诗韵雷达 (System Radar) -->
            <div class="glass-card viz-card">
              <div class="section-zen-header">
                <h3><n-icon><NCompass /></n-icon> 诗韵音律雷达</h3>
              </div>
              <div ref="radarRef" class="chart-body"></div>
            </div>

            <!-- 诗人-主题流向 (Sankey) -->
            <div class="glass-card viz-card wide">
              <div class="section-zen-header">
                <h3><n-icon><NGlobeOutline /></n-icon> 诗人与主题关系</h3>
              </div>
              <div ref="sankeyRef" class="chart-body"></div>
            </div>
          </div>
        </div>
        
        <div v-else class="loading-wrapper">
            <n-spin size="large" />
            <div class="loading-text">观象推演中...</div>
        </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { 
  HomeOutline as NHome,
  PersonOutline as NPersonOutline,
  GlobeOutline as NGlobeOutline,
  Compass as NCompass,
  Sparkles as NSparkles
} from '@vicons/ionicons5'
import { NIcon, NSpin } from 'naive-ui'
import axios from 'axios'
import * as echarts from 'echarts'
import 'echarts-wordcloud'

const router = useRouter()
const loaded = ref(false)
const stats = ref({})
const currentUser = localStorage.getItem('user') || '访客'

const wordCloudRef = ref(null)
const radarRef = ref(null)
const sankeyRef = ref(null)

let chartInstances = []

const goHome = () => router.push('/')
const goToPersonalAnalysis = () => router.push('/personal-analysis')
const goToGlobalAnalysis = () => router.push('/global-analysis')
const logout = () => {
    localStorage.removeItem('user')
    router.push('/login')
}

const initCharts = async () => {
  chartInstances.forEach(c => c.dispose())
  chartInstances = []

  try {
    const statsRes = await axios.get('/api/visual/stats', {
      params: { user_id: currentUser === '访客' ? '' : currentUser }
    })
    stats.value = statsRes.data
    
    await nextTick()
    
    // 雷达图
    if (radarRef.value && stats.value.radar_data) {
      const c = echarts.init(radarRef.value)
      c.setOption({
        color: ['#A61B1B'],
        tooltip: {},
        radar: {
          indicator: stats.value.radar_data.indicator,
          shape: 'circle',
          axisName: { color: 'var(--text-secondary)' },
          splitArea: { show: false },
          splitLine: { lineStyle: { color: 'rgba(166,27,27,0.1)' } }
        },
        series: [{
          type: 'radar',
          data: [{
            value: stats.value.radar_data.value,
            areaStyle: { color: 'rgba(166,27,27,0.2)' }
          }]
        }]
      })
      chartInstances.push(c)
    }

    // 桑基图
    if (sankeyRef.value && stats.value.sankey_data) {
      const c = echarts.init(sankeyRef.value)
      c.setOption({
        series: [{
          type: 'sankey',
          data: stats.value.sankey_data.nodes,
          links: stats.value.sankey_data.links,
          lineStyle: { color: 'gradient', curveness: 0.5, opacity: 0.3 },
          itemStyle: { color: '#A61B1B' },
          label: { color: 'var(--ink-black)', fontFamily: 'Noto Serif SC' }
        }]
      })
      chartInstances.push(c)
    }

    // 词云
    const wcRes = await axios.get('/api/visual/wordcloud', {
      params: { user_id: currentUser === '访客' ? '' : currentUser }
    })
    if (wordCloudRef.value) {
      const c = echarts.init(wordCloudRef.value)
      c.setOption({
        series: [{
          type: 'wordCloud',
          textStyle: {
            fontFamily: 'Noto Serif SC',
            color: () => `rgb(${[Math.round(Math.random()*150), 50, 50].join(',')})`
          },
          data: wcRes.data
        }]
      })
      chartInstances.push(c)
    }

  } catch(e) {
    void e
  }
  loaded.value = true
}

const handleResize = () => chartInstances.forEach(c => c.resize())

onMounted(() => {
  initCharts()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.analysis-container {
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  background: var(--gradient-bg);
  color: var(--ink-black);
}


/* Main Stage */
.analysis-main {
    flex: 1;
    max-width: var(--content-max-width);
    margin: 40px auto;
    padding: 0 var(--content-padding);
    width: 100%;
}

.page-zen-header {
    text-align: center;
    margin-bottom: 60px;
}

.zen-title {
    font-family: "Noto Serif SC", serif;
    font-size: 42px;
    font-weight: 700;
    letter-spacing: 0.2em;
    margin-bottom: 8px;
}

.zen-subtitle {
    font-size: 14px;
    color: var(--text-tertiary);
}

.zen-divider {
    width: 60px;
    height: 2px;
    background: var(--cinnabar-red);
    margin: 24px auto;
    opacity: 0.5;
}

/* Stats */
.stat-zen-row {
    display: flex;
    justify-content: space-around;
    padding: 32px;
    background: var(--paper-white);
    border-radius: var(--radius-main);
    margin-bottom: 40px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}

.stat-zen-item {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.stat-zen-item.border-left { border-left: 1px solid rgba(0,0,0,0.05); }

.stat-label { font-size: 12px; color: var(--text-tertiary); margin-bottom: 8px; }
.stat-value {
    font-size: 42px;
    font-weight: 700;
    font-family: "Playfair Display", serif;
    color: var(--cinnabar-red);
}

/* Charts */
.charts-area { display: grid; grid-template-columns: repeat(2, 1fr); gap: 32px; }
.viz-card { padding: 32px; min-height: 400px; }
.viz-card.wide { grid-column: span 2; height: 350px; }

.section-zen-header h3 {
    font-family: "Noto Serif SC", serif;
    font-size: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
}

.section-zen-header h3 .n-icon { color: var(--cinnabar-red); }

.chart-body { width: 100%; height: 300px; }

.loading-wrapper {
    height: 50vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 20px;
}
</style>
