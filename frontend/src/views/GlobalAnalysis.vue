<template>
  <div class="global-analysis-container">
    <!-- 顶部导航 (Consistent with Home) -->
    <nav class="top-nav glass-card">
      <div class="nav-brand" @click="goHome">
        <span class="logo-text">诗云</span>
        <span class="edition-badge">Zen Edition</span>
      </div>
      
      <div class="nav-actions">
        <!-- 搜索 -->
        <div class="nav-btn-card" @click="goToSearch" title="Search">
            <n-icon><NSearch /></n-icon>
            <span>搜索</span>
        </div>
        
        <!-- 个人万象 -->
        <div class="nav-btn-card" @click="goToPersonalAnalysis" title="Personal Analysis">
             <n-icon><NPersonOutline /></n-icon>
             <span>个人万象</span>
        </div>
        
        <!-- 全站万象 (Active) -->
        <div class="nav-btn-card active" title="Global Analysis">
             <n-icon><NGlobeOutline /></n-icon>
             <span>全站万象</span>
        </div>

        <div class="divider-vertical"></div>

        <!-- User Profile -->
        <div class="user-area">
             <div v-if="currentUser !== '访客'" class="user-greeting" @click="$router.push('/profile')" title="个人信息">
                <n-icon class="user-icon"><NPersonOutline /></n-icon>
                <span class="user-name">{{ currentUser }}</span>
             </div>
             <div v-else class="login-prompt" @click="$router.push('/login')">
                Login
             </div>
        </div>
      </div>
    </nav>

    <!-- Main Stage -->
    <main class="analysis-main anim-enter">
        <!-- 页面头部 -->
        <div class="page-zen-header">
            <h1 class="zen-title">全站万象</h1>
            <p class="zen-subtitle">探索诗云社区的数据宏观图景</p>
            <div class="zen-divider"></div>
        </div>

        <div class="mode-content overview-mode">
            <!-- 全站统计 -->
            <div class="stat-hero-row glass-card anim-fade-up">
                <div class="stat-hero-item" v-for="(item, idx) in [
                    { label: '诗歌馆藏', value: globalStats.totalPoems, icon: NBookOutline },
                    { label: '活跃墨客', value: globalStats.totalUsers, icon: NPeopleOutline },
                    { label: '累计雅评', value: globalStats.totalReviews, icon: NChatOutline },
                    { label: '互动频次', value: globalStats.avgEngagement, icon: NTrendingUpOutline }
                ]" :key="item.label">
                    <div class="stat-icon-gate">
                        <n-icon><component :is="item.icon" /></n-icon>
                    </div>
                    <div class="stat-content">
                        <span class="stat-label-hero">{{ item.label }}</span>
                        <span class="stat-val-hero">
                            {{ item.value }}
                            <small v-if="item.label === '诗歌馆藏'">篇</small>
                            <small v-else-if="item.label === '活跃墨客'">人</small>
                            <small v-else-if="item.label === '累计雅评'">条</small>
                            <small v-else-if="item.label === '互动频次'"></small>
                        </span>
                    </div>
                </div>
                <div class="watermark-icon"><NGlobeOutline /></div>
            </div>

            <div class="analysis-grid">
                <!-- 热门排行 -->
                <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.1s">
                    <div class="section-zen-header">
                        <div class="header-accent"></div>
                        <h3>热门诗篇</h3>
                        <n-select v-model:value="popularTimeRange" :options="timeRangeOptions" size="small" style="width: 100px" />
                    </div>
                    <div class="popular-list">
                        <div v-for="(poem, index) in popularPoems" :key="poem.id" class="popular-item" :class="{ 'top-item': index < 3 }">
                            <div class="rank">{{ index + 1 }}</div>
                            <div class="p-info">
                                <span class="p-title">{{ poem.title }}</span>
                                <span class="p-author">{{ poem.author }} · {{ poem.dynasty }}</span>
                            </div>
                            <div class="p-stats">
                                <span><n-icon><NChatOutline /></n-icon> {{ poem.review_count || 0 }}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 分布图表组 -->
                <div class="charts-column">
                    <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.15s">
                        <div class="section-zen-header">
                            <div class="header-accent"></div>
                            <h3>主题宏图</h3>
                        </div>
                        <div ref="themeChartRef" style="height: 240px;"></div>
                    </div>
                    <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.2s">
                        <div class="section-zen-header">
                            <div class="header-accent"></div>
                            <h3>朝代热度</h3>
                        </div>
                        <div ref="dynastyChartRef" style="height: 240px;"></div>
                    </div>
                </div>
            </div>
        </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import * as echarts from 'echarts'
import { 
  NIcon, 
  NButton,
  NButtonGroup,
  NSelect,
  NTag
} from 'naive-ui'
import { 
  HomeOutline as NHome,
  GlobeOutline as NGlobeOutline,
  PersonOutline as NPersonOutline,
  BookOutline as NBookOutline,
  PeopleOutline as NPeopleOutline,
  ChatbubbleEllipsesOutline as NChatOutline,
  TrendingUpOutline as NTrendingUpOutline,
  HeartOutline as NHeartOutline,
  RefreshOutline as NRefreshOutline,
  Search as NSearch,
  TimerOutline as NClock,
  BriefcaseOutline as NCompare,
  AppsOutline as NAppsOutline,
  GitCompareOutline as NGitCompareOutline
} from '@vicons/ionicons5'

const router = useRouter()
const currentUser = localStorage.getItem('user') || '访客'

const popularTimeRange = ref('week')

// 图表引用
const themeChartRef = ref(null)
const dynastyChartRef = ref(null)

let charts = []

// 全站统计数据
const globalStats = ref({
  totalUsers: 0,
  totalPoems: 0,
  totalReviews: 0,
  totalLikes: 0,
  totalViews: 0,
  totalShares: 0,
  avgEngagement: '0%',
  todayNewUsers: 0,
  todayReviews: 0
})

// 热门诗歌数据
const popularPoems = ref([])

// 主题分布数据
const themeDistribution = ref([])

// 朝代分布数据
const dynastyDistribution = ref([])

// 词云数据
const wordCloudData = ref([])

// 选择器选项
const timeRangeOptions = [
  { label: '今日', value: 'today' },
  { label: '本周', value: 'week' },
  { label: '本月', value: 'month' }
]

// 获取全站统计数据
const fetchGlobalStats = async () => {
  try {
    const res = await axios.get('/api/global/stats')
    globalStats.value = res.data
  } catch (error) {
    void error
  }
}

// 获取热门诗歌
const fetchPopularPoems = async () => {
  try {
    const res = await axios.get(`/api/global/popular-poems?time_range=${popularTimeRange.value}`)
    // 按评论数排序
    popularPoems.value = res.data.sort((a, b) => {
      const reviewCountA = a.review_count || 0
      const reviewCountB = b.review_count || 0
      return reviewCountB - reviewCountA
    })
  } catch (error) {
    void error
  }
}

// 获取主题分布
const fetchThemeDistribution = async () => {
  try {
    const res = await axios.get('/api/global/theme-distribution')
    themeDistribution.value = res.data
  } catch (error) {
    void error
  }
}

// 获取朝代分布
const fetchDynastyDistribution = async () => {
  try {
    const res = await axios.get('/api/global/dynasty-distribution')
    dynastyDistribution.value = res.data
  } catch (error) {
    void error
  }
}

// 获取词云数据
const fetchWordCloudData = async () => {
  try {
    const res = await axios.get('/api/global/wordcloud')
    wordCloudData.value = res.data
  } catch (error) {
    void error
  }
}

// 初始化图表
const initCharts = () => {
  charts.forEach(c => c.dispose())
  charts = []

  // 确保DOM元素已挂载
  const isElementMounted = (ref) => ref && ref.value && typeof ref.value === 'object' && 'clientWidth' in ref.value

  if (isElementMounted(themeChartRef) && themeDistribution.value.length > 0) {
    const c1 = echarts.init(themeChartRef.value)
    c1.setOption({
      tooltip: { 
        trigger: 'item', 
        backgroundColor: 'rgba(255,255,255,0.9)', 
        textStyle: { color: '#1a1a1a' },
        formatter: '{b}: {c} ({d}%)'
      },
      series: [{
        type: 'pie',
        radius: ['60%', '85%'],
        itemStyle: { 
          borderRadius: 16, 
          borderColor: '#fdfbf7', 
          borderWidth: 4,
          shadowColor: 'rgba(0,0,0,0.1)',
          shadowBlur: 10
        },
        data: themeDistribution.value.map(item => ({
          value: item.value,
          name: item.name,
          itemStyle: { 
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#cf3f35' },
              { offset: 1, color: '#8a1616' }
            ])
          }
        })),
        label: {
          show: true,
          formatter: '{b}: {d}%',
          position: 'outside',
          color: '#333',
          fontSize: 12,
          fontWeight: 'bold'
        },
        labelLine: {
          show: true,
          length: 20,
          length2: 30,
          lineStyle: {
            color: '#999'
          }
        }
      }]
    })
    charts.push(c1)
  }
  
  if (isElementMounted(dynastyChartRef) && dynastyDistribution.value.length > 0) {
    const c2 = echarts.init(dynastyChartRef.value)
    c2.setOption({
      xAxis: { 
        type: 'category', 
        data: dynastyDistribution.value.map(d => d.name), 
        axisLine: { lineStyle: { color: 'rgba(0,0,0,0.05)' } }, 
        axisTick: { show: false },
        axisLabel: { color: 'var(--text-tertiary)', fontSize: 11, fontFamily: 'Noto Serif SC' }
      },
      yAxis: { show: false },
      series: [{
        data: dynastyDistribution.value.map(d => d.value),
        type: 'bar',
        itemStyle: { 
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#cf3f35' },
            { offset: 1, color: 'rgba(207, 63, 53, 0.3)' }
          ]),
          borderRadius: [4, 4, 0, 0]
        }
      }],
      grid: { top: 20, bottom: 30, left: 10, right: 10 }
    })
    charts.push(c2)
  }
}

const handleResize = () => charts.forEach(c => c.resize())

onMounted(() => {
    // 获取所有数据
    fetchGlobalStats()
    fetchPopularPoems()
    fetchThemeDistribution()
    fetchDynastyDistribution()
    fetchWordCloudData()
    
    // 延迟初始化图表以确保数据加载完成
    setTimeout(() => {
        initCharts()
        window.addEventListener('resize', handleResize)
    }, 500)
})

onUnmounted(() => {
    window.removeEventListener('resize', handleResize)
})

// 监听时间范围变化
watch(popularTimeRange, () => {
    fetchPopularPoems()
})

const goHome = () => router.push('/')
const goToSearch = () => router.push('/search')
const goToPersonalAnalysis = () => router.push('/personal-analysis')
const logout = () => {
  localStorage.removeItem('user')
  router.push('/login')
}
</script>

<style scoped>
.global-analysis-container {
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  background: var(--gradient-bg);
  color: var(--ink-black);
}


/* Layout */
.analysis-main {
    flex: 1;
    max-width: var(--content-max-width);
    margin: 40px auto;
    padding: 0 var(--content-padding);
    width: 100%;
}

.page-zen-header {
    text-align: center;
    margin-bottom: 80px;
    position: relative;
}

.zen-title {
    font-family: "Noto Serif SC", serif;
    font-size: 48px;
    font-weight: 700;
    letter-spacing: 0.2em;
    margin-bottom: 8px;
    background: linear-gradient(180deg, var(--ink-black) 40%, var(--cinnabar-red));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

.zen-subtitle {
    font-size: 14px;
    color: var(--text-tertiary);
    letter-spacing: 0.1em;
}

.zen-divider {
    width: 60px;
    height: 2px;
    background: var(--cinnabar-red);
    margin: 24px auto;
    opacity: 0.5;
}

/* Mode Switcher */
.mode-switcher-container {
    display: flex;
    justify-content: center;
    margin-bottom: 60px;
}

.mode-switcher {
    display: flex;
    background: var(--paper-white);
    border-radius: 50px;
    padding: 6px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
}

.mode-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 28px;
    border-radius: 40px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    color: var(--text-tertiary);
    font-size: 14px;
    font-weight: 500;
    position: relative;
    overflow: hidden;
}

.mode-item .n-icon {
    font-size: 18px;
    transition: all 0.3s ease;
}

.mode-item:hover {
    color: var(--text-secondary);
    background: rgba(0,0,0,0.03);
}

.mode-item.active {
    color: white;
    background: var(--cinnabar-red);
    box-shadow: 0 4px 12px rgba(207, 63, 53, 0.3);
}

.mode-item.active .n-icon {
    transform: scale(1.1);
}

/* Stat Hero Row */
.stat-hero-row {
    display: flex;
    justify-content: space-around;
    padding: 60px 40px !important;
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(207,63,53,0.02)) !important;
    border: 1px solid rgba(207,63,53,0.1) !important;
    margin-bottom: 50px;
}

.stat-hero-item {
    display: flex;
    align-items: center;
    gap: 20px;
    z-index: 1;
}

.stat-icon-gate {
    width: 64px;
    height: 64px;
    background: rgba(207,63,53,0.05);
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--cinnabar-red);
    font-size: 28px;
    border: 1px solid rgba(207,63,53,0.1);
}

.stat-content {
    display: flex;
    flex-direction: column;
}

.stat-val-hero {
    display: block;
    font-size: 40px;
    font-weight: 700;
    font-family: "Playfair Display", serif;
    color: var(--ink-black);
    line-height: 1.2;
}

.stat-label-hero {
    font-size: 13px;
    color: var(--text-tertiary);
    letter-spacing: 0.1em;
    margin-bottom: 6px;
    font-weight: 500;
}

.stat-val-hero small {
    font-size: 14px;
    color: var(--text-tertiary);
    font-weight: 400;
    margin-left: 4px;
}

.watermark-icon {
    position: absolute;
    right: -20px;
    bottom: -40px;
    font-size: 200px;
    color: var(--cinnabar-red);
    opacity: 0.03;
    pointer-events: none;
}

/* Grid */
.analysis-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    margin-bottom: 60px;
}

.section-zen-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 32px;
    justify-content: space-between;
}

.header-accent {
    width: 4px;
    height: 24px;
    background: var(--cinnabar-red);
    border-radius: 2px;
}

.section-zen-header h3 {
    font-family: "Noto Serif SC", serif;
    font-size: 18px;
    font-weight: 600;
    color: var(--ink-black);
    margin: 0;
}

.section-card {
    padding: 40px !important;
}

/* Popular List */
.popular-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.popular-item {
    display: flex;
    align-items: center;
    padding: 16px 20px;
    background: rgba(0,0,0,0.02);
    border-radius: var(--radius-sub);
    transition: all 0.3s ease;
}

.popular-item:hover {
    background: rgba(0,0,0,0.04);
    transform: translateX(4px);
}

.popular-item.top-item .rank {
    color: var(--cinnabar-red);
    font-weight: 800;
}

.rank {
    width: 32px;
    font-family: "Playfair Display", serif;
    font-size: 20px;
    color: var(--text-tertiary);
}

.p-info {
    flex: 1;
    display: flex;
    flex-direction: column;
}

.p-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
}

.p-author {
    font-size: 11px;
    color: var(--text-tertiary);
}

.p-stats {
    font-size: 12px;
    color: var(--text-secondary);
}

/* Viz Cards */
.viz-card-elegant {
    padding: 40px !important;
}

.viz-card-large {
    padding: 40px !important;
    margin-bottom: 60px;
}

.mode-content {
    margin-bottom: 60px;
    width: 100%;
}

.charts-column {
    display: flex;
    flex-direction: column;
    gap: 40px;
}

/* Animation */
.anim-fade-up {
    opacity: 0;
    transform: translateY(30px);
    animation: fadeUp 0.8s var(--ease-smooth) forwards;
}

@keyframes fadeUp {
    to { opacity: 1; transform: translateY(0); }
}

/* Responsive */
@media (max-width: 1024px) {
    .analysis-grid {
        grid-template-columns: 1fr;
    }
    
    .stat-hero-row {
        flex-wrap: wrap;
        gap: 30px;
    }
    
    .stat-hero-item {
        min-width: 45%;
    }
    
    .mode-switcher {
        padding: 4px;
    }
    
    .mode-item {
        padding: 10px 20px;
        font-size: 13px;
    }
    
    .mode-item .n-icon {
        font-size: 16px;
    }
}

@media (max-width: 768px) {
    .zen-title {
        font-size: 36px;
    }
    
    .stat-hero-row {
        padding: 40px 20px !important;
    }
    
    .stat-hero-item {
        min-width: 100%;
        gap: 16px;
    }
    
    .stat-icon-gate {
        width: 56px;
        height: 56px;
        font-size: 24px;
    }
    
    .stat-val-hero {
        font-size: 32px;
    }
    
    .mode-switcher {
        flex-direction: column;
        width: 100%;
        border-radius: 20px;
    }
    
    .mode-item {
        width: 100%;
        justify-content: center;
        padding: 12px;
    }
}
</style>
