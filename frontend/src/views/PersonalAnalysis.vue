<template>
  <div class="personal-analysis-container">
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
        
        <!-- 个人万象 (Active) -->
        <div class="nav-btn-card active" title="Personal Analysis">
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
        <!-- Page Header -->
        <div class="page-zen-header">
            <h1 class="zen-title">个人万象</h1>
            <p class="zen-subtitle">阅读足迹与诗歌偏好的静谧镜像</p>
            <div class="zen-divider"></div>
        </div>

        <div class="analysis-stack">
            <!-- 阅历摘要 (Minimalist & Elevated) -->
            <div class="glass-card stat-hero-row anim-fade-up">
                <div class="stat-hero-item" v-for="(val, label, idx) in { '阅览诗章': userStats.totalReads, '贡献雅评': userStats.reviewCount, '游历时长': userStats.activeDays }" :key="label">
                    <div class="stat-icon-gate">
                        <n-icon v-if="idx===0"><NBook /></n-icon>
                        <n-icon v-if="idx===1"><NChatBubble /></n-icon>
                        <n-icon v-if="idx===2"><NCalendar /></n-icon>
                    </div>
                    <div class="stat-content">
                        <span class="stat-label-hero">{{ label }}</span>
                        <span class="stat-val-hero">
                            {{ val }}
                            <small v-if="label === '阅览诗章'">篇</small>
                            <small v-else-if="label === '贡献雅评'">条</small>
                            <small v-else-if="label === '游历时长'">天</small>
                        </span>
                    </div>
                </div>
                <div class="watermark-icon"><NHeart /></div>
            </div>

            <!-- 主题意向 (Wide Content with Elegant Details) -->
            <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.1s">
                <div class="section-zen-header">
                    <div class="header-accent"></div>
                    <h3>主题意向与内心镜像</h3>
                </div>
                <div class="preference-flex">
                    <div class="chart-container-main">
                        <div ref="preferenceChartRef" style="height: 450px;"></div>
                        <div class="chart-center-text">
                            <span class="center-label">核心意向</span>
                            <span class="center-val">{{ userPreferences[0]?.topic_name || '探索中' }}</span>
                        </div>
                    </div>
                    <div class="pref-detail-panel">
                        <div v-for="pref in userPreferences" :key="pref.topic_id" class="pref-row-styled">
                            <div class="pref-meta">
                                <span class="pref-dot" :style="{ background: pref.color }"></span>
                                <span class="pref-name-text">{{ pref.topic_name }}</span>
                                <span class="pref-percent-text">{{ pref.percentage }}%</span>
                            </div>
                            <div class="pref-bar-bg">
                                <div class="pref-bar-fill" :style="{ width: pref.percentage + '%', background: pref.color }"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 词云统计 (Word Cloud) -->
            <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.15s">
                <div class="section-zen-header">
                    <div class="header-accent"></div>
                    <h3>雅评万象词云</h3>
                </div>
                <div ref="wordCloudRef" style="height: 400px;"></div>
            </div>

            <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.2s">
                <div class="section-zen-header">
                    <div class="header-accent"></div>
                    <h3>诗人-主题流向</h3>
                </div>
                <div ref="poetThemeSankeyRef" style="height: 380px;"></div>
            </div>

            <!-- 格律与节律 (Side by Side Grid) -->
            <div class="viz-grid-row">
                <div class="glass-card viz-subcard anim-fade-up" style="animation-delay: 0.2s">
                    <div class="section-zen-header">
                        <div class="header-accent"></div>
                        <h3>格律形式偏好</h3>
                    </div>
                    <div ref="formChartRef" style="height: 380px;"></div>
                </div>
                <div class="glass-card viz-subcard anim-fade-up" style="animation-delay: 0.3s">
                    <div class="section-zen-header">
                        <div class="header-accent"></div>
                        <h3>每日阅读节律</h3>
                    </div>
                    <div ref="timeChartRef" style="height: 380px;"></div>
                </div>
            </div>
        </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import * as echarts from 'echarts'
import 'echarts-wordcloud'
import { 
  NIcon, NSpin, 
  NCard, NInput, NButton, NProgress, NTag, NModal, NEmpty, NRate
} from 'naive-ui'
import { 
  HomeOutline as NHome, 
  Search as NSearch, 
  PersonOutline as NPersonOutline, 
  GlobeOutline as NGlobeOutline, 
  BookOutline as NBook, 
  ChatbubbleOutline as NChatBubble, 
  CalendarOutline as NCalendar, 
  Heart as NHeart, 
  Sparkles as NSparkles,
  ArrowForwardOutline as NArrowRight, 
  Close as NClose, 
  Refresh as NRefresh, 
  Compass as NCompass, 
  StatsChart as NDataLine, 
  Menu as NMenu, 
  PaperPlane as NSend, 
  HeartOutline as NHeartOutline, 
  PieChart as NPieChart, 
  TimeOutline as NClock, 
  BulbOutline as NBulb
} from '@vicons/ionicons5'

const router = useRouter()
const currentUser = localStorage.getItem('user') || '访客'

const nextTickExec = (fn) => {
  setTimeout(fn, 0)
}

const preferenceChartRef = ref(null)
const timeChartRef = ref(null)
const formChartRef = ref(null)
const wordCloudRef = ref(null)
const poetThemeSankeyRef = ref(null)

let prefChart = null
let timeChart = null
let formChart = null
let wcChart = null
let poetThemeSankeyChart = null

// 用户统计数据
const userStats = ref({
  totalReads: 0,
  reviewCount: 0,
  activeDays: 0
})

const userPreferences = ref([])
const formStats = ref([])
const wordCloudData = ref([])
const timeInsights = ref([])
const poetThemeSankeyData = ref({ nodes: [], links: [] })

// Fetch Data
const fetchData = async () => {
  try {
    const [statsRes, prefRes, timeRes, formRes, wcRes, sankeyRes] = await Promise.all([
      axios.get(`/api/user/${currentUser}/stats`),
      axios.get(`/api/user/${currentUser}/preferences`),
      axios.get(`/api/user/${currentUser}/time-analysis`),
      axios.get(`/api/user/${currentUser}/form-stats`),
      axios.get(`/api/user/${currentUser}/wordcloud`),
      axios.get(`/api/user/${currentUser}/poet-topic-sankey`)
    ])
    
    userStats.value = statsRes.data
    userPreferences.value = prefRes.data.preferences
    timeInsights.value = timeRes.data.insights
    formStats.value = formRes.data
    wordCloudData.value = wcRes.data
    poetThemeSankeyData.value = sankeyRes.data
    
    nextTickExec(() => {
        initCharts()
    })
  } catch (err) {
    // Fallbacks
    userStats.value = { totalReads: 124, reviewCount: 42, activeDays: 28 }
    userPreferences.value = [
        { topic_name: '山水田园', percentage: 45, color: '#cf3f35' },
        { topic_name: '思乡情怀', percentage: 30, color: '#bfa46f' },
        { topic_name: '豪迈边塞', percentage: 25, color: '#1a1a1a' }
    ]
    formStats.value = [ { name: '七绝', value: 45 }, { name: '五律', value: 25 }, { name: '七律', value: 20 }, { name: '其他', value: 10 } ]
    wordCloudData.value = [ { name: '意境', value: 100 }, { name: '深远', value: 80 } ]
    poetThemeSankeyData.value = {
        nodes: [{ name: '李白' }, { name: '杜甫' }, { name: '思乡' }, { name: '山水' }],
        links: [
            { source: '李白', target: '山水', value: 3 },
            { source: '李白', target: '思乡', value: 2 },
            { source: '杜甫', target: '思乡', value: 4 }
        ]
    }
    
    nextTickExec(() => {
        initCharts()
    })
  }
}

const initCharts = () => {
    // 1. 主题偏好 (Premium Doughnut)
    if (preferenceChartRef.value) {
        prefChart = echarts.init(preferenceChartRef.value)
        prefChart.setOption({
            backgroundColor: 'transparent',
            tooltip: { trigger: 'item', backgroundColor: 'rgba(255,255,255,0.9)', textStyle: { color: '#1a1a1a' } },
            series: [{
                type: 'pie',
                radius: ['60%', '85%'],
                center: ['50%', '50%'],
                avoidLabelOverlap: false,
                itemStyle: { 
                    borderRadius: 16, 
                    borderColor: '#fdfbf7', 
                    borderWidth: 4,
                    shadowColor: 'rgba(0,0,0,0.1)',
                    shadowBlur: 10
                },
                label: { show: false },
                emphasis: {
                    scale: true,
                    scaleSize: 10,
                },
                data: userPreferences.value.map(p => ({ 
                    value: p.percentage, 
                    name: p.topic_name, 
                    itemStyle: { 
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: p.color || '#cf3f35' },
                            { offset: 1, color: p.color ? p.color + 'aa' : '#8a1616' }
                        ])
                    } 
                }))
            }]
        })
    }

    // 2. 阅读时间 (Calligraphic Line)
    if (timeChartRef.value) {
        timeChart = echarts.init(timeChartRef.value)
        const data = timeInsights.value.length > 0 ? timeInsights.value : [
            {"time": "子时", "value": 15},
            {"time": "卯时", "value": 10},
            {"time": "午时", "value": 40},
            {"time": "酉时", "value": 85},
            {"time": "亥时", "value": 30}
        ]
        timeChart.setOption({
            backgroundColor: 'transparent',
            grid: { top: 40, bottom: 40, left: 40, right: 30 },
            xAxis: { 
                type: 'category', 
                data: data.map(d => d.time),
                axisLine: { lineStyle: { color: 'rgba(0,0,0,0.05)' } },
                axisLabel: { color: 'var(--text-tertiary)', fontSize: 11, fontFamily: 'Noto Serif SC' }
            },
            yAxis: { show: false },
            tooltip: { trigger: 'axis' },
            series: [{
                data: data.map(d => d.value),
                type: 'line',
                smooth: 0.4,
                symbol: 'circle',
                symbolSize: 8,
                itemStyle: { color: 'var(--cinnabar-red)' },
                lineStyle: { color: 'var(--cinnabar-red)', width: 4, shadowBlur: 15, shadowColor: 'rgba(207,63,53,0.3)' },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(207, 63, 53, 0.25)' },
                        { offset: 1, color: 'rgba(207, 63, 53, 0)' }
                    ])
                }
            }]
        })
    }

    // 3. 格律分布 (Rose Pie)
    if (formChartRef.value) {
        formChart = echarts.init(formChartRef.value)
        formChart.setOption({
            backgroundColor: 'transparent',
            tooltip: { trigger: 'item' },
            series: [{
                name: '格律形式',
                type: 'pie',
                radius: [40, 140],
                center: ['50%', '50%'],
                roseType: 'area',
                itemStyle: { borderRadius: 8 },
                data: formStats.value.map(f => ({
                    value: f.value,
                    name: f.name,
                    itemStyle: { color: new echarts.graphic.LinearGradient(0,0,1,1, [
                        {offset: 0, color: '#cf3f35'},
                        {offset: 1, color: '#8a1616'}
                    ])}
                }))
            }]
        })
    }

    // 4. Word Cloud
    if (wordCloudRef.value && wordCloudData.value.length > 0) {
        wcChart = echarts.init(wordCloudRef.value)
        wcChart.setOption({
            series: [{
                type: 'wordCloud',
                shape: 'circle',
                left: 'center',
                top: 'center',
                width: '90%',
                height: '90%',
                right: null,
                bottom: null,
                sizeRange: [14, 60],
                rotationRange: [-45, 90],
                rotationStep: 45,
                gridSize: 8,
                drawOutOfBound: false,
                textStyle: {
                    fontFamily: 'Noto Serif SC',
                    fontWeight: 'bold',
                    color: function () {
                        return 'rgb(' + [
                            Math.round(Math.random() * 160 + 50),
                            Math.round(Math.random() * 50),
                            Math.round(Math.random() * 50)
                        ].join(',') + ')';
                    }
                },
                emphasis: {
                    textStyle: { shadowBlur: 10, shadowColor: '#333' }
                },
                data: wordCloudData.value
            }]
        })
    }

    if (poetThemeSankeyRef.value && poetThemeSankeyData.value.nodes.length > 0) {
        poetThemeSankeyChart = echarts.init(poetThemeSankeyRef.value)
        poetThemeSankeyChart.setOption({
            tooltip: { trigger: 'item' },
            series: [{
                type: 'sankey',
                data: poetThemeSankeyData.value.nodes,
                links: poetThemeSankeyData.value.links,
                lineStyle: { color: 'gradient', curveness: 0.5, opacity: 0.3 },
                itemStyle: { color: '#A61B1B' },
                label: { color: 'var(--ink-black)', fontFamily: 'Noto Serif SC' }
            }]
        })
    }
}

const handleResize = () => {
    prefChart?.resize()
    timeChart?.resize()
    formChart?.resize()
    wcChart?.resize()
    poetThemeSankeyChart?.resize()
}

onMounted(() => {
    fetchData()
    window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
    window.removeEventListener('resize', handleResize)
    if (prefChart) prefChart.dispose()
    if (timeChart) timeChart.dispose()
    if (formChart) formChart.dispose()
    if (wcChart) wcChart.dispose()
    if (poetThemeSankeyChart) poetThemeSankeyChart.dispose()
})

// Navigation
const goHome = () => router.push('/')
const goToSearch = () => router.push('/search')
const goToGlobalAnalysis = () => router.push('/global-analysis')
const goToPoem = (id) => router.push(`/?poemId=${id}`)
const logout = () => {
  localStorage.removeItem('user')
  router.push('/login')
}
</script>

<style scoped>
.personal-analysis-container {
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  background: var(--gradient-bg);
  color: var(--ink-black);
}


/* Main Layout */
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
    color: var(--ink-black);
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

.analysis-stack {
    display: flex;
    flex-direction: column;
    gap: 50px;
    margin-bottom: 100px;
}

/* Page Header Enhancement */
.page-zen-header {
    margin-bottom: 80px;
    position: relative;
}

.zen-title {
    font-size: 48px;
    background: linear-gradient(180deg, var(--ink-black) 40%, var(--cinnabar-red));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
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

.watermark-icon {
    position: absolute;
    right: -20px;
    bottom: -40px;
    font-size: 200px;
    color: var(--cinnabar-red);
    opacity: 0.03;
    pointer-events: none;
}

/* Viz Section Wide */
.viz-card-elegant {
    padding: 50px !important;
}

.section-zen-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 40px;
}

.header-accent {
    width: 4px;
    height: 24px;
    background: var(--cinnabar-red);
    border-radius: 2px;
}
.header-accent.gold { background: var(--antique-gold); }

.preference-flex {
    display: flex;
    gap: 80px;
    align-items: center;
}

.chart-container-main {
    flex: 1.2;
    position: relative;
}

.chart-center-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    pointer-events: none;
}

.center-label { display: block; font-size: 13px; color: var(--text-tertiary); letter-spacing: 0.1em; }
.center-val { display: block; font-size: 20px; font-weight: 700; color: var(--ink-black); margin-top: 4px; }

.pref-detail-panel {
    flex: 0.8;
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.pref-row-styled { display: flex; flex-direction: column; gap: 12px; }
.pref-meta { display: flex; align-items: center; gap: 12px; font-size: 15px; }
.pref-dot { width: 8px; height: 8px; border-radius: 50%; }
.pref-name-text { flex: 1; font-weight: 500; color: var(--text-secondary); }
.pref-percent-text { font-family: "Playfair Display", serif; font-weight: 700; color: var(--ink-black); font-size: 18px; }

.pref-bar-bg { height: 4px; background: rgba(0,0,0,0.04); border-radius: 2px; }
.pref-bar-fill { height: 100%; border-radius: 2px; transition: width 1.5s var(--ease-smooth); }

/* Grid Viz Row */
.viz-grid-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
}

/* Animation Utils */
.anim-fade-up {
    opacity: 0;
    transform: translateY(30px);
    animation: fadeUp 0.8s var(--ease-smooth) forwards;
}

@keyframes fadeUp {
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 1400px) {
    .preference-flex { flex-direction: column; }
    .viz-grid-row { grid-template-columns: 1fr; }
}
</style>
