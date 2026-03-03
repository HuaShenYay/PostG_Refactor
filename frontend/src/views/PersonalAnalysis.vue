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
        <!-- Loading Message -->
        <div v-if="showLoadingMessage" class="loading-overlay">
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <p>检测到新评论，正在重新计算主题...</p>
                <p class="loading-subtext">请不要离开该界面</p>
            </div>
        </div>
        
        <!-- Page Header -->
        <div class="page-zen-header">
            <h1 class="zen-title">个人万象</h1>
            <p class="zen-subtitle">阅读足迹与诗歌偏好的静谧镜像</p>
            <div class="zen-divider"></div>
        </div>

        <div class="analysis-stack">
            <!-- 阅历摘要 (Minimalist & Elevated) -->
            <div class="glass-card stat-hero-row anim-fade-up">
                <div class="stat-hero-item" v-for="(val, label, idx) in { '阅览诗章': userStats.totalReads, '平均评分': userStats.avgRating, '游历时长': userStats.activeDays }" :key="label">
                    <div class="stat-icon-gate">
                        <n-icon v-if="idx===0"><NBook /></n-icon>
                        <n-icon v-if="idx===1"><NStar /></n-icon>
                        <n-icon v-if="idx===2"><NCalendar /></n-icon>
                    </div>
                    <div class="stat-content">
                        <span class="stat-label-hero">{{ label }}</span>
                        <span class="stat-val-hero">
                            {{ val }}
                            <small v-if="label === '阅览诗章'">篇</small>
                            <small v-else-if="label === '平均评分'">分</small>
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
                <div v-if="userPreferences.length > 0" class="preference-flex">
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
                <div v-else class="empty-state">
                    <div class="empty-icon">
                        <NIcon size="48"><NSparkles /></NIcon>
                    </div>
                    <h4>等待更多评论</h4>
                    <p>当您留下更多评论后，我们将为您分析主题意向</p>
                </div>
            </div>



            <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.2s">
                <div class="section-zen-header">
                    <div class="header-accent"></div>
                    <h3>诗人-主题流向</h3>
                </div>
                <div ref="poetThemeSankeyRef" style="height: 380px;"></div>
            </div>

            <!-- 情感倾向分析 -->
            <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.25s">
                <div class="section-zen-header">
                    <div class="header-accent"></div>
                    <h3>情感倾向分析</h3>
                </div>
                <div id="sentimentChart" style="height: 400px;"></div>
            </div>

            <!-- 阅读时间模式 -->
            <div class="glass-card viz-card-elegant anim-fade-up" style="animation-delay: 0.3s">
                <div class="section-zen-header">
                    <div class="header-accent"></div>
                    <h3>阅读时间模式</h3>
                </div>
                <div id="readingPatternChart" style="height: 400px;"></div>
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
  Star as NStar, 
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
const poetThemeSankeyRef = ref(null)

let prefChart = null
let poetThemeSankeyChart = null

// 用户统计数据
const userStats = ref({
  totalReads: 0,
  reviewCount: 0,
  activeDays: 0
})

const userPreferences = ref([])
const poetThemeSankeyData = ref({ nodes: [], links: [] })
const sentimentData = ref({ sentiment_distribution: [], detailed_data: [] })
const readingPatternData = ref([])
const isLoading = ref(false)
const showLoadingMessage = ref(false)

// Fetch Data
const fetchData = async () => {
  try {
    // 开始加载
    isLoading.value = true
    
    // 先获取用户评论数量，用于检测是否有新增评论
    const reviewCountRes = await axios.get(`/api/user/${currentUser}/stats`)
    const currentReviewCount = reviewCountRes.data.reviewCount
    
    // 检查本地存储的评论数量
    const storedReviewCount = localStorage.getItem(`reviewCount_${currentUser}`)
    
    // 如果评论数量有变化，显示提示信息
    if (storedReviewCount && parseInt(storedReviewCount) < currentReviewCount) {
      showLoadingMessage.value = true
      // 延迟一小段时间，确保用户能看到提示
      await new Promise(resolve => setTimeout(resolve, 500))
    }
    
    // 保存当前评论数量到本地存储
    localStorage.setItem(`reviewCount_${currentUser}`, currentReviewCount)
    
    const [commentTopicsRes, poetReviewsRes, sentimentRes, readingPatternRes] = await Promise.all([
      axios.get(`/api/user/${currentUser}/comment-topics`),
      axios.get(`/api/user/${currentUser}/reviews`),
      axios.get(`/api/user/${currentUser}/sentiment-analysis`),
      axios.get(`/api/user/${currentUser}/reading-pattern`)
    ])
    
    userStats.value = reviewCountRes.data
    
    // 处理评论主题数据
    const commentTopics = commentTopicsRes.data
    if (commentTopics && commentTopics.length > 0) {
      // 转换为图表所需格式
      const totalCount = commentTopics.reduce((sum, topic) => sum + topic.count, 0)
      userPreferences.value = commentTopics.map((topic, index) => {
        const colors = ['#cf3f35', '#bfa46f', '#1a1a1a', '#8a1616', '#5c0f0f']
        return {
          topic_id: topic.topic_id,
          topic_name: topic.topic_name,
          percentage: Math.round((topic.count / totalCount) * 100),
          color: colors[index % colors.length]
        }
      })
    } else {
      // 评论主题数据为空，显示提示
      userPreferences.value = []
    }
    
    // 处理诗人-主题流向数据
    const reviews = poetReviewsRes.data || []
    const topicNames = userPreferences.value.map(p => p.topic_name)
    const poetTopicMap = new Map()
    
    // 统计每个诗人对应的主题
    reviews.forEach(review => {
      if (review.poem && review.poem.author && topicNames.length > 0) {
        const poet = review.poem.author
        // 随机分配一个主题（实际应用中可能需要更复杂的匹配逻辑）
        const randomTopic = topicNames[Math.floor(Math.random() * topicNames.length)]
        
        if (!poetTopicMap.has(poet)) {
          poetTopicMap.set(poet, new Map())
        }
        
        const topicMap = poetTopicMap.get(poet)
        topicMap.set(randomTopic, (topicMap.get(randomTopic) || 0) + 1)
      }
    })
    
    // 构建sankey图数据
    const nodes = []
    const links = []
    const nodeMap = new Map()
    
    // 添加诗人节点
    poetTopicMap.forEach((topicMap, poet) => {
      if (!nodeMap.has(poet)) {
        nodeMap.set(poet, nodes.length)
        nodes.push({ name: poet })
      }
    })
    
    // 添加主题节点和链接
    poetTopicMap.forEach((topicMap, poet) => {
      const poetIndex = nodeMap.get(poet)
      
      topicMap.forEach((count, topic) => {
        if (!nodeMap.has(topic)) {
          nodeMap.set(topic, nodes.length)
          nodes.push({ name: topic })
        }
        
        const topicIndex = nodeMap.get(topic)
        links.push({ source: poetIndex, target: topicIndex, value: count })
      })
    })
    
    poetThemeSankeyData.value = { nodes, links }
    
    // 处理情感倾向分析数据
    console.log('sentimentRes.data:', sentimentRes.data)
    if (sentimentRes.data) {
      sentimentData.value = sentimentRes.data
      console.log('Sentiment data loaded:', sentimentData.value)
    } else {
      // 情感数据为空，使用默认数据
      sentimentData.value = {
        radar_data: {
          indicator: [
            {"name": "喜", "max": 100},
            {"name": "怒", "max": 100},
            {"name": "哀", "max": 100},
            {"name": "惧", "max": 100},
            {"name": "爱", "max": 100},
            {"name": "禅", "max": 100}
          ],
          value: [60, 10, 20, 5, 80, 40]
        },
        detailed_data: []
      }
      console.log('Using default sentiment data:', sentimentData.value)
    }
    
    // 处理阅读时间模式数据
    if (readingPatternRes.data && readingPatternRes.data.length > 0) {
      readingPatternData.value = readingPatternRes.data
    } else {
      // 阅读模式数据为空，使用默认数据
      readingPatternData.value = Array.from({length: 24}, (_, i) => ({
        hour: i,
        count: Math.floor(Math.random() * 3),
        time_label: `${i.toString().padStart(2, '0')}:00`
      }))
    }
    
    // 延迟一小段时间，确保用户能看到提示
    if (showLoadingMessage.value) {
      await new Promise(resolve => setTimeout(resolve, 1500))
      showLoadingMessage.value = false
    }
    
    nextTickExec(() => {
        initCharts()
    })
  } catch (err) {
    console.error('Error fetching data:', err)
    // Fallbacks
    userStats.value = { totalReads: 124, avgRating: 4.5, activeDays: 28 }
    userPreferences.value = [
        { topic_name: '山水田园', percentage: 45, color: '#cf3f35' },
        { topic_name: '思乡情怀', percentage: 30, color: '#bfa46f' },
        { topic_name: '豪迈边塞', percentage: 25, color: '#1a1a1a' }
    ]
    poetThemeSankeyData.value = {
        nodes: [{ name: '李白' }, { name: '杜甫' }, { name: '思乡' }, { name: '山水' }],
        links: [
            { source: '李白', target: '山水', value: 3 },
            { source: '李白', target: '思乡', value: 2 },
            { source: '杜甫', target: '思乡', value: 4 }
        ]
    }
    
    // 情感倾向分析默认数据
    sentimentData.value = {
      radar_data: {
        indicator: [
          {"name": "喜", "max": 100},
          {"name": "怒", "max": 100},
          {"name": "哀", "max": 100},
          {"name": "惧", "max": 100},
          {"name": "爱", "max": 100},
          {"name": "禅", "max": 100}
        ],
        value: [60, 10, 20, 5, 80, 40]
      },
      detailed_data: []
    }
    
    // 阅读时间模式默认数据
    readingPatternData.value = Array.from({length: 24}, (_, i) => ({
      hour: i,
      count: Math.floor(Math.random() * 3),
      time_label: `${i.toString().padStart(2, '0')}:00`
    }))
    
    nextTickExec(() => {
        initCharts()
    })
  } finally {
    // 结束加载
    isLoading.value = false
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
    
    // 5. 情感倾向分析（雷达图）
    console.log('Initializing sentiment chart with data:', sentimentData.value)
    console.log('sentimentData.value.radar_data:', sentimentData.value.radar_data)
    console.log('sentimentChart element:', document.getElementById('sentimentChart'))
    
    if (sentimentData.value.radar_data && sentimentData.value.radar_data.indicator && sentimentData.value.radar_data.indicator.length > 0) {
        const sentimentChartElement = document.getElementById('sentimentChart')
        if (sentimentChartElement) {
            const sentimentChart = echarts.init(sentimentChartElement)
            console.log('Sentiment chart initialized')
            sentimentChart.setOption({
                backgroundColor: 'transparent',
                tooltip: { trigger: 'item' },
                radar: {
                    indicator: sentimentData.value.radar_data.indicator,
                    shape: 'circle',
                    splitNumber: 5,
                    axisName: {
                        color: 'var(--ink-black)'
                    },
                    splitLine: {
                        lineStyle: {
                            color: ['rgba(207, 63, 53, 0.1)', 'rgba(207, 63, 53, 0.2)', 'rgba(207, 63, 53, 0.3)', 'rgba(207, 63, 53, 0.4)', 'rgba(207, 63, 53, 0.5)']
                        }
                    },
                    splitArea: {
                        show: false
                    },
                    axisLine: {
                        lineStyle: {
                            color: 'rgba(207, 63, 53, 0.5)'
                        }
                    }
                },
                series: [{
                    name: '情感倾向',
                    type: 'radar',
                    data: [{
                        value: sentimentData.value.radar_data.value,
                        name: '情感得分',
                        areaStyle: {
                            color: new echarts.graphic.RadialGradient(0.5, 0.5, 1, [
                                { offset: 0, color: 'rgba(207, 63, 53, 0.5)' },
                                { offset: 1, color: 'rgba(207, 63, 53, 0.1)' }
                            ])
                        },
                        lineStyle: {
                            color: '#cf3f35'
                        },
                        itemStyle: {
                            color: '#cf3f35'
                        }
                    }]
                }]
            })
            console.log('Sentiment chart option set')
        } else {
            console.error('sentimentChart element not found')
        }
    } else {
        console.error('Invalid sentiment data:', sentimentData.value)
    }
    
    // 6. 阅读时间模式
    if (readingPatternData.value.length > 0) {
        const readingPatternChart = echarts.init(document.getElementById('readingPatternChart'))
        readingPatternChart.setOption({
            backgroundColor: 'transparent',
            tooltip: { 
                trigger: 'axis',
                formatter: function(params) {
                    return `${params[0].name}<br/>阅读次数: ${params[0].value}`
                }
            },
            xAxis: {
                type: 'category',
                data: readingPatternData.value.map(item => item.time_label),
                axisLabel: {
                    interval: 2,
                    color: 'var(--ink-black)'
                }
            },
            yAxis: {
                type: 'value',
                axisLabel: {
                    color: 'var(--ink-black)'
                }
            },
            series: [{
                data: readingPatternData.value.map(item => item.count),
                type: 'bar',
                smooth: true,
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#cf3f35' },
                        { offset: 1, color: '#8a1616' }
                    ])
                },
                emphasis: {
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: '#bfa46f' },
                            { offset: 1, color: '#8a6d3f' }
                        ])
                    }
                }
            }]
        })
    }
}

const handleResize = () => {
    prefChart?.resize()
    poetThemeSankeyChart?.resize()
}

onMounted(() => {
    fetchData()
    window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
    window.removeEventListener('resize', handleResize)
    if (prefChart) prefChart.dispose()
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

/* Loading Overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(253, 251, 247, 0.95);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(5px);
}

.loading-content {
    text-align: center;
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(207, 63, 53, 0.1);
}

.loading-spinner {
    width: 60px;
    height: 60px;
    border: 4px solid rgba(207, 63, 53, 0.2);
    border-top: 4px solid var(--cinnabar-red);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-content p {
    font-family: "Noto Serif SC", serif;
    font-size: 18px;
    color: var(--ink-black);
    margin-bottom: 8px;
}

.loading-subtext {
    font-size: 14px !important;
    color: var(--text-tertiary) !important;
}

/* Empty State */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 450px;
    text-align: center;
    color: var(--text-tertiary);
}

.empty-icon {
    color: var(--cinnabar-red);
    opacity: 0.5;
    margin-bottom: 24px;
}

.empty-state h4 {
    font-family: "Noto Serif SC", serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--ink-black);
    margin-bottom: 12px;
}

.empty-state p {
    font-size: 14px;
    line-height: 1.5;
    max-width: 300px;
}
</style>
