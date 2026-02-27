<template>
  <div class="search-container">
    <nav class="top-nav glass-card anim-enter">
      <div class="nav-brand" @click="goHome">
        <span class="logo-text">诗云</span>
        <span class="edition-badge">Zen Edition</span>
      </div>
      
      <div class="nav-actions">
        <!-- 搜索 (Active) -->
        <div class="nav-btn-card active" title="Search">
            <n-icon><NSearch /></n-icon>
            <span>搜索</span>
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

    <main class="search-main anim-enter">
        <div class="search-hero">
            <h1 class="zen-title">寻觅诗心</h1>
            <p class="zen-subtitle">在诗意的云海中，检索那份跨越千年的共鸣</p>
            
            <div class="search-input-wrapper glass-card">
                <n-input 
                    v-model:value="searchQuery" 
                    placeholder="输入标题、作者、千古名句或意象关键词..." 
                    size="large" 
                    @keyup.enter="handleSearch" 
                    @input="handleInputSearch"
                    class="search-bar-zen"
                    clearable
                    autofocus
                >
                    <template #prefix><n-icon><NSearch /></n-icon></template>
                    <template #suffix>
                        <n-button text @click="handleSearch" :disabled="!searchQuery.trim()">
                            <n-icon><NSearch /></n-icon>
                        </n-button>
                    </template>
                </n-input>
            </div>
        </div>

        <!-- Search Content -->
        <div class="search-content">
            <!-- Loading -->
            <div v-if="searchLoading" class="search-status">
                <n-spin size="large" />
                <span class="status-text">研墨寻索中...</span>
            </div>

            <!-- Results -->
            <div v-else-if="searchResults.length" class="results-layout">
                <div class="results-header">
                    <span class="result-count">寻得 {{ searchResults.length }} 首缘分诗篇</span>
                </div>
                <div class="results-grid">
                    <div v-for="poem in searchResults" :key="poem.id" class="poem-card-minimal glass-card" @click="goToPoem(poem.id)">
                        <div class="p-header">
                            <h3 class="p-title">{{ poem.title }}</h3>
                            <n-tag :bordered="false" type="error" size="small" class="p-dynasty">{{ poem.dynasty }}</n-tag>
                        </div>
                        <span class="p-author">{{ poem.author }}</span>
                        <p class="p-excerpt">{{ poem.content.substring(0, 50) }}...</p>
                        <div class="p-footer">
                            <span class="p-reason"><n-icon><NSparkles /></n-icon> {{ poem.recommend_reason }}</span>
                            <n-icon class="p-arrow"><NArrowRight /></n-icon>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Empty / Initial State -->
            <div v-else-if="!searchQuery && !searchLoading" class="search-placeholder">
                <div class="suggestion-section">
                    <h3>时下热搜</h3>
                    <div class="tag-cloud">
                        <n-tag v-for="tag in hotTags" :key="tag" clickable round @click="quickSearch(tag)">{{ tag }}</n-tag>
                    </div>
                </div>
                <div class="suggestion-section">
                    <h3>诗意分类</h3>
                    <div class="category-grid">
                        <div v-for="cat in categories" :key="cat.name" class="category-item glass-card" @click="quickSearch(cat.name)">
                            <n-icon size="24" :component="cat.icon" />
                            <span>{{ cat.name }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div v-else-if="searchQuery && !searchLoading" class="search-empty">
                <n-empty description="此间意象，尚待诗人落笔。换个词试试？" />
            </div>
        </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import axios from 'axios'
import { 
  NInput, 
  NButton, 
  NIcon, 
  NEmpty, 
  NSpin,
  NTag
} from 'naive-ui'
import { 
  Search as NSearch, 
  ChevronForward as NArrowRight, 
  PersonOutline as NPersonOutline,
  GlobeOutline as NGlobeOutline,
  Sparkles as NSparkles,
  LeafOutline as NLeaf,
  MoonOutline as NMoon,
  WineOutline as NWine,
  BoatOutline as NBoat,
  CloudOutline as NCloud,
  FlowerOutline as NFlower
} from '@vicons/ionicons5'

const router = useRouter()
const route = useRoute()
const currentUser = localStorage.getItem('user') || '访客'

const searchQuery = ref('')
const searchLoading = ref(false)
const searchResults = ref([])

const hotTags = ['李白', '苏轼', '明月', '春风', '江南', '相思', '杜甫', '山水']
const categories = [
  { name: '山水', icon: NLeaf },
  { name: '月色', icon: NMoon },
  { name: '美酒', icon: NWine },
  { name: '归舟', icon: NBoat },
  { name: '云烟', icon: NCloud },
  { name: '繁花', icon: NFlower }
]

const goHome = () => router.push('/')
const goToPersonalAnalysis = () => router.push('/personal-analysis')
const goToGlobalAnalysis = () => router.push('/global-analysis')
const goToPoem = (id) => router.push(`/?poemId=${id}`)
const logout = () => {
  localStorage.removeItem('user')
  router.push('/login')
}

// 防抖函数实现
const debounce = (fn, delay) => {
  let timer = null
  return function() {
    clearTimeout(timer)
    timer = setTimeout(() => fn.apply(this, arguments), delay)
  }
}

const handleSearch = async () => {
  if (!searchQuery.value.trim()) {
    searchResults.value = []
    return
  }
  
  searchLoading.value = true
  try {
    const res = await axios.get(`/api/search_poems?q=${encodeURIComponent(searchQuery.value)}`)
    searchResults.value = res.data
  } catch (e) {
    searchResults.value = []
  } finally {
    searchLoading.value = false
  }
}

// 带防抖的实时搜索处理
const handleInputSearch = debounce(() => {
  if (searchQuery.value.trim()) {
    handleSearch()
  } else {
    searchResults.value = []
  }
}, 300)

const quickSearch = (tag) => {
  searchQuery.value = tag
  handleSearch()
}

onMounted(() => {
    // 检查是否有来自 URL 的搜索参数
    if (route.query.q) {
        searchQuery.value = route.query.q
        handleSearch()
    }
})
</script>

<style scoped>
.search-container {
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  background: var(--gradient-bg);
  color: var(--ink-black);
}

.search-main {
    flex: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 60px 40px;
    width: 100%;
}

.search-hero {
    text-align: center;
    margin-bottom: 80px;
}

.zen-title {
    font-family: "Noto Serif SC", serif;
    font-size: 48px;
    font-weight: 700;
    letter-spacing: 0.25em;
    margin-bottom: 16px;
    color: var(--ink-black);
}

.zen-subtitle {
    font-size: 16px;
    color: var(--text-tertiary);
    letter-spacing: 0.15em;
    margin-bottom: 48px;
}

.search-input-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 12px;
    border-radius: 40px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}

.search-bar-zen :deep(.n-input) {
    background: transparent;
    font-size: 18px;
    border-radius: 30px;
    padding: 16px 24px;
    border: none !important;
    outline: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-family: "Noto Serif SC", serif;
    color: var(--ink-black);
    box-shadow: none !important;
}

.search-bar-zen :deep(.n-input:focus-within) {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: rgba(255, 255, 255, 0.95);
}

.search-bar-zen :deep(.n-input)::placeholder {
    color: var(--text-tertiary);
    font-style: italic;
    font-family: "Noto Serif SC", serif;
    letter-spacing: 0.05em;
}

.search-bar-zen :deep(.n-input:disabled) {
    opacity: 0.6;
    cursor: not-allowed;
    background: rgba(0, 0, 0, 0.02);
}

.search-bar-zen :deep(.n-input-prefix) {
    color: var(--text-tertiary);
    font-size: 20px;
    transition: color 0.3s ease;
}

.search-bar-zen :deep(.n-input:focus-within .n-input-prefix) {
    color: var(--cinnabar-red);
}

.search-bar-zen :deep(.n-button) {
    color: var(--text-tertiary);
    transition: all 0.3s ease;
}

.search-bar-zen :deep(.n-button:hover:not(:disabled)) {
    color: var(--cinnabar-red);
    transform: scale(1.1);
}

.search-bar-zen :deep(.n-button:disabled) {
    color: rgba(0, 0, 0, 0.2);
}

.search-input-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 12px;
    border-radius: 40px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(207, 63, 53, 0.02));
    border: 1px solid rgba(207, 63, 53, 0.05);
}

.search-content {
    min-height: 400px;
}

.search-status {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    padding-top: 60px;
}

.status-text {
    color: var(--text-tertiary);
    letter-spacing: 0.2em;
}

/* Results */
.results-header {
    margin-bottom: 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(0,0,0,0.05);
    padding-bottom: 16px;
}

.result-count {
    font-size: 14px;
    color: var(--text-tertiary);
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 24px;
}

.poem-card-minimal {
    padding: 24px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
    height: 100%;
    border: 1px solid rgba(166, 27, 27, 0.05);
}

.poem-card-minimal:hover {
    transform: translateY(-8px);
    border-color: rgba(166, 27, 27, 0.2);
    box-shadow: 0 15px 40px rgba(0,0,0,0.08);
}

.p-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}

.p-title {
    font-family: "Noto Serif SC", serif;
    font-size: 20px;
    font-weight: 600;
    margin: 0;
    color: var(--ink-black);
}

.p-author {
    font-size: 14px;
    color: var(--cinnabar-red);
    margin-bottom: 16px;
}

.p-excerpt {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.8;
    margin-bottom: 20px;
    flex: 1;
}

.p-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(0,0,0,0.03);
    padding-top: 16px;
}

.p-reason {
    font-size: 11px;
    color: var(--text-tertiary);
    display: flex;
    align-items: center;
    gap: 6px;
}

.p-arrow {
    color: var(--cinnabar-red);
    opacity: 0;
    transform: translateX(-10px);
    transition: all 0.3s ease;
}

.poem-card-minimal:hover .p-arrow {
    opacity: 1;
    transform: translateX(0);
}

/* Suggestions */
.search-placeholder {
    display: flex;
    flex-direction: column;
    gap: 60px;
}

.suggestion-section h3 {
    font-family: "Noto Serif SC", serif;
    font-size: 18px;
    margin-bottom: 24px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 12px;
}

.suggestion-section h3::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,0,0,0.05), transparent);
}

.tag-cloud {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

.category-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 20px;
}

.category-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 24px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.category-item:hover {
    background: rgba(166, 27, 27, 0.05);
    color: var(--cinnabar-red);
    transform: scale(1.05);
}

.search-empty {
    padding: 100px 0;
}
</style>
