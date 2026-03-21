<template>
  <div class="admin-container">
    <!-- 顶部导航 -->
    <nav class="top-nav glass-card">
      <div class="nav-brand">
        <span class="logo-text">诗云</span>
        <span class="edition-badge">管理后台</span>
      </div>

      <div class="nav-actions">
        <div
          v-for="item in menuItems"
          :key="item.key"
          class="nav-btn-card"
          :class="{ active: activeKey === item.key }"
          @click="activeKey = item.key"
        >
          <n-icon :component="item.icon" />
          <span>{{ item.label }}</span>
        </div>

        <div class="divider-vertical"></div>

        <div class="nav-btn-card" @click="router.push('/')">
          <n-icon :component="HomeOutline" />
          <span>返回前台</span>
        </div>

        <div class="nav-btn-card logout" @click="logout">
          <n-icon :component="LogOutOutline" />
          <span>退出</span>
        </div>
      </div>
    </nav>

    <!-- 主内容区 -->
    <main class="admin-main">
      <!-- 概览页 -->
      <div v-show="activeKey === 'overview'" class="content-section">
        <div class="stats-grid">
          <div class="stat-card glass-card">
            <div class="stat-icon users">
              <n-icon :component="PeopleOutline" />
            </div>
            <div class="stat-info">
              <span class="stat-value">{{ overview.users }}</span>
              <span class="stat-label">注册用户</span>
            </div>
          </div>
          <div class="stat-card glass-card">
            <div class="stat-icon poems">
              <n-icon :component="BookOutline" />
            </div>
            <div class="stat-info">
              <span class="stat-value">{{ overview.poems }}</span>
              <span class="stat-label">诗歌总量</span>
            </div>
          </div>
          <div class="stat-card glass-card">
            <div class="stat-icon reviews">
              <n-icon :component="ChatboxEllipsesOutline" />
            </div>
            <div class="stat-info">
              <span class="stat-value">{{ overview.reviews }}</span>
              <span class="stat-label">评论总量</span>
            </div>
          </div>
          <div class="stat-card glass-card">
            <div class="stat-icon today">
              <n-icon :component="TodayOutline" />
            </div>
            <div class="stat-info">
              <span class="stat-value">{{ overview.today_reviews }}</span>
              <span class="stat-label">今日评论</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 诗歌管理 -->
      <div v-show="activeKey === 'poems'" class="content-section">
        <div class="panel glass-card">
          <div class="panel-head">
            <div class="panel-title">
              <h2>诗歌内容管理</h2>
              <span class="panel-subtitle">维护诗歌资料与内容</span>
            </div>
            <div class="panel-actions">
              <n-input
                v-model:value="poemQuery"
                clearable
                placeholder="搜索标题、作者、标签"
                @keyup.enter="handlePoemSearch"
                class="search-input"
              />
              <n-button @click="handlePoemSearch">搜索</n-button>
              <n-button type="primary" @click="openCreateModal">新增诗歌</n-button>
            </div>
          </div>

          <div class="poem-list">
            <div v-for="poem in poems" :key="poem.id" class="poem-item">
              <div class="poem-meta">
                <div>
                  <h3>{{ poem.title }}</h3>
                  <p>{{ poem.author || '佚名' }} · {{ poem.dynasty || '未标注朝代' }}</p>
                </div>
                <div class="poem-badges">
                  <span class="meta-pill">浏览 {{ poem.views || 0 }}</span>
                  <span class="meta-pill">评论 {{ poem.review_count || 0 }}</span>
                  <span class="meta-pill">评分 {{ Number(poem.average_rating || 0).toFixed(1) }}</span>
                </div>
              </div>

              <p class="poem-content">{{ poem.content }}</p>

              <div class="poem-footer">
                <div class="tag-row">
                  <span v-if="poem.category" class="tag-pill">{{ poem.category }}</span>
                  <span v-if="poem.topic_tags" class="tag-pill">{{ poem.topic_tags }}</span>
                </div>

                <div class="item-actions">
                  <n-button size="small" @click="openEditModal(poem)">编辑</n-button>
                  <n-button size="small" type="error" secondary @click="removePoem(poem)">删除</n-button>
                </div>
              </div>
            </div>

            <n-empty v-if="!poems.length && !poemLoading" description="暂无诗歌数据" />
            <div v-if="poemLoading" class="state-line">诗歌列表加载中...</div>
          </div>

          <div class="pager-wrap">
            <n-pagination
              v-model:page="poemPage"
              :page-count="poemPageCount"
              @update:page="loadPoems"
            />
          </div>
        </div>
      </div>

      <!-- 评论审核 -->
      <div v-show="activeKey === 'reviews'" class="content-section">
        <div class="panel glass-card">
          <div class="panel-head">
            <div class="panel-title">
              <h2>评论审核</h2>
              <span class="panel-subtitle">检查评论质量与内容</span>
            </div>
            <n-button @click="loadReviews">刷新</n-button>
          </div>

          <div class="review-list">
            <div v-for="review in reviews" :key="review.id" class="review-item">
              <div class="review-topline">
                <strong>{{ review.user }}</strong>
                <span>{{ review.poem_title }}</span>
              </div>
              <div class="review-score">评分 {{ review.rating }}</div>
              <p class="review-content">{{ review.comment || '该评论为空' }}</p>
              <div class="review-bottom">
                <span>{{ formatTime(review.created_at) }}</span>
                <n-button size="small" type="error" tertiary @click="removeReview(review)">删除评论</n-button>
              </div>
            </div>

            <n-empty v-if="!reviews.length && !reviewLoading" description="暂无评论数据" />
            <div v-if="reviewLoading" class="state-line">评论列表加载中...</div>
          </div>

          <div class="pager-wrap">
            <n-pagination
              v-model:page="reviewPage"
              :page-count="reviewPageCount"
              @update:page="loadReviews"
            />
          </div>
        </div>
      </div>

      <!-- 用户管理 -->
      <div v-show="activeKey === 'users'" class="content-section">
        <div class="panel glass-card">
          <div class="panel-head">
            <div class="panel-title">
              <h2>用户管理</h2>
              <span class="panel-subtitle">管理用户信息与权限</span>
            </div>
            <div class="panel-actions">
              <n-input
                v-model:value="userQuery"
                clearable
                placeholder="搜索用户名"
                @keyup.enter="handleUserSearch"
                class="search-input"
              />
              <n-button @click="handleUserSearch">搜索</n-button>
            </div>
          </div>

          <div class="user-list">
            <div v-for="user in users" :key="user.id" class="user-item">
              <div class="user-main">
                <div>
                  <h3>{{ user.username }}</h3>
                  <p>注册于 {{ formatTime(user.created_at) }}</p>
                </div>
                <div class="poem-badges">
                  <span class="meta-pill">ID {{ user.id }}</span>
                  <span class="meta-pill">评论 {{ user.review_count || 0 }}</span>
                </div>
              </div>
              <div class="user-actions">
                <n-button size="small" @click="openUserEditor(user)">编辑</n-button>
                <n-button size="small" type="warning" secondary @click="openPasswordReset(user)">重置密码</n-button>
                <n-button size="small" type="error" secondary @click="removeUser(user)">删除</n-button>
              </div>
            </div>

            <n-empty v-if="!users.length && !userLoading" description="暂无用户数据" />
            <div v-if="userLoading" class="state-line">用户列表加载中...</div>
          </div>

          <div class="pager-wrap">
            <n-pagination
              v-model:page="userPage"
              :page-count="userPageCount"
              @update:page="loadUsers"
            />
          </div>
        </div>
      </div>
    </main>
  </div>

  <!-- 编辑/新增诗歌弹窗 -->
  <n-modal v-model:show="showEditor">
    <n-card
      :title="editingId ? '编辑诗歌' : '新增诗歌'"
      class="editor-modal"
      :bordered="false"
    >
      <n-form :model="poemForm" label-placement="top" class="editor-form">
        <div class="editor-grid">
          <n-form-item label="标题">
            <n-input v-model:value="poemForm.title" placeholder="请输入标题" />
          </n-form-item>
          <n-form-item label="作者">
            <n-input v-model:value="poemForm.author" placeholder="请输入作者" />
          </n-form-item>
          <n-form-item label="朝代">
            <n-input v-model:value="poemForm.dynasty" placeholder="请输入朝代" />
          </n-form-item>
          <n-form-item label="类别">
            <n-input v-model:value="poemForm.category" placeholder="请输入类别" />
          </n-form-item>
          <n-form-item label="词牌 / 韵律">
            <n-input v-model:value="poemForm.rhythmic" placeholder="请输入词牌或韵律" />
          </n-form-item>
          <n-form-item label="主题标签">
            <n-input v-model:value="poemForm.topic_tags" placeholder="示例：山水-思乡-离别" />
          </n-form-item>
          <n-form-item label="章节">
            <n-input v-model:value="poemForm.chapter" placeholder="请输入章节" />
          </n-form-item>
          <n-form-item label="分段">
            <n-input v-model:value="poemForm.section" placeholder="请输入分段" />
          </n-form-item>
          <n-form-item label="浏览量">
            <n-input-number v-model:value="poemForm.views" :min="0" />
          </n-form-item>
        </div>

        <n-form-item label="正文">
          <n-input
            v-model:value="poemForm.content"
            type="textarea"
            :autosize="{ minRows: 8, maxRows: 14 }"
            placeholder="请输入诗歌正文"
          />
        </n-form-item>
      </n-form>

      <template #footer>
        <div class="modal-actions">
          <n-button @click="showEditor = false">取消</n-button>
          <n-button type="primary" :loading="saving" @click="savePoem">保存</n-button>
        </div>
      </template>
    </n-card>
  </n-modal>

  <!-- 编辑用户弹窗 -->
  <n-modal v-model:show="showUserEditor">
    <n-card title="编辑用户" class="editor-modal" :bordered="false">
      <n-form :model="userForm" label-placement="top">
        <n-form-item label="用户名">
          <n-input v-model:value="userForm.username" placeholder="请输入用户名" />
        </n-form-item>
      </n-form>

      <template #footer>
        <div class="modal-actions">
          <n-button @click="showUserEditor = false">取消</n-button>
          <n-button type="primary" :loading="savingUser" @click="saveUser">保存</n-button>
        </div>
      </template>
    </n-card>
  </n-modal>

  <!-- 重置密码弹窗 -->
  <n-modal v-model:show="showPasswordEditor">
    <n-card title="重置用户密码" class="editor-modal" :bordered="false">
      <n-form :model="passwordForm" label-placement="top">
        <n-form-item label="新密码">
          <n-input
            v-model:value="passwordForm.password"
            type="password"
            show-password-on="mousedown"
            placeholder="请输入新密码"
          />
        </n-form-item>
      </n-form>

      <template #footer>
        <div class="modal-actions">
          <n-button @click="showPasswordEditor = false">取消</n-button>
          <n-button type="primary" :loading="savingPassword" @click="savePasswordReset">确认重置</n-button>
        </div>
      </template>
    </n-card>
  </n-modal>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useDialog, useMessage, NIcon } from 'naive-ui'
import {
  GridOutline,
  BookOutline,
  ChatboxEllipsesOutline,
  PeopleOutline,
  HomeOutline,
  LogOutOutline,
  TodayOutline
} from '@vicons/ionicons5'
import axios from 'axios'

const router = useRouter()
const dialog = useDialog()
const message = useMessage()

const activeKey = ref('overview')

const menuItems = [
  { key: 'overview', label: '概览', icon: GridOutline },
  { key: 'poems', label: '诗歌管理', icon: BookOutline },
  { key: 'reviews', label: '评论审核', icon: ChatboxEllipsesOutline },
  { key: 'users', label: '用户管理', icon: PeopleOutline }
]

const overview = reactive({
  users: 0,
  poems: 0,
  reviews: 0,
  today_reviews: 0,
})

const poemQuery = ref('')
const poemPage = ref(1)
const poemPageSize = 8
const poemTotal = ref(0)
const poems = ref([])
const poemLoading = ref(false)

const reviewPage = ref(1)
const reviewPageSize = 8
const reviewTotal = ref(0)
const reviews = ref([])
const reviewLoading = ref(false)

const userQuery = ref('')
const userPage = ref(1)
const userPageSize = 8
const userTotal = ref(0)
const users = ref([])
const userLoading = ref(false)

const showEditor = ref(false)
const saving = ref(false)
const editingId = ref(null)
const poemForm = reactive({
  title: '',
  author: '',
  dynasty: '',
  category: '',
  rhythmic: '',
  topic_tags: '',
  chapter: '',
  section: '',
  views: 0,
  content: '',
})

const showUserEditor = ref(false)
const savingUser = ref(false)
const editingUserId = ref(null)
const userForm = reactive({
  username: '',
})

const showPasswordEditor = ref(false)
const savingPassword = ref(false)
const passwordUserId = ref(null)
const passwordForm = reactive({
  username: '',
  password: '',
})

const poemPageCount = computed(() => Math.max(1, Math.ceil(poemTotal.value / poemPageSize)))
const reviewPageCount = computed(() => Math.max(1, Math.ceil(reviewTotal.value / reviewPageSize)))
const userPageCount = computed(() => Math.max(1, Math.ceil(userTotal.value / userPageSize)))

const adminHeaders = () => ({
  Authorization: `Bearer ${localStorage.getItem('admin_token') || ''}`,
})

const resetPoemForm = () => {
  editingId.value = null
  Object.assign(poemForm, {
    title: '',
    author: '',
    dynasty: '',
    category: '',
    rhythmic: '',
    topic_tags: '',
    chapter: '',
    section: '',
    views: 0,
    content: '',
  })
}

const handleAdminError = (error, fallback) => {
  if (error.response?.status === 401) {
    localStorage.removeItem('admin_token')
    localStorage.removeItem('admin_name')
    message.error('管理员登录已失效，请重新登录')
    router.push('/admin/login')
    return
  }
  message.error(error.response?.data?.message || fallback)
}

const loadOverview = async () => {
  try {
    const { data } = await axios.get('/api/admin/overview', { headers: adminHeaders() })
    Object.assign(overview, data.overview)
  } catch (error) {
    handleAdminError(error, '概览数据加载失败')
  }
}

const loadPoems = async () => {
  poemLoading.value = true
  try {
    const { data } = await axios.get('/api/admin/poems', {
      headers: adminHeaders(),
      params: {
        q: poemQuery.value,
        page: poemPage.value,
        page_size: poemPageSize,
      },
    })
    poems.value = data.items
    poemTotal.value = data.pagination.total
  } catch (error) {
    handleAdminError(error, '诗歌列表加载失败')
  } finally {
    poemLoading.value = false
  }
}

const loadReviews = async () => {
  reviewLoading.value = true
  try {
    const { data } = await axios.get('/api/admin/reviews', {
      headers: adminHeaders(),
      params: {
        page: reviewPage.value,
        page_size: reviewPageSize,
      },
    })
    reviews.value = data.items
    reviewTotal.value = data.pagination.total
  } catch (error) {
    handleAdminError(error, '评论列表加载失败')
  } finally {
    reviewLoading.value = false
  }
}

const loadUsers = async () => {
  userLoading.value = true
  try {
    const { data } = await axios.get('/api/admin/users', {
      headers: adminHeaders(),
      params: {
        q: userQuery.value,
        page: userPage.value,
        page_size: userPageSize,
      },
    })
    users.value = data.items
    userTotal.value = data.pagination.total
  } catch (error) {
    handleAdminError(error, '用户列表加载失败')
  } finally {
    userLoading.value = false
  }
}

const handlePoemSearch = () => {
  poemPage.value = 1
  loadPoems()
}

const handleUserSearch = () => {
  userPage.value = 1
  loadUsers()
}

const openCreateModal = () => {
  resetPoemForm()
  showEditor.value = true
}

const openEditModal = (poem) => {
  editingId.value = poem.id
  Object.assign(poemForm, {
    title: poem.title || '',
    author: poem.author || '',
    dynasty: poem.dynasty || '',
    category: poem.category || '',
    rhythmic: poem.rhythmic || '',
    topic_tags: poem.topic_tags || '',
    chapter: poem.chapter || '',
    section: poem.section || '',
    views: poem.views || 0,
    content: poem.content || '',
  })
  showEditor.value = true
}

const savePoem = async () => {
  if (!poemForm.title.trim() || !poemForm.content.trim()) {
    message.warning('标题和正文不能为空')
    return
  }

  saving.value = true
  try {
    if (editingId.value) {
      await axios.put(`/api/admin/poems/${editingId.value}`, poemForm, { headers: adminHeaders() })
      message.success('诗歌已更新')
    } else {
      await axios.post('/api/admin/poems', poemForm, { headers: adminHeaders() })
      message.success('诗歌已创建')
    }
    showEditor.value = false
    await Promise.all([loadOverview(), loadPoems()])
  } catch (error) {
    handleAdminError(error, '保存失败')
  } finally {
    saving.value = false
  }
}

const removePoem = (poem) => {
  dialog.warning({
    title: '删除诗歌',
    content: `确定删除《${poem.title}》吗？相关评论也会一并移除。`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        await axios.delete(`/api/admin/poems/${poem.id}`, { headers: adminHeaders() })
        message.success('诗歌已删除')
        await Promise.all([loadOverview(), loadPoems(), loadReviews()])
      } catch (error) {
        handleAdminError(error, '删除诗歌失败')
      }
    },
  })
}

const removeReview = (review) => {
  dialog.warning({
    title: '删除评论',
    content: `确定删除来自 ${review.user} 的这条评论吗？`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        await axios.delete(`/api/admin/reviews/${review.id}`, { headers: adminHeaders() })
        message.success('评论已删除')
        await Promise.all([loadOverview(), loadPoems(), loadReviews()])
      } catch (error) {
        handleAdminError(error, '删除评论失败')
      }
    },
  })
}

const openUserEditor = (user) => {
  editingUserId.value = user.id
  userForm.username = user.username || ''
  showUserEditor.value = true
}

const saveUser = async () => {
  if (!userForm.username.trim()) {
    message.warning('用户名不能为空')
    return
  }

  savingUser.value = true
  try {
    await axios.put(
      `/api/admin/users/${editingUserId.value}`,
      { username: userForm.username },
      { headers: adminHeaders() }
    )
    message.success('用户信息已更新')
    showUserEditor.value = false
    await Promise.all([loadOverview(), loadUsers(), loadReviews()])
  } catch (error) {
    handleAdminError(error, '更新用户失败')
  } finally {
    savingUser.value = false
  }
}

const openPasswordReset = (user) => {
  passwordUserId.value = user.id
  passwordForm.username = user.username || ''
  passwordForm.password = ''
  showPasswordEditor.value = true
}

const savePasswordReset = async () => {
  if (!passwordForm.password.trim()) {
    message.warning('新密码不能为空')
    return
  }

  savingPassword.value = true
  try {
    await axios.put(
      `/api/admin/users/${passwordUserId.value}`,
      { username: passwordForm.username, reset_password: passwordForm.password },
      { headers: adminHeaders() }
    )
    message.success('密码已重置')
    showPasswordEditor.value = false
    await loadUsers()
  } catch (error) {
    handleAdminError(error, '重置密码失败')
  } finally {
    savingPassword.value = false
  }
}

const removeUser = (user) => {
  dialog.warning({
    title: '删除用户',
    content: `确定删除用户 ${user.username} 吗？该用户的评论记录也会一并删除。`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        await axios.delete(`/api/admin/users/${user.id}`, { headers: adminHeaders() })
        message.success('用户已删除')
        await Promise.all([loadOverview(), loadUsers(), loadReviews(), loadPoems()])
      } catch (error) {
        handleAdminError(error, '删除用户失败')
      }
    },
  })
}

const formatTime = (value) => {
  if (!value) return '未知时间'
  return new Date(value).toLocaleString()
}

const logout = () => {
  localStorage.removeItem('admin_token')
  localStorage.removeItem('admin_name')
  router.push('/admin/login')
}

onMounted(async () => {
  await Promise.all([loadOverview(), loadPoems(), loadReviews(), loadUsers()])
})
</script>

<style scoped>
.admin-container {
  min-height: 100vh;
  background: var(--gradient-bg);
  padding: 24px;
}

.admin-main {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding-top: 24px;
}

.content-section {
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 导航栏样式复用主应用 */
.top-nav {
  position: sticky;
  top: 0;
  z-index: 1000;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 var(--content-padding);
  height: var(--header-height);
  margin: 0 auto;
  max-width: var(--content-max-width);
  box-sizing: border-box;
}

.nav-brand {
  display: flex;
  align-items: baseline;
  gap: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.nav-brand:hover {
  color: var(--cinnabar-red);
}

.logo-text {
  font-family: "Noto Serif SC", serif;
  font-size: 26px;
  font-weight: 600;
  letter-spacing: 0.35em;
  color: var(--ink-black);
  transition: color 0.2s ease;
}

.nav-brand:hover .logo-text {
  color: var(--cinnabar-red);
}

.edition-badge {
  font-size: 10px;
  font-weight: 300;
  letter-spacing: 0.2em;
  color: var(--text-tertiary);
  text-transform: uppercase;
  margin-left: 4px;
  opacity: 0.8;
}

.nav-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.nav-btn-card {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 18px;
  height: 48px;
  border-radius: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
  color: var(--text-secondary);
  background: rgba(0, 0, 0, 0.02);
  font-size: 14px;
  font-weight: 500;
  letter-spacing: 0.05em;
  box-sizing: border-box;
}

.nav-btn-card:hover {
  background: rgba(0, 0, 0, 0.06);
  color: var(--ink-black);
  transform: translateY(-1px);
}

.nav-btn-card .n-icon {
  font-size: 18px;
  transition: all 0.2s ease;
}

.nav-btn-card:hover .n-icon {
  color: var(--cinnabar-red);
}

.nav-btn-card.active {
  background: var(--cinnabar-red) !important;
  color: white !important;
  box-shadow: 0 4px 12px rgba(166, 27, 27, 0.2);
}

.nav-btn-card.active .n-icon {
  color: white !important;
}

.nav-btn-card.logout:hover {
  background: rgba(207, 63, 53, 0.1);
  color: var(--cinnabar-red);
}

.divider-vertical {
  width: 1px;
  height: 24px;
  background: rgba(0, 0, 0, 0.08);
  margin: 0 8px;
}

/* 统计卡片 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.stat-card {
  padding: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: transform 0.2s, box-shadow 0.2s;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-hover);
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  font-size: 24px;
  color: white;
}

.stat-icon.users {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.stat-icon.poems {
  background: linear-gradient(135deg, #f093fb, #f5576c);
}

.stat-icon.reviews {
  background: linear-gradient(135deg, #4facfe, #00f2fe);
}

.stat-icon.today {
  background: linear-gradient(135deg, #43e97b, #38f9d7);
}

.stat-info {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--ink-black);
  line-height: 1.2;
}

.stat-label {
  font-size: 14px;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* 面板 */
.panel {
  padding: 28px;
  border-radius: var(--radius-main);
}

.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  flex-wrap: wrap;
  gap: 16px;
}

.panel-title {
  display: flex;
  flex-direction: column;
}

.panel-title h2 {
  font-size: 22px;
  font-weight: 600;
  color: var(--ink-black);
  margin: 0;
}

.panel-subtitle {
  font-size: 13px;
  color: var(--text-secondary);
  margin-top: 4px;
}

.panel-actions {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.search-input {
  width: 240px;
}

.poem-list,
.review-list,
.user-list {
  display: grid;
  gap: 14px;
}

.poem-item,
.review-item,
.user-item {
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 16px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.6);
  transition: all 0.2s ease;
}

.poem-item:hover,
.review-item:hover,
.user-item:hover {
  background: rgba(255, 255, 255, 0.9);
  border-color: rgba(0, 0, 0, 0.1);
  box-shadow: var(--shadow-sm);
}

.poem-meta,
.poem-footer,
.review-bottom,
.review-topline,
.user-main,
.user-actions {
  display: flex;
  justify-content: space-between;
  gap: 14px;
}

.user-main {
  align-items: center;
}

.user-main h3,
.poem-meta h3 {
  font-size: 17px;
  margin-bottom: 4px;
  color: var(--ink-black);
  font-weight: 600;
}

.user-main p,
.poem-meta p,
.review-score,
.review-bottom {
  color: var(--text-secondary);
  font-size: 13px;
}

.user-actions {
  margin-top: 14px;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: 8px;
}

.poem-badges,
.tag-row,
.item-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.meta-pill,
.tag-pill {
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(207, 63, 53, 0.08);
  color: var(--cinnabar-red);
}

.tag-pill {
  background: rgba(102, 126, 234, 0.08);
  color: #667eea;
}

.poem-content {
  margin: 12px 0;
  line-height: 1.7;
  color: var(--text-secondary);
  white-space: pre-wrap;
  font-size: 14px;
}

.review-topline {
  align-items: center;
}

.review-topline strong {
  color: var(--ink-black);
  font-weight: 600;
}

.review-score {
  margin-top: 8px;
  font-weight: 500;
  color: var(--cinnabar-red);
}

.state-line {
  padding: 24px 8px;
  text-align: center;
  color: var(--text-secondary);
}

.pager-wrap {
  display: flex;
  justify-content: flex-end;
  margin-top: 18px;
}

.editor-modal {
  width: min(960px, calc(100vw - 48px));
  border-radius: 20px !important;
}

.editor-form {
  margin-top: 10px;
}

.editor-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0 14px;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

@media (max-width: 1200px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }

  .editor-grid {
    grid-template-columns: 1fr;
  }

  .panel-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .search-input {
    width: 100%;
  }

  .nav-actions {
    flex-wrap: wrap;
  }
}</style>
