import { createRouter, createWebHistory } from 'vue-router'
import Login from './views/Login.vue'
import Register from './views/Register.vue'
import Home from './views/Home.vue'
import Analysis from './views/Analysis.vue'
import PersonalAnalysis from './views/PersonalAnalysis.vue'
import GlobalAnalysis from './views/GlobalAnalysis.vue'
import Search from './views/Search.vue'
import Profile from './views/Profile.vue'
import AdminLogin from './views/AdminLogin.vue'
import AdminDashboard from './views/AdminDashboard.vue'

const routes = [
    { path: '/', component: Home },
    { path: '/login', component: Login },
    { path: '/register', component: Register },
    { path: '/analysis', component: Analysis },
    { path: '/personal-analysis', component: PersonalAnalysis },
    { path: '/global-analysis', component: GlobalAnalysis },
    { path: '/search', component: Search },
    { path: '/profile', component: Profile },
    { path: '/admin/login', component: AdminLogin },
    { path: '/admin', component: AdminDashboard }
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

router.beforeEach((to, from, next) => {
    const user = localStorage.getItem('user')
    const adminToken = localStorage.getItem('admin_token')
    const publicPages = ['/login', '/register', '/admin/login']

    if (to.path === '/admin') {
        if (!adminToken) {
            next('/admin/login')
            return
        }
        next()
        return
    }

    if (to.path === '/admin/login' && adminToken) {
        next('/admin')
        return
    }

    const authRequired = !publicPages.includes(to.path)
    if (authRequired && !user) {
        next('/login')
        return
    }

    next()
})

export default router
