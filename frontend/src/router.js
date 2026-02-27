import { createRouter, createWebHistory } from 'vue-router'
import Login from './views/Login.vue'
import Register from './views/Register.vue'
import Home from './views/Home.vue'
import Analysis from './views/Analysis.vue'
import PersonalAnalysis from './views/PersonalAnalysis.vue'
import GlobalAnalysis from './views/GlobalAnalysis.vue'
import Search from './views/Search.vue'
import Profile from './views/Profile.vue'

const routes = [
    { path: '/', component: Home },
    { path: '/login', component: Login },
    { path: '/register', component: Register },
    { path: '/analysis', component: Analysis },
    { path: '/personal-analysis', component: PersonalAnalysis },
    { path: '/global-analysis', component: GlobalAnalysis },
    { path: '/search', component: Search },
    { path: '/profile', component: Profile }
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

router.beforeEach((to, from, next) => {
    const user = localStorage.getItem('user');
    const publicPages = ['/login', '/register'];
    const authRequired = !publicPages.includes(to.path);

    if (authRequired && !user) {
        next('/login');
    } else {
        next();
    }
})

export default router
