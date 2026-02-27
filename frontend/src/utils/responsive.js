/**
 * 响应式适配脚本
 * - 移动端 (<= 768px): 动态缩放，以 375px 为基准
 * - 平板端 (768px - 1200px): 固定 14px
 * - 桌面端 (> 1200px): 固定 16px
 */

function setRemUnit() {
    const docEl = document.documentElement;
    const clientWidth = docEl.clientWidth;

    if (clientWidth <= 768) {
        // 移动端：以 375px 为基准，10rem = 375px
        docEl.style.fontSize = (clientWidth / 37.5) + 'px';
    } else if (clientWidth <= 1200) {
        // 平板端：固定 14px
        docEl.style.fontSize = '14px';
    } else {
        // 桌面端：固定 16px，不随屏幕缩放
        docEl.style.fontSize = '16px';
    }
}

// 初始化
setRemUnit();

// 监听窗口变化
window.addEventListener('resize', setRemUnit);
window.addEventListener('pageshow', (e) => {
    if (e.persisted) {
        setRemUnit();
    }
});

export default setRemUnit;
