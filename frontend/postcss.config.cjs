module.exports = {
    plugins: {
        'postcss-pxtorem': {
            rootValue: 16, // 基于固定的 16px 基准
            propList: ['*'],
            selectorBlackList: ['norem', 'html'], // 过滤掉 .norem- 和 html 选择器
            minPixelValue: 2, // 最小转换像素，避免 1px 边框被转换
        },
    },
};
