<template>
  <div class="app-container">
    <n-dialog-provider>
      <n-message-provider>
        <router-view />
      </n-message-provider>
    </n-dialog-provider>
  </div>
</template>

<script setup>
import { NMessageProvider, NDialogProvider } from 'naive-ui'
</script>

<style>
#app {
  width: 100%;
  min-height: 100vh;
  margin: 0;
  padding: 0;
  overflow-x: hidden; /* Prevent horizontal scroll but allow vertical */
}

.app-container {
  width: 100%;
  min-height: 100vh;
  position: relative;
}
</style>
