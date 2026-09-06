<script setup>
import { nextTick, onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vitepress'
import { createAnalytics } from './analytics.mjs'

const router = useRouter()
const visible = ref(false)
let analytics
let previousAfterRouteChange
let afterRouteChange

onMounted(() => {
  analytics = createAnalytics(window)
  visible.value = !analytics.choice()
  analytics.start()
  previousAfterRouteChange = router.onAfterRouteChange
  afterRouteChange = async (to) => {
    await previousAfterRouteChange?.(to)
    await nextTick()
    analytics.pageView()
  }
  router.onAfterRouteChange = afterRouteChange
})
onUnmounted(() => {
  if (router.onAfterRouteChange === afterRouteChange) {
    router.onAfterRouteChange = previousAfterRouteChange
  }
})
function choose(accepted) {
  const wasEnabled = analytics.choice() === 'granted'
  analytics.choose(accepted)
  visible.value = false
  // Stop already-loaded automatic measurement listeners after withdrawal.
  if (wasEnabled && !accepted) window.location.reload()
}
</script>

<template>
  <div
    v-if="visible"
    class="analytics-choice"
    role="region"
    aria-label="Website analytics"
  >
    <p>
      Allow Google Analytics to measure visits and use of this website?
      <a href="/mold/privacy#website-analytics">Privacy details</a>
    </p>
    <div class="analytics-actions">
      <button type="button" @click="choose(false)">Decline</button>
      <button type="button" @click="choose(true)">Allow analytics</button>
    </div>
  </div>
  <button
    v-else
    class="analytics-settings"
    type="button"
    @click="visible = true"
  >
    Analytics preferences
  </button>
</template>

<style scoped>
.analytics-choice {
  position: fixed;
  z-index: 50;
  bottom: 16px;
  left: 16px;
  right: 16px;
  max-width: 480px;
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  box-shadow: var(--vp-shadow-3);
  font-size: 14px;
  line-height: 1.6;
}
.analytics-choice a {
  color: var(--vp-c-brand-1);
  text-decoration: underline;
}
.analytics-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
}
.analytics-actions button {
  padding: 6px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  font-weight: 500;
}
.analytics-actions button:hover {
  background: var(--vp-c-bg-soft);
}
.analytics-settings {
  display: block;
  margin: 12px auto;
  color: var(--vp-c-text-2);
  font-size: 12px;
  text-decoration: underline;
}
</style>
