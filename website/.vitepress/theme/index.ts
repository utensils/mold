import DefaultTheme from 'vitepress/theme'
import { h, nextTick, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vitepress'
import { createAnalytics } from './analytics.mjs'
import SupportStrip from './support-strip.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  setup() {
    const router = useRouter()
    let previousAfterRouteChange
    let afterRouteChange
    onMounted(() => {
      const analytics = createAnalytics(window)
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
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-hero-info-after': () => h(SupportStrip),
    })
  },
}
