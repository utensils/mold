import DefaultTheme from 'vitepress/theme'
import { h, onUnmounted } from 'vue'
import { useRouter } from 'vitepress'
import { leaveDocsForSecurePage } from './secure-pages.mjs'
import SupportStrip from './support-strip.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  setup() {
    const router = useRouter()
    const previousBeforeRouteChange = router.onBeforeRouteChange
    const beforeRouteChange = async (to) => {
      if (
        typeof window !== 'undefined' &&
        leaveDocsForSecurePage(window, to) === false
      )
        return false
      return previousBeforeRouteChange?.(to)
    }
    router.onBeforeRouteChange = beforeRouteChange
    onUnmounted(() => {
      if (router.onBeforeRouteChange === beforeRouteChange)
        router.onBeforeRouteChange = previousBeforeRouteChange
    })
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-hero-info-after': () => h(SupportStrip),
    })
  },
}
