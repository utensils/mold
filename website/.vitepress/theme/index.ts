import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import SupportStrip from './support-strip.vue'
import AnalyticsChoice from './analytics-choice.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-hero-info-after': () => h(SupportStrip),
      'layout-bottom': () => h(AnalyticsChoice),
    })
  },
}
