// Public documentation only. The apps and embedded Studio never import this.
export const measurementId = 'G-RG6PPTGX2T'
const storageKey = 'mold.website.analytics'

function cleanUrl(value) {
  try {
    const url = new URL(value)
    return url.origin + url.pathname
  } catch {
    return ''
  }
}

export function createAnalytics(win) {
  let selected
  let loaded = false
  let previousPage = cleanUrl(win.document.referrer)
  const eligible = () =>
    win.location.origin === 'https://utensils.io' &&
    win.location.pathname.startsWith('/mold/')
  function choice() {
    if (selected !== undefined) return selected
    try {
      const saved = win.localStorage.getItem(storageKey)
      if (saved === 'granted' || saved === 'denied') selected = saved
    } catch {
      // Storage may be disabled. Consent still works for the current page.
    }
    return selected
  }
  function gtag() {
    win.dataLayer.push(arguments)
  }
  function pageView() {
    if (!eligible() || choice() !== 'granted' || !loaded) return
    const page = cleanUrl(win.location.href)
    gtag('set', { page_location: page, page_referrer: previousPage })
    gtag('event', 'page_view', {
      page_location: page,
      page_title: win.document.title,
      page_referrer: previousPage,
    })
    previousPage = page
  }
  function start() {
    if (!eligible() || choice() !== 'granted' || loaded) return
    loaded = true
    win.dataLayer = win.dataLayer || []
    win.gtag = gtag
    win[`ga-disable-${measurementId}`] = false
    gtag('consent', 'default', {
      analytics_storage: 'granted',
      ad_storage: 'denied',
      ad_user_data: 'denied',
      ad_personalization: 'denied',
    })
    gtag('js', new Date())
    gtag('set', {
      page_location: cleanUrl(win.location.href),
      page_referrer: previousPage,
    })
    gtag('config', measurementId, {
      send_page_view: false,
      allow_google_signals: false,
      allow_ad_personalization_signals: false,
      cookie_path: '/mold/',
      cookie_domain: 'utensils.io',
      cookie_prefix: 'mold',
    })
    const script = win.document.createElement('script')
    script.async = true
    script.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`
    win.document.head.appendChild(script)
    pageView()
  }
  function choose(accepted) {
    selected = accepted ? 'granted' : 'denied'
    try {
      win.localStorage.setItem(storageKey, selected)
    } catch {
      // Keep the choice in memory if persistent storage is unavailable.
    }
    if (loaded) {
      win[`ga-disable-${measurementId}`] = !accepted
      gtag('consent', 'update', { analytics_storage: selected })
    } else if (accepted) {
      start()
    }
  }
  return { choice, choose, start, pageView }
}
