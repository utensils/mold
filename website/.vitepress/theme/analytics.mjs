// Public documentation only. The apps and embedded Studio never import this.
export const measurementId = 'G-RG6PPTGX2T'

function cleanUrl(value) {
  try {
    const url = new URL(value)
    return url.origin + url.pathname
  } catch {
    return ''
  }
}

export function createAnalytics(win) {
  let loaded = false
  let previousPage = cleanUrl(win.document.referrer)
  const eligible = () =>
    win.location.origin === 'https://utensils.io' &&
    win.location.pathname.startsWith('/mold/')
  function gtag() {
    win.dataLayer.push(arguments)
  }
  function pageView() {
    if (!eligible() || !loaded) return
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
    if (!eligible() || loaded) return
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
  return { start, pageView }
}
