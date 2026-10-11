/** Security-sensitive documents have their own script/CSP trust boundary. */
export function isSecureDocument(pathname) {
  try {
    return /^\/mold\/(?:pair(?:\.html)?|studio)(?:\/|$)/.test(
      decodeURIComponent(pathname)
    )
  } catch {
    return true
  }
}

/** A full document load discards any third-party scripts loaded by the docs. */
export function leaveDocsForSecurePage(win, destination) {
  const url = new URL(destination, win.location.href)
  if (url.origin === win.location.origin && isSecureDocument(url.pathname)) {
    win.location.assign(url.href)
    return false
  }
}
