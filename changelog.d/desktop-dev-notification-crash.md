- **The desktop dev build no longer aborts on its second background
  notification.** In `tauri dev` the first native notification correctly declined
  and fell back to the notification plugin, but that fallback rewrites the
  process's bundle identifier, so the next native notification believed it was
  running from an app bundle and macOS terminated the app with
  `bundleProxyForCurrentProcess is nil`. The app-bundle check now reads the
  executable path, which nothing can rewrite.
