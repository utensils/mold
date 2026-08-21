# Mold mobile native plugin

Android-only Tauri plugin for platform responsibilities that the shared Vue
mobile app cannot own. It provides:

- Keystore-backed, AES-GCM encrypted per-host API-key storage
- `_mold._tcp` discovery through Android NSD
- PNG/JPEG clipboard sharing through a cache-scoped `FileProvider`
- Image and streamed video saving through MediaStore
- Authenticated GIF/APNG/WebP export through the Android share sheet
- System appearance and status/navigation-bar glyph synchronization

The plugin never stores durable API keys in URLs, exported files, logs, or
plain preferences. Its `FileProvider` exposes only the plugin's `cache/shared`
directory, and gallery videos stream directly to MediaStore instead of being
buffered in memory. Product state, routing, HTTP/SSE behavior, and UI remain in
the shared mobile frontend.
