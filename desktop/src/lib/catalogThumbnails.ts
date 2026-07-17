const CATALOG_THUMBNAIL_WIDTH = 512;
const CIVITAI_IMAGE_HOSTS = new Set(["image.civitai.com", "imagecache.civitai.com"]);

/**
 * Keep Grid and Table on one small, stable Civitai CDN URL. The shared URL is
 * important: switching layouts reuses the WebView HTTP/decode cache instead of
 * fetching a second derivative, and old remote Mold servers get the same
 * optimization as current ones.
 */
export function catalogThumbnailUrl(raw: string): string {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return raw;
  }
  if (!CIVITAI_IMAGE_HOSTS.has(url.hostname)) return raw;

  const segments = url.pathname.split("/");
  const transformIndex = segments.findIndex(
    (segment) => segment.startsWith("original=") || segment.startsWith("width="),
  );
  if (transformIndex < 0) return raw;

  segments[transformIndex] = `width=${CATALOG_THUMBNAIL_WIDTH}`;
  url.pathname = segments.join("/");
  return url.toString();
}
