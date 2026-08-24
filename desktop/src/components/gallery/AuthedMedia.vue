<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from "vue";
import { galleryThumbnailScheduler, type ThumbnailHandle } from "@studio/lib/thumbnailScheduler";
import { authedMediaUrl, fullSizeMediaUrl } from "../../lib/gallery/media";
import type { ApiTarget } from "../../lib/api/client";

const props = withDefaults(
  defineProps<{
    path: string;
    video?: boolean;
    /** Audio-only print: renders a transport instead of a raster element. */
    audio?: boolean;
    alt?: string;
    controls?: boolean;
    /** Explicit host to fetch from; defaults to the primary connection. */
    target?: ApiTarget | null;
    /** Blob-cache bucket, usually the origin host id. */
    cacheKey?: string | null;
    mediaVersion?: string | null;
  }>(),
  {
    video: false,
    audio: false,
    alt: "",
    controls: false,
    target: null,
    cacheKey: null,
    mediaVersion: null,
  },
);

const src = ref<string | null>(null);
const failed = ref(false);
let loadEpoch = 0;
let thumbnailHandle: ThumbnailHandle<string> | null = null;

const retryDelaysMs = [0, 250, 1_000] as const;

async function load() {
  thumbnailHandle?.cancel();
  thumbnailHandle = null;
  const epoch = ++loadEpoch;
  src.value = null;
  failed.value = false;
  const delays = props.path.startsWith("/api/gallery/thumbnail/") ? retryDelaysMs : ([0] as const);
  for (const delayMs of delays) {
    if (delayMs > 0) await new Promise((resolve) => setTimeout(resolve, delayMs));
    if (epoch !== loadEpoch) return;
    try {
      const options = {
        ...(props.target ? { target: props.target } : {}),
        ...(props.cacheKey ? { cacheKey: props.cacheKey } : {}),
        ...(props.mediaVersion ? { mediaVersion: props.mediaVersion } : {}),
      };
      // Thumbnails and full-size stills/audio go native-first in the desktop
      // app (#1132's rationale: a media element pointed straight at a host
      // shares WebKit's per-host connection pool with every held-open
      // generation/download stream to that host). Video keeps the ticketed
      // or direct streaming URL so it can seek without buffering; outside
      // Tauri, or when the native route refuses, stills fall back to it too.
      const thumbnail = props.path.startsWith("/api/gallery/thumbnail/");
      const url = thumbnail
        ? await (() => {
            const handle = galleryThumbnailScheduler.schedule({
              key: `${props.cacheKey ?? "primary"}|${props.path}|${props.mediaVersion ?? "legacy"}|${props.target?.baseUrl ?? "primary"}|${props.target?.apiKey ?? ""}`,
              hostKey: props.cacheKey ?? props.target?.baseUrl ?? "primary",
              priority: "visible",
              run: (signal) => authedMediaUrl(props.path, { ...options, signal }),
            });
            thumbnailHandle = handle;
            return handle.promise;
          })()
        : await fullSizeMediaUrl(props.path, {
            ...options,
            allowLegacyBlob: !props.video && !props.audio,
            video: props.video,
          });
      if (epoch === loadEpoch) src.value = url;
      return;
    } catch {
      // Retry bounded transient native/network failures. The media cache
      // evicts rejected promises, so each attempt performs a fresh request.
    }
  }
  if (epoch === loadEpoch) failed.value = true;
}

// Watch the route's VALUES, never a freshly built array: a getter returning
// `[...]` yields a new array on every parent re-render, and Vue compares the
// source by identity, so each re-render (selecting a tile, a poll landing)
// re-ran `load()`, nulled `src`, and swapped the <img> for the shimmer. Parents
// build `target` per render, so that swap happened between the two clicks of
// a double-click on every remote tile — the lightbox never opened for
// remote-only prints. The multi-source form compares each value on its own.
watch(
  [
    () => props.path,
    () => props.cacheKey,
    () => props.mediaVersion,
    () => props.target?.baseUrl,
    () => props.target?.apiKey,
  ],
  load,
);
onMounted(load);
onUnmounted(() => {
  loadEpoch += 1;
  thumbnailHandle?.cancel();
  thumbnailHandle = null;
});
</script>

<template>
  <video
    v-if="video && src"
    :src="src"
    class="h-full w-full object-contain"
    :controls="controls"
    loop
    playsinline
    disablepictureinpicture
  />
  <audio v-else-if="audio && src" :src="src" class="w-full" controls />
  <img v-else-if="src" :src="src" :alt="alt" class="h-full w-full object-cover" draggable="false" />
  <div v-else-if="failed" class="flex h-full w-full items-center justify-center bg-bench">
    <span class="edge-code">UNREADABLE</span>
  </div>
  <div v-else class="media-placeholder grain-shimmer h-full w-full" aria-hidden="true">
    <span>Loading preview</span>
  </div>
</template>

<style scoped>
.media-placeholder {
  display: grid;
  place-items: center;
  background-color: color-mix(in srgb, var(--bench) 82%, var(--safelight) 18%);
  background-image:
    radial-gradient(
      circle at 28% 24%,
      color-mix(in srgb, var(--halide) 16%, transparent),
      transparent 34%
    ),
    radial-gradient(
      circle at 72% 76%,
      color-mix(in srgb, var(--safelight) 14%, transparent),
      transparent 38%
    );
}

.media-placeholder span {
  border: 1px solid color-mix(in srgb, var(--rebate) 20%, transparent);
  border-radius: 999px;
  padding: 5px 9px;
  background: color-mix(in srgb, var(--bench) 78%, transparent);
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
</style>
