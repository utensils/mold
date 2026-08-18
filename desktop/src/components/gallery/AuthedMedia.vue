<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from "vue";
import { authedMediaUrl, streamableMediaUrl } from "../../lib/gallery/media";
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
  }>(),
  { video: false, audio: false, alt: "", controls: false, target: null, cacheKey: null },
);

const src = ref<string | null>(null);
const failed = ref(false);
let loadEpoch = 0;

const retryDelaysMs = [0, 250, 1_000] as const;

async function load() {
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
      };
      // Full-size remote media must stay streamable. Fetching it into a blob
      // blocks the viewer until the entire file crosses WebKit and prevents
      // video seeking; authenticated hosts instead mint a short-lived,
      // exact-path ticket that the media element can load directly.
      const url = props.path.startsWith("/api/gallery/thumbnail/")
        ? await authedMediaUrl(props.path, options)
        : await streamableMediaUrl(props.path, {
            ...options,
            allowLegacyBlob: !props.video && !props.audio,
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

watch(() => [props.path, props.cacheKey, props.target?.baseUrl, props.target?.apiKey], load);
onMounted(load);
onUnmounted(() => {
  loadEpoch += 1;
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
  <div v-else class="grain-shimmer h-full w-full" />
</template>
