<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { authedMediaUrl } from "../../lib/gallery/media";
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

async function load() {
  src.value = null;
  failed.value = false;
  try {
    src.value = await authedMediaUrl(props.path, {
      ...(props.target ? { target: props.target } : {}),
      ...(props.cacheKey ? { cacheKey: props.cacheKey } : {}),
    });
  } catch {
    failed.value = true;
  }
}

watch(() => [props.path, props.cacheKey], load);
onMounted(load);
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
