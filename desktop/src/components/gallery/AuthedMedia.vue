<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { authedMediaUrl } from "../../lib/gallery/media";

const props = withDefaults(
  defineProps<{
    path: string;
    video?: boolean;
    alt?: string;
    controls?: boolean;
  }>(),
  { video: false, alt: "", controls: false },
);

const src = ref<string | null>(null);
const failed = ref(false);

async function load() {
  src.value = null;
  failed.value = false;
  try {
    src.value = await authedMediaUrl(props.path);
  } catch {
    failed.value = true;
  }
}

watch(() => props.path, load);
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
  />
  <img v-else-if="src" :src="src" :alt="alt" class="h-full w-full object-cover" draggable="false" />
  <div v-else-if="failed" class="flex h-full w-full items-center justify-center bg-bench">
    <span class="edge-code">UNREADABLE</span>
  </div>
  <div v-else class="h-full w-full animate-pulse bg-bench" />
</template>
