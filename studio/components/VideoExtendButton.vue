<script setup lang="ts">
import { inject, onBeforeUnmount, ref, watch } from "vue";
import { apiJsonTo } from "../api/client";
import type { VideoJobOwner } from "../api/videoJobs";
import { inspectVideoLibraryExtension } from "../api/videoLibraryExtension";
import { VIDEO_LIBRARY_EXTENSION_NAVIGATION } from "../lib/videoLibraryExtensionHandoff";
const props = defineProps<{
  owner: VideoJobOwner | null;
  filename: string;
  metadata: object;
  title?: string | null;
}>();
const emit = defineEmits<{ (event: "available", value: boolean): void }>();
const navigate = inject(VIDEO_LIBRARY_EXTENSION_NAVIGATION, null);
const available = ref(false),
  busy = ref(false),
  error = ref("");
let epoch = 0;
function identity() {
  const o = props.owner;
  return [
    o?.hostId,
    o?.instanceId,
    o?.target.baseUrl,
    o?.target.apiKey,
    o?.target.connectionMode,
    props.filename,
  ];
}
watch(
  () => JSON.stringify(identity()),
  async () => {
    const token = ++epoch;
    available.value = false;
    error.value = "";
    emit("available", false);
    const owner = props.owner;
    if (!owner || !navigate) return;
    try {
      const caps = await apiJsonTo<{
        video_jobs?: {
          supported?: boolean;
          literal_prompts?: boolean;
          library_extend?: boolean;
        };
      }>(owner.target, "/api/capabilities");
      if (token !== epoch) return;
      available.value =
        caps.video_jobs?.supported === true &&
        caps.video_jobs.literal_prompts === true &&
        caps.video_jobs.library_extend === true;
      emit("available", available.value);
    } catch {
      /* Existing Continue remains available on an older/offline host. */
    }
  },
  { immediate: true },
);
async function extend() {
  const owner = props.owner,
    token = epoch;
  if (!owner || !navigate || busy.value || !available.value) return;
  const captured = { ...owner, target: { ...owner.target } },
    filename = props.filename;
  busy.value = true;
  error.value = "";
  try {
    const result = await inspectVideoLibraryExtension(
      captured,
      filename,
      props.metadata as Record<string, unknown>,
      () => token === epoch,
      props.title ?? undefined,
    );
    if (token === epoch) await navigate(result);
  } catch (cause) {
    if (token === epoch)
      error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
  }
}
onBeforeUnmount(() => {
  epoch++;
});
</script>
<template>
  <button v-if="available" type="button" :disabled="busy" @click.stop="extend">
    {{ busy ? "Checking original…" : "Extend video…" }}
  </button>
  <p v-if="error" role="alert">{{ error }}</p>
</template>
