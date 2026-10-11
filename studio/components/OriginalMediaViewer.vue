<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";

/** An inspection-only control. Callers supply original bytes, never thumbnails. */
const props = defineProps<{
  data?: string | null | undefined;
  mimeType?: string | null | undefined;
  label?: string | undefined;
  load?: ((signal: AbortSignal) => Promise<Blob>) | undefined;
  /** Immutable authority fingerprint for asynchronous originals. */
  authority?: string | undefined;
}>();
const emit = defineEmits<{ close: [] }>();
const open = ref(true);
const url = ref("");
const kind = ref("");
const error = ref("");
const loading = ref(false);
const actual = ref(false);
const player = ref<HTMLMediaElement>();
let controller: AbortController | undefined;
function release() {
  controller?.abort();
  controller = undefined;
  if (player.value) {
    player.value.pause();
    player.value.removeAttribute("src");
    player.value.load();
  }
  if (url.value) URL.revokeObjectURL(url.value);
  url.value = "";
  loading.value = false;
}
function close() {
  release();
  open.value = false;
  emit("close");
}
async function show() {
  release();
  open.value = true;
  error.value = "";
  actual.value = false;
  loading.value = true;
  const request = new AbortController();
  controller = request;
  try {
    let blob: Blob;
    if (props.load) blob = await props.load(request.signal);
    else {
      if (!props.data || !props.mimeType)
        throw new Error(
          "Original media unavailable. Reattach the original to inspect it.",
        );
      if (props.data.length > Math.ceil((64 * 1024 * 1024) / 3) * 4)
        throw new Error("Original media is too large.");
      const decoded = atob(props.data);
      const bytes = Uint8Array.from(decoded, (c) => c.charCodeAt(0));
      const ascii = (start: number, end: number) =>
        String.fromCharCode(...bytes.slice(start, end));
      const type =
        bytes[0] === 137 && ascii(1, 4) === "PNG"
          ? "image/png"
          : bytes[0] === 255 && bytes[1] === 216
            ? "image/jpeg"
            : ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP"
              ? "image/webp"
              : ascii(0, 4) === "RIFF" && ascii(8, 12) === "WAVE"
                ? "audio/wav"
                : ascii(4, 8) === "ftyp"
                  ? "video/mp4"
                  : props.mimeType;
      blob = new Blob([bytes], { type });
    }
    if (request.signal.aborted) return;
    if (blob.size > 64 * 1024 * 1024)
      throw new Error("Original media is too large.");
    const type = blob.type.split(";")[0] ?? "";
    kind.value = type.startsWith("image/")
      ? "image"
      : type.startsWith("video/")
        ? "video"
        : type.startsWith("audio/")
          ? "audio"
          : "";
    if (!kind.value)
      throw new Error("This original media format cannot be viewed.");
    url.value = URL.createObjectURL(blob);
  } catch (reason) {
    if (!request.signal.aborted)
      error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    if (!request.signal.aborted) loading.value = false;
  }
}
watch(
  [() => props.data, () => props.mimeType, () => props.authority],
  () => void show(),
  { immediate: true },
);
onBeforeUnmount(release);
</script>
<template>
  <ModalPanel
    :open="open"
    :width="1100"
    :title="label || 'Original media'"
    @close="close"
  >
    <p v-if="loading" role="status">Loading original…</p>
    <div v-else-if="error">
      <p role="alert">{{ error }}</p>
      <button v-if="load" type="button" @click="show">Retry</button>
    </div>
    <div
      v-else
      class="original-media__viewport"
      tabindex="0"
      role="region"
      :aria-label="label || 'Original media preview'"
    >
      <img
        v-if="kind === 'image'"
        :src="url"
        :alt="label || 'Original image'"
        class="ms-alpha-bed"
        :class="{ 'original-media__actual': actual }"
        @error="error = 'The original image could not be decoded.'"
      />
      <video
        v-else-if="kind === 'video'"
        ref="player"
        :src="url"
        controls
        preload="metadata"
        @error="error = 'This original media could not be played.'"
      />
      <audio
        v-else-if="kind === 'audio'"
        ref="player"
        :src="url"
        controls
        preload="metadata"
        @error="error = 'This original media could not be played.'"
      />
    </div>
    <template #footer>
      <button
        v-if="kind === 'image' && url"
        type="button"
        data-test="original-actual-size"
        :aria-pressed="actual"
        @click="actual = !actual"
      >
        {{ actual ? "Fit" : "Actual size" }}
      </button>
      <button type="button" data-test="original-close" @click="close">
        Done
      </button>
    </template>
  </ModalPanel>
</template>
<style scoped>
button {
  min-height: 48px;
  min-width: 48px;
}

.original-media__viewport {
  max-height: 75vh;
  overflow: auto;
  text-align: center;
}
.original-media__viewport img,
.original-media__viewport video {
  display: block;
  margin: 0 auto;
  max-width: 100%;
  max-height: 70vh;
  object-fit: contain;
}
.original-media__viewport img.original-media__actual {
  max-width: none;
  max-height: none;
}
.original-media__viewport audio {
  width: 100%;
}
</style>
