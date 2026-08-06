<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type {
  GifPlayback,
  GifRepeat,
  VideoExportFormat,
  VideoExportOptions,
} from "@studio/lib/videoExport";

const props = withDefaults(
  defineProps<{
    open: boolean;
    filename: string;
    formats: VideoExportFormat[];
    busy?: boolean;
    error?: string;
  }>(),
  { busy: false, error: "" },
);

const emit = defineEmits<{
  close: [];
  export: [options: VideoExportOptions];
}>();

const format = ref<VideoExportFormat>("gif");
const playback = ref<GifPlayback>("loop");
const repeat = ref<GifRepeat>("forever");
const maxDimension = ref<number | null>(720);
const fps = ref<number | null>(12);
const isGif = computed(() => format.value === "gif");

watch(
  () => [props.open, props.formats] as const,
  ([open]) => {
    if (!open) return;
    if (!props.formats.includes(format.value))
      format.value = props.formats[0] ?? "gif";
  },
  { immediate: true },
);

function submit(): void {
  emit("export", {
    format: format.value,
    playback: isGif.value ? playback.value : "loop",
    repeat: isGif.value ? repeat.value : "forever",
    max_dimension: maxDimension.value,
    fps: fps.value,
  });
}
</script>

<template>
  <div
    v-if="open"
    class="video-export-scrim"
    role="dialog"
    aria-modal="true"
    aria-labelledby="video-export-title"
    data-test="video-export-dialog"
    @click.self="!busy && emit('close')"
    @keydown.esc.stop="!busy && emit('close')"
  >
    <form class="video-export-card" @submit.prevent="submit">
      <div class="video-export-heading">
        <div>
          <span class="video-export-kicker">Video export</span>
          <h2 id="video-export-title">Export another format</h2>
        </div>
        <button
          type="button"
          aria-label="Close export options"
          :disabled="busy"
          @click="emit('close')"
        >
          ×
        </button>
      </div>
      <p class="video-export-file">{{ filename }}</p>

      <fieldset>
        <legend>Format</legend>
        <div class="video-export-options">
          <label v-for="candidate in formats" :key="candidate">
            <input
              v-model="format"
              type="radio"
              name="export-format"
              :value="candidate"
            />
            <span>{{ candidate.toUpperCase() }}</span>
          </label>
        </div>
      </fieldset>

      <template v-if="isGif">
        <fieldset>
          <legend>Playback</legend>
          <div class="video-export-options">
            <label>
              <input
                v-model="playback"
                type="radio"
                name="gif-playback"
                value="loop"
              />
              <span>Loop</span>
            </label>
            <label>
              <input
                v-model="playback"
                type="radio"
                name="gif-playback"
                value="bounce"
              />
              <span>Bounce</span>
            </label>
          </div>
          <p>Bounce plays forward, then reverses smoothly.</p>
        </fieldset>
        <fieldset>
          <legend>Repeat</legend>
          <div class="video-export-options">
            <label>
              <input
                v-model="repeat"
                type="radio"
                name="gif-repeat"
                value="forever"
              />
              <span>Forever</span>
            </label>
            <label>
              <input
                v-model="repeat"
                type="radio"
                name="gif-repeat"
                value="once"
              />
              <span>Once</span>
            </label>
          </div>
        </fieldset>
      </template>

      <fieldset>
        <legend>Longest side</legend>
        <div class="video-export-options video-export-options--four">
          <label
            v-for="choice in [null, 1080, 720, 480]"
            :key="choice ?? 'original'"
          >
            <input
              v-model="maxDimension"
              type="radio"
              name="export-size"
              :value="choice"
            />
            <span>{{ choice ? `${choice} px` : "Original" }}</span>
          </label>
        </div>
      </fieldset>
      <fieldset>
        <legend>Frame rate</legend>
        <div class="video-export-options video-export-options--four">
          <label
            v-for="choice in [null, 24, 12, 8]"
            :key="choice ?? 'original'"
          >
            <input
              v-model="fps"
              type="radio"
              name="export-fps"
              :value="choice"
            />
            <span>{{ choice ? `${choice} fps` : "Original" }}</span>
          </label>
        </div>
      </fieldset>

      <p v-if="error" class="video-export-error" role="alert">{{ error }}</p>
      <div class="video-export-actions">
        <button type="button" :disabled="busy" @click="emit('close')">
          Cancel
        </button>
        <button
          type="submit"
          class="video-export-primary"
          :disabled="busy || formats.length === 0"
        >
          {{ busy ? "Converting…" : "Export" }}
        </button>
      </div>
    </form>
  </div>
</template>

<style scoped>
.video-export-scrim {
  position: fixed;
  z-index: 1000;
  inset: 0;
  display: grid;
  place-items: center;
  padding: max(20px, env(safe-area-inset-top)) 20px
    max(20px, env(safe-area-inset-bottom));
  background: rgb(6 5 10 / 72%);
  backdrop-filter: blur(8px);
}
.video-export-card {
  width: min(420px, 100%);
  max-height: 100%;
  overflow: auto;
  padding: 22px;
  border: 1px solid var(--edge, rgb(128 120 140 / 35%));
  border-radius: 18px;
  background: var(--bench, #f7f4ef);
  color: var(--rebate, #171316);
  box-shadow: 0 24px 80px rgb(0 0 0 / 38%);
}
.video-export-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}
.video-export-heading h2 {
  margin: 4px 0 0;
  font-size: 20px;
  line-height: 1.2;
}
.video-export-heading button {
  width: 44px;
  height: 44px;
  border: 0;
  border-radius: 999px;
  background: rgb(128 120 140 / 12%);
  color: inherit;
  font-size: 25px;
}
.video-export-kicker,
legend {
  font-family: var(--f-mono, monospace);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3, #777078);
}
.video-export-file {
  margin: 14px 0 18px;
  overflow: hidden;
  color: var(--ink-2, #575057);
  font-family: var(--f-mono, monospace);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
fieldset {
  margin: 0 0 17px;
  padding: 0;
  border: 0;
}
legend {
  margin-bottom: 8px;
}
fieldset > p {
  margin: 8px 0 0;
  color: var(--ink-3, #777078);
  font-size: 12px;
}
.video-export-options {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}
.video-export-options label {
  position: relative;
}
.video-export-options--four {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.video-export-options--four span {
  padding: 0 4px;
  font-size: 12px;
}
.video-export-options input {
  position: absolute;
  opacity: 0;
}
.video-export-options span {
  display: grid;
  min-height: 44px;
  place-items: center;
  border: 1px solid var(--edge, rgb(128 120 140 / 35%));
  border-radius: 10px;
  font-weight: 650;
}
.video-export-options input:checked + span {
  border-color: var(--safelight, #ad5700);
  background: color-mix(in srgb, var(--safelight, #ad5700) 12%, transparent);
  color: var(--safelight, #ad5700);
}
.video-export-options input:focus-visible + span {
  outline: 2px solid var(--safelight, #ad5700);
  outline-offset: 2px;
}
.video-export-error {
  margin: 0 0 14px;
  color: var(--stop, #b32222);
  font-size: 13px;
}
.video-export-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 5px;
}
.video-export-actions button {
  min-height: 44px;
  border: 1px solid var(--edge, rgb(128 120 140 / 35%));
  border-radius: 10px;
  background: transparent;
  color: inherit;
  font-weight: 700;
}
.video-export-actions .video-export-primary {
  border-color: transparent;
  background: var(--safelight, #ad5700);
  color: var(--on-accent, white);
}
button:disabled {
  opacity: 0.55;
}
</style>
