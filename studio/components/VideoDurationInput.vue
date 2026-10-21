<script setup lang="ts">
import { computed, ref, watch } from "vue";
import {
  formatRequestedVideoDuration,
  parseRequestedVideoDuration,
  type VideoDurationSummary,
} from "../lib/videoRequestedDuration";
const props = defineProps<{
  modelValue: string | null;
  plan?: VideoDurationSummary | null;
  checking?: boolean;
  error?: string | null;
  disabled?: boolean;
  label?: string;
  presets?: boolean;
}>();
const emit = defineEmits<{
  (event: "update:modelValue", value: string | null): void;
}>();
const text = ref(
  props.modelValue === null
    ? ""
    : formatRequestedVideoDuration(props.modelValue),
);
const touched = ref(false);
watch(
  () => props.modelValue,
  (value) => {
    if (value !== null) text.value = formatRequestedVideoDuration(value);
  },
);
const invalid = computed(
  () => touched.value && parseRequestedVideoDuration(text.value) === null,
);
const safeSeconds = computed(() => {
  const plan = props.plan;
  if (
    !plan ||
    !Number.isFinite(plan.fps) ||
    plan.fps <= 0 ||
    !Number.isSafeInteger(plan.single_clip_max_frames) ||
    plan.single_clip_max_frames < 1
  )
    return null;
  return formatRequestedVideoDuration(
    String(Math.floor((plan.single_clip_max_frames * 1000) / plan.fps)),
  );
});
const count = computed(() =>
  props.plan && /^[1-9]\d*$/.test(props.plan.clip_count)
    ? props.plan.clip_count
    : null,
);
function update(value: string) {
  touched.value = true;
  text.value = value;
  emit("update:modelValue", parseRequestedVideoDuration(value));
}
</script>
<template>
  <div class="video-duration" data-test="video-duration">
    <label
      >{{ label ?? "Duration" }}
      <span class="video-duration__input"
        ><input
          :value="text"
          type="text"
          inputmode="decimal"
          :aria-label="`${label ?? 'Duration'} in seconds`"
          :disabled="disabled"
          @input="update(($event.target as HTMLInputElement).value)"
        />
        seconds</span
      >
    </label>
    <div
      v-if="presets !== false"
      class="video-duration__presets"
      aria-label="Duration presets"
    >
      <button
        v-for="seconds in [5, 10, 30, 60]"
        :key="seconds"
        type="button"
        :disabled="disabled"
        @click="update(String(seconds))"
      >
        {{ seconds }}s
      </button>
    </div>
    <p v-if="safeSeconds">Single generation: up to {{ safeSeconds }} seconds</p>
    <p v-if="checking" role="status">Checking this machine…</p>
    <p v-else-if="count && count !== '1'" role="status">
      {{ count }} clips joined into one video
    </p>
    <p v-if="invalid || error" role="alert" class="video-duration__error">
      {{
        invalid
          ? "Enter a positive duration in seconds, with up to three decimal places."
          : error
      }}
    </p>
  </div>
</template>
<style scoped>
.video-duration {
  display: grid;
  gap: 8px;
}
.video-duration label {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.video-duration__input {
  display: flex;
  align-items: center;
  gap: 8px;
}
.video-duration input {
  width: 9em;
}
.video-duration input,
.video-duration button {
  min-height: 44px;
  padding: 8px 12px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg);
  color: var(--mold-text);
  font: inherit;
  font-size: max(
    16px,
    var(--mold-fs-md)
  ); /* literal: prevent mobile input zoom */
}
.video-duration__presets {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.video-duration p {
  margin: 0;
  color: var(--mold-text-dim);
}
.video-duration .video-duration__error {
  color: var(--mold-error);
}
</style>
