<script setup lang="ts">
import { computed, watch } from "vue";
import {
  clampVideoFrames,
  formatVideoDuration,
  maxVideoFrames,
  minVideoFrames,
  videoFrameStep,
  type VideoFrameContract,
} from "@studio/lib/videoDuration";
import SliderRow from "./SliderRow.vue";

const props = withDefaults(
  defineProps<{
    frames: number;
    fps: number;
    model?: VideoFrameContract | null;
    label?: string;
    touchFriendly?: boolean;
  }>(),
  { model: null, label: "Duration" },
);

const emit = defineEmits<{ "update:frames": [frames: number] }>();

const rate = computed(() => Math.max(1, Math.round(props.fps) || 24));
const sliderValue = computed(() =>
  clampVideoFrames(props.frames, rate.value, props.model),
);
const maximum = computed(() => maxVideoFrames(props.model, rate.value));
const isLongVideo = computed(() => props.frames > maximum.value);
const displayedFrames = computed(() =>
  isLongVideo.value ? props.frames : sliderValue.value,
);
const readout = computed(() =>
  formatVideoDuration(displayedFrames.value, rate.value),
);

// FPS/model changes can lower a previously legal single-shot value. Clamp that
// transition, but preserve an intentional above-ceiling exact frame count: the
// advanced fields use those to request automatic long-video chaining.
watch(maximum, (next, previous) => {
  if (props.frames <= previous && props.frames > next) {
    emit(
      "update:frames",
      clampVideoFrames(props.frames, rate.value, props.model),
    );
  }
});

function update(frames: number): void {
  emit("update:frames", clampVideoFrames(frames, rate.value, props.model));
}
</script>

<template>
  <div
    class="video-duration"
    :class="{ 'video-duration--touch': touchFriendly }"
    data-test="video-duration"
  >
    <SliderRow
      :model-value="sliderValue"
      :min="minVideoFrames(model)"
      :max="maximum"
      :step="videoFrameStep(model)"
      :label="label"
      :value-label="readout"
      @update:model-value="update"
    />
    <p class="video-duration__hint" data-test="video-duration-detail">
      {{ displayedFrames }} frames · {{ rate }} fps · {{ readout }}
      <template v-if="isLongVideo"> · automatic sequence</template>
    </p>
  </div>
</template>

<style scoped>
.video-duration__hint {
  margin: 7px 0 0;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  line-height: 1.4;
}

.video-duration--touch :deep(.ms-slider__input) {
  box-sizing: border-box;
  height: 44px;
  background: linear-gradient(
    to bottom,
    transparent 20px,
    var(--ce) 20px,
    var(--ce) 24px,
    transparent 24px
  );
}
</style>
