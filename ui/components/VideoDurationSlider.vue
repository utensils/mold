<script setup lang="ts">
import { computed, watch } from "vue";
import {
  clampVideoFrames,
  formatVideoDuration,
  maxVideoFrames,
  minVideoFrames,
  videoGenerationCount,
  videoGenerationMarks,
  videoFrameStep,
  type VideoFrameContract,
} from "@studio/lib/videoDuration";
import type { GenerateRoutingRequest } from "@studio/lib/chainRouting";
import SliderRow from "./SliderRow.vue";

const props = withDefaults(
  defineProps<{
    frames: number;
    fps: number;
    model?: VideoFrameContract | null;
    family?: string | null | undefined;
    modelName?: string | null | undefined;
    sourceImageCapability?: string | null | undefined;
    routingRequest?: Partial<GenerateRoutingRequest> | null | undefined;
    label?: string;
    touchFriendly?: boolean;
  }>(),
  { model: null, label: "Duration" },
);

const emit = defineEmits<{ "update:frames": [frames: number] }>();

const contract = computed<VideoFrameContract>(() => ({
  ...(props.model ?? {}),
  family: props.model?.family ?? props.family,
  name: props.model?.name ?? props.modelName,
  source_image: props.model?.source_image ?? props.sourceImageCapability,
}));
const rate = computed(() => Math.max(1, Math.round(props.fps) || 24));
const sliderValue = computed(() =>
  clampVideoFrames(props.frames, rate.value, contract.value),
);
const maximum = computed(() => maxVideoFrames(contract.value, rate.value));
/** An intentional above-ceiling count. Drives the readout only: whether it is
 * a sequence is `generations`' answer, and a refused count is not one. */
const isLongVideo = computed(() => props.frames > maximum.value);
const displayedFrames = computed(() =>
  isLongVideo.value ? props.frames : sliderValue.value,
);
const readout = computed(() =>
  formatVideoDuration(displayedFrames.value, rate.value),
);
// `null` when the routing authority REFUSES this duration outright — a
// text-to-video wan tier past its clip size, say, which cannot be split into a
// sequence. It is not "1 generation": saying so advertised a single render for
// a frame count submit answers 422 to. The ceiling above normally keeps the
// slider out of that range; this is what keeps the readout honest if anything
// else puts a refused count in the field.
const generations = computed(() =>
  videoGenerationCount(
    displayedFrames.value,
    rate.value,
    contract.value,
    props.routingRequest ?? {},
  ),
);
const generationsLabel = computed(() =>
  generations.value === null
    ? null
    : `${generations.value} ${generations.value === 1 ? "generation" : "generations"}`,
);
const marks = computed(() =>
  videoGenerationMarks(
    rate.value,
    contract.value,
    props.routingRequest ?? {},
  ).map((mark) => ({
    value: mark.frames,
    label: mark.label,
    title: `${mark.generations} ${mark.generations === 1 ? "generation" : "generations"} · ${formatVideoDuration(mark.frames, rate.value)}`,
  })),
);

// FPS/model changes can lower a previously legal single-shot value. Clamp that
// transition, but preserve an intentional above-ceiling exact frame count: the
// advanced fields use those to request automatic long-video chaining.
watch(maximum, (next, previous) => {
  if (props.frames <= previous && props.frames > next) {
    emit(
      "update:frames",
      clampVideoFrames(props.frames, rate.value, contract.value),
    );
  }
});

function update(frames: number): void {
  emit("update:frames", clampVideoFrames(frames, rate.value, contract.value));
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
      :min="minVideoFrames(contract)"
      :max="maximum"
      :step="videoFrameStep(contract)"
      :label="label"
      :value-label="readout"
      :aria-value-text="
        generationsLabel ? `${readout}, ${generationsLabel}` : readout
      "
      :marks="marks"
      :snap-threshold-ratio="touchFriendly ? 0.04 : 0.015"
      @update:model-value="update"
    />
    <p class="video-duration__hint" data-test="video-duration-detail">
      {{ displayedFrames }} frames · {{ rate }} fps · {{ readout
      }}<template v-if="generationsLabel"> · {{ generationsLabel }}</template>
      <template v-if="(generations ?? 0) > 1"> · automatic sequence</template>
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
