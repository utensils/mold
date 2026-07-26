<script setup lang="ts">
/*
 * Stage card (Mold Studio Sequence composer) — one clip in a chain script.
 * Kit-native: Icon glyphs (no emoji/text symbols), token surfaces, a proper
 * drop-target attach tile, and a transition segmented control. The grip keeps
 * the `.drag-handle` class VueDraggable binds to for reordering.
 */
import { computed } from "vue";
import Icon from "@ui/components/Icon.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import type { ChainStageToml } from "../lib/chainToml";
import { sequenceFrameOptions, transitionLabel } from "@studio/lib/sequence";

const props = defineProps<{
  index: number;
  isFirst: boolean;
  stage: ChainStageToml;
  framesPerClipCap: number;
  fadeFramesMax: number;
  motionTailFrames: number;
}>();

const emit = defineEmits<{
  (e: "update:stage", v: ChainStageToml): void;
  (e: "delete"): void;
  (e: "move-up"): void;
  (e: "move-down"): void;
  (e: "duplicate"): void;
  (e: "expand"): void;
  (e: "pick-image"): void;
  (e: "clear-image"): void;
}>();

function updatePrompt(v: string) {
  emit("update:stage", { ...props.stage, prompt: v });
}
function updateFrames(v: number) {
  emit("update:stage", { ...props.stage, frames: v });
}
function updateTransition(v: string) {
  emit("update:stage", {
    ...props.stage,
    transition: v as "smooth" | "cut" | "fade",
  });
}

const transitionOptions = computed(() => [
  { value: "smooth", label: transitionLabel("smooth", props.motionTailFrames) },
  { value: "cut", label: transitionLabel("cut", props.motionTailFrames) },
  { value: "fade", label: transitionLabel("fade", props.motionTailFrames) },
]);
const transitionValue = computed(() => props.stage.transition ?? "smooth");

const durationSec = computed(() => (props.stage.frames / 24).toFixed(2));
const frameOptions = computed(() =>
  sequenceFrameOptions(props.framesPerClipCap, props.motionTailFrames),
);

const hasSourceImage = computed(() => Boolean(props.stage.source_image_b64));
const sourceImageLabel = computed(() => {
  const name = props.stage.source_image;
  if (name && name.length > 0) return name;
  return "source image";
});
</script>

<template>
  <div class="sc" data-test="stage-card">
    <div class="sc__head">
      <button
        type="button"
        class="sc__grip drag-handle"
        data-test="stage-drag-handle"
        aria-label="Drag to reorder clip"
        title="Drag to reorder"
      >
        <Icon name="grip" :size="16" />
      </button>
      <span class="sc__index">{{ index + 1 }}</span>

      <SegmentedControl
        v-if="!isFirst"
        class="sc__transition"
        :model-value="transitionValue"
        :options="transitionOptions"
        label="Transition"
        data-test="stage-transition"
        @update:model-value="updateTransition"
      />
      <span v-else class="sc__opening">Opening clip</span>

      <div class="sc__actions">
        <select
          class="sc__frames"
          :value="stage.frames"
          aria-label="Duration for this clip"
          @change="
            updateFrames(Number(($event.target as HTMLSelectElement).value))
          "
        >
          <option v-for="n in frameOptions" :key="n" :value="n">
            {{ n }}f · {{ (n / 24).toFixed(2) }}s
          </option>
        </select>
        <button
          type="button"
          class="sc__icon"
          aria-label="Duplicate clip"
          title="Duplicate"
          @click="emit('duplicate')"
        >
          <Icon name="copy" :size="15" />
        </button>
        <button
          type="button"
          class="sc__icon"
          aria-label="Move clip up"
          title="Move up"
          @click="emit('move-up')"
        >
          <Icon name="arrow-up" :size="15" />
        </button>
        <button
          type="button"
          class="sc__icon"
          aria-label="Move clip down"
          title="Move down"
          @click="emit('move-down')"
        >
          <Icon name="arrow-down" :size="15" />
        </button>
        <button
          type="button"
          class="sc__icon sc__icon--danger"
          aria-label="Delete clip"
          title="Delete"
          @click="emit('delete')"
        >
          <Icon name="close" :size="15" />
        </button>
      </div>
    </div>

    <div class="sc__body">
      <div v-if="hasSourceImage" class="sc__thumb">
        <img
          :src="`data:image/png;base64,${stage.source_image_b64}`"
          :alt="sourceImageLabel"
        />
        <button
          type="button"
          class="sc__thumb-remove"
          aria-label="Remove source image"
          title="Remove source image"
          @click="emit('clear-image')"
        >
          <Icon name="close" :size="12" />
        </button>
      </div>
      <button
        v-else
        type="button"
        class="sc__attach"
        data-test="stage-attach"
        aria-label="Attach source image"
        title="Attach source image"
        @click="emit('pick-image')"
      >
        <Icon name="image" :size="18" />
      </button>

      <textarea
        class="sc__prompt"
        :value="stage.prompt"
        :placeholder="
          isFirst ? 'How does the sequence begin?' : 'What happens next?'
        "
        @input="updatePrompt(($event.target as HTMLTextAreaElement).value)"
      />
    </div>

    <div class="sc__foot">
      <button type="button" class="sc__link" @click="emit('expand')">
        <Icon name="sparkle" :size="13" />
        expand
      </button>
      <button type="button" class="sc__link" @click="emit('pick-image')">
        <Icon name="image" :size="13" />
        {{ hasSourceImage ? "replace image" : "attach image" }}
      </button>
      <span class="sc__dur">{{ durationSec }}s</span>
      <span
        v-if="
          !isFirst &&
          (stage.transition ?? 'smooth') === 'smooth' &&
          stage.source_image_b64
        "
        class="sc__anchor"
        title="Smooth transitions use the prior clip's motion tail; this image still anchors character/scene identity across the stage boundary"
      >
        identity anchor across the cut
      </span>
    </div>
  </div>
</template>

<style scoped>
.sc {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.sc__head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.sc__grip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  cursor: grab;
  padding: 2px;
}
.sc__grip:active {
  cursor: grabbing;
}
.sc__index {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
  min-width: 12px;
}
.sc__transition {
  flex: 0 0 auto;
}
.sc__opening {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.sc__actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 5px;
}
.sc__frames {
  height: 30px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 11px;
  padding: 0 8px;
}
.sc__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 30px;
  width: 30px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.sc__icon:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}
.sc__icon--danger:hover {
  border-color: var(--stop);
  color: var(--stop);
}
.sc__body {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}
.sc__thumb {
  position: relative;
  flex: 0 0 auto;
}
.sc__thumb img {
  height: 48px;
  width: 48px;
  border-radius: var(--radius-control);
  object-fit: cover;
  display: block;
}
.sc__thumb-remove {
  position: absolute;
  top: -6px;
  right: -6px;
  height: 20px;
  width: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--edge);
  background: var(--bench);
  color: var(--rebate);
  border-radius: 50%;
  cursor: pointer;
}
.sc__thumb-remove:hover {
  color: var(--stop);
  border-color: var(--stop);
}
.sc__attach {
  height: 48px;
  width: 48px;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1.5px dashed var(--ce);
  background: var(--bath);
  color: var(--ink-3);
  border-radius: var(--radius-control);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.sc__attach:hover {
  border-color: var(--safelight);
  color: var(--ink-2);
}
.sc__prompt {
  flex: 1;
  min-width: 0;
  min-height: 48px;
  resize: none;
  background: transparent;
  border: 0;
  outline: none;
  color: var(--rebate);
  font-family: var(--f-body);
  /* 16px avoids iOS focus-zoom on phone browsers. */
  font-size: 16px;
  line-height: 1.4;
}
.sc__prompt::placeholder {
  color: var(--ink-3);
}
.sc__foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}
.sc__link {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  cursor: pointer;
  padding: 0;
}
.sc__link:hover {
  color: var(--rebate);
}
.sc__dur {
  font-variant-numeric: tabular-nums;
}
.sc__anchor {
  color: var(--warning);
}
</style>
