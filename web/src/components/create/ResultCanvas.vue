<script setup lang="ts">
/*
 * Result canvas (Mold Studio Create) — the develop bed. Four states:
 *  - empty: brand headline + what-to-do guidance.
 *  - generating: the develop bed — live latent preview under the thinning
 *    Develop grain (desktop parity), with a progress ring + stage line until
 *    the first preview arrives.
 *  - result: the finished print with a mono provenance caption.
 *  - variations: batch>1 expansion review — editable prompts with per-row Use,
 *    plus Discard / Queue N.
 * Presentational only: the parent owns job state and the submit paths.
 */
import { computed } from "vue";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import ErrorNotice from "@ui/components/ErrorNotice.vue";
import ProgressRing from "@ui/components/ProgressRing.vue";
import DevelopCanvas from "@ui/components/DevelopCanvas.vue";
import type { DevelopPhase } from "@ui/lib/grain";

const props = withDefaults(
  defineProps<{
    mode: "empty" | "generating" | "result" | "error" | "variations";
    /** generating — 0–100. */
    progress?: number;
    /** generating — stage line, e.g. "Developing 12 / 28". */
    stage?: string;
    /** generating — latest live latent preview (data URI). */
    previewSrc?: string;
    /** generating — 0–1 denoise progress for the blur/grain math. */
    progressFraction?: number;
    /** generating — seed feeding the Develop grain (job `seedVisual`). */
    developSeed?: string;
    /** generating — develop phase for the grain shader. */
    developPhase?: DevelopPhase;
    /** generating — submitted print dims; the bed adopts their aspect. */
    printWidth?: number;
    printHeight?: number;
    /** result — image URL and prebuilt mono caption. */
    resultSrc?: string;
    resultCaption?: string;
    /** error — server or transport failure copy. */
    error?: string;
    /** error — optional raw diagnostic appended to clipboard copy. */
    errorCopy?: string;
    /** variations — the editable prompt list. */
    variations?: string[];
  }>(),
  { progress: 0, stage: "", progressFraction: 0, variations: () => [] },
);

// Desktop parity (GenerateView.vue): the preview's blur tightens as denoising
// progresses while the grain canvas thins to reveal the forming print.
const clampedFraction = computed(() =>
  Math.min(1, Math.max(0, props.progressFraction)),
);
const previewBlurPx = computed(() =>
  Math.max(2, 14 - 12 * clampedFraction.value),
);
const grainOpacity = computed(() =>
  props.previewSrc
    ? String(Math.max(0.18, 1 - clampedFraction.value * 0.9))
    : "1",
);
const bedStyle = computed(() =>
  props.printWidth && props.printHeight
    ? { aspectRatio: `${props.printWidth} / ${props.printHeight}` }
    : undefined,
);

const emit = defineEmits<{
  "update:variations": [value: string[]];
  "use-variation": [index: number];
  discard: [];
  queue: [];
}>();

function editVariation(index: number, value: string) {
  const next = props.variations.slice();
  next[index] = value;
  emit("update:variations", next);
}
</script>

<template>
  <div class="canvas" data-test="result-canvas" :data-mode="mode">
    <div v-if="mode === 'empty'" class="canvas__empty">
      <EmptyStateBlock
        brand
        headline="Your print develops here"
        guidance="Describe an image below, pick a look, and press Generate. Everything runs on your own machine."
      />
    </div>

    <div
      v-else-if="mode === 'generating'"
      class="canvas__generating"
      data-test="canvas-generating"
    >
      <div class="canvas__bed ms-shimmer" :style="bedStyle">
        <!-- Live latent preview: a tiny PNG upscaled by CSS; the blur tightens
             as denoising progresses so the print develops on the canvas. -->
        <img
          v-if="previewSrc"
          :src="previewSrc"
          alt=""
          class="canvas__preview"
          data-test="canvas-preview"
          :style="{ filter: `blur(${previewBlurPx}px)` }"
        />
        <!-- The grain canvas paints edge-to-edge; once previews exist it thins
             with progress to reveal the forming print underneath. -->
        <DevelopCanvas
          :seed="developSeed ?? 'mold'"
          :progress="clampedFraction"
          :phase="developPhase ?? 'developing'"
          class="canvas__develop"
          :style="{ opacity: grainOpacity }"
        />
        <!-- Ring + stage overlay the grain only until the first latent
             preview arrives; then the forming print takes over. -->
        <div v-if="!previewSrc" class="canvas__ring">
          <ProgressRing :value="progress" :size="104" show-label />
        </div>
      </div>
      <div v-if="!previewSrc" class="canvas__stage" data-test="canvas-stage">
        {{ stage }}
      </div>
    </div>

    <div
      v-else-if="mode === 'result'"
      class="canvas__result ms-fade-up"
      data-test="canvas-result"
    >
      <img class="canvas__img" :src="resultSrc" alt="Generated print" />
      <div class="canvas__caption" data-test="canvas-caption">
        {{ resultCaption }}
      </div>
    </div>

    <div
      v-else-if="mode === 'error'"
      class="canvas__error"
      data-test="canvas-error"
    >
      <div class="canvas__error-mark" aria-hidden="true">!</div>
      <div class="min-w-0 flex-1">
        <div class="canvas__error-title">Generation failed</div>
        <ErrorNotice
          class="mt-2 text-left"
          :message="
            error || 'Something went wrong while developing this print.'
          "
          :copy-message="errorCopy"
        />
      </div>
    </div>

    <div v-else class="canvas__variations" data-test="canvas-variations">
      <div class="canvas__var-head">
        <span class="canvas__var-title"
          >{{ variations.length }} variations ready</span
        >
        <span class="canvas__var-sub">Edit any before queueing</span>
      </div>
      <div class="canvas__var-list">
        <div
          v-for="(text, index) in variations"
          :key="index"
          class="canvas__var-row"
          :data-test="`variation-row-${index}`"
        >
          <span class="canvas__var-n">{{ index + 1 }}</span>
          <textarea
            class="canvas__var-input"
            :data-test="`variation-input-${index}`"
            :value="text"
            @input="
              editVariation(index, ($event.target as HTMLTextAreaElement).value)
            "
          />
          <button
            type="button"
            class="canvas__var-use"
            :data-test="`variation-use-${index}`"
            @click="emit('use-variation', index)"
          >
            Use
          </button>
        </div>
      </div>
      <div class="canvas__var-actions">
        <button
          type="button"
          class="canvas__discard"
          data-test="variations-discard"
          @click="emit('discard')"
        >
          Discard
        </button>
        <button
          type="button"
          class="canvas__queue"
          data-test="variations-queue"
          @click="emit('queue')"
        >
          Queue {{ variations.length }} prints
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.canvas {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.canvas__empty {
  max-width: 360px;
}

.canvas__generating,
.canvas__result {
  text-align: center;
}

.canvas__bed {
  position: relative;
  overflow: hidden;
  width: min(300px, 60vw);
  aspect-ratio: 1;
  border-radius: var(--radius-control-lg);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto;
}

.canvas__preview {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.canvas__develop {
  position: absolute;
  inset: 0;
}

.canvas__ring {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
}

.canvas__stage {
  margin-top: 14px;
  font-family: var(--f-mono);
  font-size: 12px;
  color: var(--safelight);
}

.canvas__img {
  max-width: 420px;
  max-height: 440px;
  width: 100%;
  border-radius: 8px;
  border: 1px solid var(--edge);
  box-shadow: 0 20px 50px -12px rgba(0, 0, 0, 0.6);
  display: block;
}

.canvas__caption {
  margin-top: 12px;
  font-family: var(--f-mono);
  font-size: 10.5px;
  color: var(--ink-3);
}

.canvas__error {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  max-width: 520px;
  color: var(--stop);
}

.canvas__error-mark {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  border: 1px solid currentColor;
  border-radius: 50%;
  font-family: var(--f-mono);
  font-weight: 700;
}

.canvas__error-title {
  font-family: var(--f-display);
  font-weight: 600;
}

.canvas__error-copy {
  margin-top: 4px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-2);
  overflow-wrap: anywhere;
}

.canvas__variations {
  width: 100%;
  max-width: 600px;
}

.canvas__var-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 14px;
}

.canvas__var-title {
  font-family: var(--f-display);
  font-size: 18px;
  font-weight: 600;
}

.canvas__var-sub {
  font-size: 12px;
  color: var(--ink-3);
}

.canvas__var-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.canvas__var-row {
  display: flex;
  gap: 11px;
  align-items: flex-start;
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: 11px;
  padding: 12px 14px;
}

.canvas__var-n {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--safelight);
  margin-top: 4px;
}

.canvas__var-input {
  flex: 1;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 13.5px;
  line-height: 1.45;
  min-height: 42px;
  resize: none;
  outline: none;
}

.canvas__var-use {
  flex: 0 0 auto;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 7px 13px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.canvas__var-actions {
  display: flex;
  gap: 10px;
  margin-top: 16px;
  justify-content: flex-end;
}

.canvas__discard {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 10px 18px;
  border-radius: var(--radius-control-lg);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.canvas__queue {
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 10px 20px;
  border-radius: var(--radius-control-lg);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}
</style>
