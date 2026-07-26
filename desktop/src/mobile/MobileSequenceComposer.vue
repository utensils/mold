<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue";
import {
  DEFAULT_SEQUENCE_CLIP_FRAMES,
  defaultSequenceStages,
  friendlySequenceError,
  sequenceFrameOptions,
  sequenceDuration,
  sequenceMotionTailFrames,
  sequenceValidation,
  transitionLabel,
  type SequenceTransition,
} from "@studio/lib/sequence";
import { apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { ChainJobDetail, ChainLimits, ChainRequest, ModelEntry } from "../lib/api/types";
import { modelDisplayName } from "../lib/models";

const props = defineProps<{
  models: ModelEntry[];
  target: ApiTarget;
  hostName: string;
  job: ChainJobDetail | null;
  starting: boolean;
  error: string;
}>();

const emit = defineEmits<{
  submit: [request: ChainRequest];
  cancel: [];
  dismiss: [];
  browse: [];
}>();

const form = reactive({
  model: "",
  stages: defaultSequenceStages(),
  width: 768,
  height: 512,
  fps: 24,
  steps: 8,
  guidance: 3,
  motionTailFrames: 17,
  seed: "",
  enableAudio: false,
});
const limits = ref<ChainLimits | null>(null);
const limitsError = ref("");

const selectedModel = computed(
  () => props.models.find((model) => model.name === form.model) ?? null,
);
const duration = computed(() => sequenceDuration(form.stages, form.fps, form.motionTailFrames));
const frameOptions = computed(() =>
  sequenceFrameOptions(limits.value?.frames_per_clip_cap ?? 97, form.motionTailFrames),
);
const validation = computed(() =>
  sequenceValidation(form.stages, {
    maxStages: limits.value?.max_stages ?? 16,
    maxTotalFrames: limits.value?.max_total_frames ?? 777,
    motionTailFrames: form.motionTailFrames,
  }),
);
const active = computed(
  () => props.starting || props.job?.state === "queued" || props.job?.state === "running",
);
const settled = computed(
  () =>
    props.job?.state === "completed" ||
    props.job?.state === "failed" ||
    props.job?.state === "cancelled",
);

function applyModel(model: ModelEntry) {
  form.model = model.name;
  form.motionTailFrames = sequenceMotionTailFrames(model);
  form.width = model.default_width;
  form.height = model.default_height;
  form.steps = model.default_steps;
  form.guidance = model.default_guidance;
}

async function loadLimits(modelName: string) {
  limitsError.value = "";
  try {
    const next = await apiJsonTo<ChainLimits>(
      props.target,
      `/api/capabilities/chain-limits?model=${encodeURIComponent(modelName)}`,
    );
    if (form.model !== modelName) return;
    limits.value = next;
    form.motionTailFrames = Math.min(form.motionTailFrames, next.frames_per_clip_recommended);
    if (!next.supports_audio) form.enableAudio = false;
    for (const stage of form.stages) {
      stage.frames = Math.min(stage.frames, next.frames_per_clip_recommended);
    }
  } catch (error) {
    limits.value = null;
    limitsError.value = error instanceof Error ? error.message : "Couldn't load sequence limits.";
  }
}

function addClip() {
  const recommended = Math.min(
    DEFAULT_SEQUENCE_CLIP_FRAMES,
    limits.value?.frames_per_clip_recommended ?? DEFAULT_SEQUENCE_CLIP_FRAMES,
  );
  form.stages.push({
    prompt: "",
    frames: recommended,
    transition: "smooth",
  });
}

function removeClip(index: number) {
  if (form.stages.length <= 2) return;
  form.stages.splice(index, 1);
}

function updateTransition(index: number, transition: SequenceTransition) {
  const stage = form.stages[index];
  if (!stage) return;
  stage.transition = transition;
  if (transition === "fade" && !stage.fade_frames) stage.fade_frames = 8;
}

function submit() {
  if (active.value || validation.value.length || !selectedModel.value) return;
  const seed = form.seed.trim() ? Number(form.seed) : null;
  emit("submit", {
    model: form.model,
    stages: form.stages.map((stage, index) => ({
      prompt: stage.prompt.trim(),
      frames: stage.frames,
      transition: index === 0 ? "smooth" : stage.transition,
      ...(stage.transition === "fade" ? { fade_frames: stage.fade_frames ?? 8 } : {}),
    })),
    motion_tail_frames: form.motionTailFrames,
    width: form.width,
    height: form.height,
    fps: form.fps,
    ...(seed !== null && Number.isFinite(seed) ? { seed } : {}),
    steps: form.steps,
    guidance: form.guidance,
    strength: 1,
    output_format: "mp4",
    ...(form.enableAudio ? { enable_audio: true } : {}),
  });
}

watch(
  () => props.models.map((model) => model.name).join("|"),
  () => {
    if (!form.model || !props.models.some((model) => model.name === form.model)) {
      const first = props.models[0];
      if (first) applyModel(first);
    }
  },
  { immediate: true },
);

watch(
  () => form.model,
  (modelName) => {
    if (modelName) void loadLimits(modelName);
  },
  { immediate: true },
);
</script>

<template>
  <section class="mobile-sequence" data-test="mobile-sequence-composer">
    <div v-if="models.length === 0" class="mobile-sequence-empty">
      <strong>Sequences need a video model</strong>
      <p>
        Pull a chain-capable LTX Video or distilled LTX-2 checkpoint on
        {{ hostName }}.
      </p>
      <button class="secondary-button" type="button" @click="emit('browse')">
        Browse video models
      </button>
    </div>
    <template v-else>
      <label class="field">
        <span>Video model</span>
        <select
          v-model="form.model"
          class="control"
          :disabled="active"
          data-test="mobile-sequence-model"
        >
          <option v-for="model in models" :key="model.name" :value="model.name">
            {{ modelDisplayName(model) }}
          </option>
        </select>
      </label>

      <div class="mobile-sequence-clips">
        <article
          v-for="(stage, index) in form.stages"
          :key="index"
          class="mobile-sequence-clip"
          data-test="mobile-sequence-clip"
        >
          <div class="mobile-sequence-clip-head">
            <strong>{{ index === 0 ? "Opening clip" : `Clip ${index + 1}` }}</strong>
            <button
              v-if="form.stages.length > 2"
              type="button"
              :aria-label="`Remove clip ${index + 1}`"
              :disabled="active"
              @click="removeClip(index)"
            >
              Remove
            </button>
          </div>
          <div
            v-if="index > 0"
            class="mobile-sequence-transitions"
            role="radiogroup"
            :aria-label="`Transition into clip ${index + 1}`"
          >
            <button
              v-for="transition in ['smooth', 'cut', 'fade'] as SequenceTransition[]"
              :key="transition"
              type="button"
              :class="{ active: stage.transition === transition }"
              :aria-pressed="stage.transition === transition"
              :disabled="active"
              @click="updateTransition(index, transition)"
            >
              {{ transitionLabel(transition, form.motionTailFrames) }}
            </button>
          </div>
          <textarea
            v-model="stage.prompt"
            class="control mobile-sequence-prompt"
            :aria-label="index === 0 ? 'Opening clip prompt' : `Clip ${index + 1} prompt`"
            :placeholder="index === 0 ? 'How does the sequence begin?' : 'What happens next?'"
            :disabled="active"
          />
          <details class="mobile-native-disclosure">
            <summary>
              <span>Clip tools</span>
              <small>{{ stage.frames }} frames</small>
            </summary>
            <label class="field">
              <span>Duration</span>
              <select v-model.number="stage.frames" class="control" :disabled="active">
                <option v-for="frames in frameOptions" :key="frames" :value="frames">
                  {{ frames }} frames · {{ (frames / form.fps).toFixed(1) }}s
                </option>
              </select>
            </label>
          </details>
        </article>
      </div>

      <button
        class="secondary-button mobile-sequence-add"
        type="button"
        :disabled="active || form.stages.length >= (limits?.max_stages ?? 16)"
        @click="addClip"
      >
        + Add clip
      </button>

      <details class="mobile-native-disclosure mobile-sequence-tools">
        <summary>
          <span>Sequence tools</span>
          <small>{{ duration.seconds.toFixed(1) }}s · {{ duration.frames }} frames</small>
        </summary>
        <div class="mobile-sequence-tool-grid">
          <label class="field">
            <span>Width</span>
            <input v-model.number="form.width" class="control" type="number" inputmode="numeric" />
          </label>
          <label class="field">
            <span>Height</span>
            <input v-model.number="form.height" class="control" type="number" inputmode="numeric" />
          </label>
          <label class="field">
            <span>FPS</span>
            <input v-model.number="form.fps" class="control" type="number" inputmode="numeric" />
          </label>
          <label class="field">
            <span>Seed</span>
            <input v-model="form.seed" class="control" inputmode="numeric" placeholder="Random" />
          </label>
        </div>
        <label v-if="limits?.supports_audio" class="mobile-sequence-check">
          <input v-model="form.enableAudio" type="checkbox" />
          Generate audio
        </label>
      </details>

      <p v-if="limitsError" class="mobile-sequence-error" role="alert">
        {{ limitsError }}
      </p>
      <p v-else-if="validation[0]" class="mobile-sequence-error" role="alert">
        {{ validation[0] }}
      </p>
      <p v-if="error" class="mobile-sequence-error" role="alert">{{ error }}</p>

      <button
        class="primary-button"
        type="button"
        data-test="mobile-generate-sequence"
        :disabled="active || validation.length > 0 || !selectedModel"
        @click="submit"
      >
        {{ starting ? "Starting…" : "Generate sequence" }}
      </button>

      <article v-if="job" class="mobile-sequence-job" data-test="mobile-sequence-job">
        <div>
          <strong>{{ job.state }}</strong>
          <span>{{ job.current_stage }}/{{ job.stage_count }} clips</span>
        </div>
        <p v-if="job.error" class="mobile-sequence-error">
          {{ friendlySequenceError(job.error) }}
        </p>
        <button
          v-if="job.state === 'queued' || job.state === 'running'"
          class="secondary-button"
          type="button"
          @click="emit('cancel')"
        >
          Cancel sequence
        </button>
        <button v-else-if="settled" class="secondary-button" type="button" @click="emit('dismiss')">
          Dismiss
        </button>
      </article>
    </template>
  </section>
</template>

<style scoped>
.mobile-sequence {
  display: grid;
  gap: 14px;
}

.mobile-sequence-empty,
.mobile-sequence-clip,
.mobile-sequence-job {
  display: grid;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--edge);
  border-radius: 16px;
  background: var(--bench);
}

.mobile-sequence-empty p,
.mobile-sequence-job p {
  margin: 0;
  color: var(--ink-2);
}

.mobile-sequence-clips {
  display: grid;
  gap: 12px;
}

.mobile-sequence-clip-head,
.mobile-sequence-job > div {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.mobile-sequence-clip-head button {
  min-height: 44px;
  color: var(--stop);
}

.mobile-sequence-prompt {
  min-height: 92px;
  resize: vertical;
  font-size: 16px;
}

.mobile-sequence-transitions {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  padding: 3px;
  border: 1px solid var(--edge);
  border-radius: 12px;
  background: var(--bath);
}

.mobile-sequence-transitions button {
  min-height: 44px;
  padding: 6px;
  border-radius: 9px;
  color: var(--ink-2);
  font-size: 12px;
}

.mobile-sequence-transitions button.active {
  color: var(--on-accent);
  background: var(--safelight);
}

.mobile-sequence-add {
  width: 100%;
  min-height: 44px;
}

.mobile-sequence-tool-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.mobile-sequence-check {
  display: flex;
  min-height: 44px;
  align-items: center;
  gap: 10px;
}

.mobile-sequence-error {
  margin: 0;
  color: var(--stop);
  font-size: 14px;
}
</style>
