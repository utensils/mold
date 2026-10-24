<script setup lang="ts">
import { computed } from "vue";
import VideoDurationInput from "./VideoDurationInput.vue";
import type { VideoJobPlan } from "../lib/api/videoJobTypes";
import type {
  VideoDurationAuthoring,
  VideoDurationScene,
} from "../lib/videoDurationAuthoring";
import {
  formatRequestedVideoDuration,
  oneClipSceneSeconds,
  singleGenerationSeconds,
} from "../lib/videoRequestedDuration";
const props = defineProps<{
  modelValue: VideoDurationAuthoring;
  prompt: string;
  plan?: VideoJobPlan | null;
  checking?: boolean;
  error?: string | null;
  disabled?: boolean;
  sourceTitle?: string | null | undefined;
  sourceMachine?: string | null | undefined;
}>();
const emit = defineEmits<{
  (event: "update:modelValue", value: VideoDurationAuthoring): void;
  (event: "singlePrompt", value: string): void;
  (event: "startFresh"): void;
}>();
const scenesActive = computed(
  () => props.modelValue.scenesEnabled && props.modelValue.scenes.length > 0,
);
function updateScenes(scenes: VideoDurationScene[]) {
  const valid = scenes.every(
    (scene) => scene.durationMs && /^[1-9]\d*$/.test(scene.durationMs),
  );
  const total = valid
    ? scenes.reduce((sum, scene) => sum + BigInt(scene.durationMs!), 0n)
    : null;
  emit("update:modelValue", {
    ...props.modelValue,
    scenes,
    durationMs:
      total !== null && total <= 18_446_744_073_709_551_615n
        ? String(total)
        : null,
  });
}
function patch(id: string, fields: Partial<VideoDurationScene>) {
  updateScenes(
    props.modelValue.scenes.map((scene) =>
      scene.id === id ? { ...scene, ...fields } : scene,
    ),
  );
}
function move(index: number, delta: number) {
  const scenes = [...props.modelValue.scenes],
    destination = index + delta;
  if (destination < 0 || destination >= scenes.length) return;
  [scenes[index], scenes[destination]] = [scenes[destination]!, scenes[index]!];
  updateScenes(scenes);
}
function defineScenes() {
  emit("update:modelValue", {
    ...props.modelValue,
    scenesEnabled: true,
    scenes: props.modelValue.scenes.length
      ? props.modelValue.scenes
      : [
          {
            id: crypto.randomUUID(),
            prompt: props.prompt,
            durationMs: props.modelValue.durationMs,
          },
        ],
  });
}
function addScene() {
  updateScenes([
    ...props.modelValue.scenes,
    { id: crypto.randomUUID(), prompt: "", durationMs: "5000" },
  ]);
}
function onePrompt() {
  emit("singlePrompt", props.modelValue.scenes[0]?.prompt ?? props.prompt);
  emit("update:modelValue", { ...props.modelValue, scenesEnabled: false });
}
</script>
<template>
  <section
    class="duration-editor"
    aria-label="Video duration"
    :aria-busy="checking"
  >
    <div v-if="sourceTitle" class="duration-editor__source">
      <strong>Extending {{ sourceTitle }}</strong>
      <p v-if="sourceMachine">On {{ sourceMachine }}</p>
      <p>The original and new footage save together as a new video.</p>
      <button type="button" :disabled="disabled" @click="emit('startFresh')">
        Start new video
      </button>
      <div
        v-if="modelValue.extension?.references.length"
        aria-label="Original references"
      >
        <p>Original references are retained for this extension:</p>
        <p
          v-for="(reference, index) in modelValue.extension.references"
          :key="index"
        >
          {{ index + 1 }}. {{ reference.name }} · {{ reference.role }}
        </p>
      </div>
      <slot name="source" />
    </div>
    <p v-if="modelValue.reuseNotice" role="status">
      {{ modelValue.reuseNotice }}
    </p>
    <template v-if="scenesActive">
      <p>
        {{ sourceTitle ? "Add duration" : "Duration" }}:
        {{
          modelValue.durationMs
            ? formatRequestedVideoDuration(modelValue.durationMs)
            : "Check scene times"
        }}
        seconds
      </p>
      <fieldset
        v-for="(scene, index) in modelValue.scenes"
        :key="scene.id"
        :disabled="disabled"
        class="duration-editor__scene"
      >
        <legend>Scene {{ index + 1 }}</legend>
        <label
          >Prompt<textarea
            :value="scene.prompt"
            :data-test="`video-scene-${scene.id}-prompt`"
            rows="2"
            @input="
              patch(scene.id, {
                prompt: ($event.target as HTMLTextAreaElement).value,
              })
            "
          />
        </label>
        <VideoDurationInput
          :model-value="scene.durationMs"
          :label="`Scene ${index + 1} duration`"
          :presets="false"
          @update:model-value="patch(scene.id, { durationMs: $event })"
        />
        <div class="duration-editor__actions">
          <button
            type="button"
            :disabled="index === 0"
            :data-test="`video-scene-${scene.id}-up`"
            :aria-label="`Move scene ${index + 1} up`"
            @click="move(index, -1)"
          >
            Move up
          </button>
          <button
            type="button"
            :disabled="index === modelValue.scenes.length - 1"
            :aria-label="`Move scene ${index + 1} down`"
            @click="move(index, 1)"
          >
            Move down
          </button>
          <button
            type="button"
            :disabled="modelValue.scenes.length === 1"
            :aria-label="`Remove scene ${index + 1}`"
            @click="
              updateScenes(
                modelValue.scenes.filter((row) => row.id !== scene.id),
              )
            "
          >
            Remove
          </button>
        </div>
      </fieldset>
      <div class="duration-editor__actions">
        <button
          type="button"
          :disabled="disabled || modelValue.scenes.length >= 4096"
          @click="addScene"
        >
          Add scene
        </button>
        <button
          type="button"
          :disabled="disabled"
          data-test="video-one-prompt"
          @click="onePrompt"
        >
          Use one prompt
        </button>
      </div>
      <p>
        References apply to every scene. Opening and closing frames bound the
        whole video.
      </p>
    </template>
    <VideoDurationInput
      v-else
      :model-value="modelValue.durationMs"
      :label="sourceTitle ? 'Add duration' : 'Duration'"
      :plan="plan ?? null"
      :checking="checking"
      :disabled="disabled"
      @update:model-value="
        emit('update:modelValue', { ...modelValue, durationMs: $event })
      "
    />
    <button
      v-if="!scenesActive"
      type="button"
      :disabled="disabled"
      data-test="define-video-scenes"
      @click="defineScenes"
    >
      Define scenes
    </button>
    <template v-if="plan">
      <template v-if="scenesActive">
        <p v-if="singleGenerationSeconds(plan)">
          Single generation: up to {{ singleGenerationSeconds(plan) }} seconds
        </p>
        <p v-if="oneClipSceneSeconds(plan)" data-test="video-scene-limit">
          Scenes up to {{ oneClipSceneSeconds(plan) }} s each stay one clip
        </p>
        <p>
          {{ plan.clip_count }}
          {{ plan.clip_count === "1" ? "clip" : "clips joined into one video" }}
        </p>
      </template>
      <p v-if="plan.join_mode === 'blended_shots' && plan.clip_count !== '1'">
        Shots blend together; motion does not carry between them.
      </p>
      <p>
        {{ sourceTitle ? "New total" : "Video" }}:
        {{ formatRequestedVideoDuration(plan.total_duration_ms) }} seconds
      </p>
      <p
        v-for="(warning, index) in plan.warnings"
        :key="index"
        class="duration-editor__warning"
        data-test="video-plan-warning"
        role="note"
      >
        {{ warning }}
      </p>
    </template>
    <p v-if="error" role="alert" class="duration-editor__error">{{ error }}</p>
  </section>
</template>
<style scoped>
.duration-editor,
.duration-editor__source,
.duration-editor__scene,
.duration-editor label {
  display: grid;
  gap: var(--mold-space-2, 8px);
}
.duration-editor {
  padding: var(--mold-space-3, 12px);
}
.duration-editor__scene {
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  padding: var(--mold-space-3, 12px);
  min-width: 0;
}
.duration-editor textarea {
  width: 100%;
  resize: vertical;
  min-height: 44px;
  background: var(--mold-bg);
  color: var(--mold-text);
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  padding: var(--mold-space-2, 8px);
  font: inherit;
  font-size: max(
    16px,
    var(--mold-fs-md)
  ); /* literal: prevent mobile input zoom */
}
.duration-editor button {
  min-height: 44px;
  padding: var(--mold-space-2, 8px);
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg);
  color: var(--mold-text);
  font: inherit;
}
.duration-editor__actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--mold-space-2, 8px);
}
.duration-editor p {
  margin: 0;
  color: var(--mold-text-dim);
}
/* A plan warning ("…its action will repeat…") is advice the person must
 * read before rendering: full text, wrapped, never clamped or dimmed. */
.duration-editor .duration-editor__warning {
  color: var(--mold-warning, var(--mold-text));
  overflow-wrap: anywhere;
  white-space: normal;
}
.duration-editor .duration-editor__error {
  color: var(--mold-error);
}
</style>
