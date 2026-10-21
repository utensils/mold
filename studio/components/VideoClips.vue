<script setup lang="ts">
import ModalPanel from "@ui/components/ModalPanel.vue";
import {
  useVideoClips,
  type VideoClipsOptions,
  type VideoClipsEvents,
} from "../composables/useVideoClips";
const props = defineProps<VideoClipsOptions>();
const emit = defineEmits<VideoClipsEvents>();
const {
  active,
  loading,
  busy,
  error,
  captured,
  clips,
  limits,
  plan,
  admission,
  detail,
  settled,
  cancelBusy,
  editing,
  restoredDraft,
  restoredSettings,
  minFrames,
  step,
  maxFrames,
  lastFilename,
  ownerChanged,
  refusal,
  status,
  inputSummary,
  open,
  close,
  edited,
  add,
  move,
  setFrames,
  validate,
  submit,
  refresh,
  action,
  cancelKnownOperation,
} = useVideoClips(props, emit);
defineExpose({ submit });
</script>

<template>
  <button
    v-if="!active && (enabled || source) && owner"
    type="button"
    class="video-clips-door"
    data-test="video-clips-open"
    @click="open()"
  >
    {{ source ? "Continue video" : "Add clip" }}
  </button>
  <component
    :is="source ? ModalPanel : 'div'"
    v-if="active"
    :open="active"
    label="Continue video"
    :width="560"
    @close="close"
  >
    <section
      class="video-clips"
      aria-label="Video clips"
      data-test="video-clips"
    >
      <header>
        <strong>{{ source ? "Continue video" : "Video clips" }}</strong
        ><button type="button" @click="close">
          {{ admission ? "Done" : source ? "Cancel" : "Back to one clip" }}
        </button>
      </header>
      <p v-if="captured?.request.model" class="video-clips-note">
        {{ captured.owner.label }} · {{ captured.request.width }}×{{
          captured.request.height
        }}
        · {{ captured.request.fps ?? 24 }} fps
      </p>
      <p v-if="source" class="video-clips-note">
        Creates the next separate video part from the latest completed take. New
        random seed.
      </p>
      <p v-if="restoredSettings && !admission" class="video-clips-note">
        {{ restoredSettings.notice }}
      </p>
      <p v-if="restoredDraft && !admission" class="video-clips-note">
        Restored your clips. Review them before making the video.
      </p>
      <p v-if="inputSummary && !admission" class="video-clips-note">
        {{ inputSummary }}
      </p>
      <p v-if="loading" role="status">Checking this machine…</p>
      <template v-else-if="limits && !admission">
        <p
          v-if="limits.authoring?.carry_mode === 'independent'"
          class="video-clips-note"
        >
          These clips join without carrying motion between them.
        </p>
        <div
          v-for="(clip, index) in clips"
          :key="clip.id"
          class="video-clips-row"
        >
          <p v-if="clip.restored" class="video-clips-note">
            {{ clip.restored.transition }} join<span
              v-if="clip.restored.fade_frames"
            >
              · {{ clip.restored.fade_frames }} fade frames</span
            >
            · seed
            {{
              String(
                BigInt(
                  String(
                    (captured?.request as { seed?: number | string })?.seed ??
                      0,
                  ),
                ) ^ BigInt(String(clip.restored.seed_offset ?? 0)),
              )
            }}
            · {{ clip.restored.loras?.length ?? 0 }} adapters ·
            {{ clip.restored.generation_references?.length ?? 0 }} references ·
            {{ clip.restored.keyframes?.length ?? 0 }} keyframes<span
              v-if="clip.restored.source_image"
            >
              · first frame</span
            >
          </p>
          <div class="video-clips-row-header">
            <strong>Clip {{ index + 1 }}</strong
            ><span />
            <button
              type="button"
              :disabled="!editing || index === 0"
              :aria-label="`Move clip ${index + 1} up`"
              @click="move(clip.id, -1)"
            >
              ↑
            </button>
            <button
              type="button"
              :disabled="!editing || index === clips.length - 1"
              :aria-label="`Move clip ${index + 1} down`"
              @click="move(clip.id, 1)"
            >
              ↓
            </button>
            <button
              type="button"
              :disabled="!editing || clips.length === 1"
              :aria-label="`Remove clip ${index + 1}`"
              @click="
                clips.splice(index, 1);
                edited();
              "
            >
              Remove
            </button>
          </div>
          <textarea
            v-model="clip.prompt"
            :aria-label="`Clip ${index + 1} prompt`"
            :disabled="!editing"
            placeholder="Describe what happens in this clip"
            @input="edited"
          />
          <label class="video-clips-length"
            >Length
            <input
              type="range"
              :aria-label="`Clip ${index + 1} length`"
              :min="minFrames"
              :max="maxFrames"
              :step="step"
              :value="clip.frames"
              :disabled="!editing"
              @input="
                setFrames(clip, ($event.target as HTMLInputElement).value)
              "
            />
            <output
              >{{
                (clip.frames / (captured?.request.fps ?? 24)).toFixed(1)
              }}s</output
            ></label
          >
        </div>
        <button
          type="button"
          :disabled="!editing || clips.length >= limits.max_stages"
          @click="add"
        >
          Add clip
        </button>
        <p v-if="refusal" role="alert">{{ refusal }}</p>
        <p v-if="plan" role="status">
          {{ (plan.estimated_duration_ms / 1000).toFixed(2) }} seconds delivered
          · {{ plan.estimated_total_frames }} frames
        </p>
        <p v-for="warning in plan?.warnings ?? []" :key="warning">
          {{ warning }}
        </p>
        <footer>
          <button
            type="button"
            :disabled="busy || !!refusal || !!ownerChanged"
            @click="validate"
          >
            Check length</button
          ><button
            type="button"
            class="video-clips-make"
            data-test="video-clips-submit"
            :disabled="
              busy ||
              !!refusal ||
              !!ownerChanged ||
              clips.some((clip) => !clip.prompt.trim())
            "
            @click="submit"
          >
            {{ busy ? "Checking video…" : "Make video" }}
          </button>
        </footer>
      </template>
      <div v-if="admission" role="status">
        <p>{{ status }}</p>
        <p v-if="lastFilename">{{ lastFilename }}</p>
        <button
          v-if="!settled"
          type="button"
          :disabled="busy || !!ownerChanged"
          @click="refresh"
        >
          Reconnect
        </button>
        <button
          v-if="
            detail &&
            ['paused', 'interrupted', 'failed', 'cancelled'].includes(
              detail.state,
            )
          "
          type="button"
          :disabled="busy || !!ownerChanged"
          @click="action('resume')"
        >
          Resume video
        </button>
        <button
          v-if="
            !settled &&
            (!detail || ['queued', 'running', 'paused'].includes(detail.state))
          "
          type="button"
          :disabled="
            cancelBusy || !!ownerChanged || admission.cancelAcknowledged
          "
          @click="cancelKnownOperation"
        >
          Cancel video
        </button>
        <button
          v-if="lastFilename && captured"
          type="button"
          @click="
            emit('published', { filename: lastFilename, owner: captured.owner })
          "
        >
          View in Library
        </button>
      </div>
      <p v-if="error" role="alert" class="video-clips-error">{{ error }}</p>
    </section>
  </component>
</template>

<style scoped>
.video-clips {
  color: var(--mold-text, inherit);
  background: var(--mold-surface, transparent);
  border: 1px solid var(--mold-border, currentColor);
  border-radius: var(--mold-radius-2, 0.5rem);
  padding: 1rem;
  overflow: auto;
  max-height: 70vh;
}
.video-clips header,
.video-clips-row-header,
.video-clips-length,
.video-clips footer {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.video-clips header {
  justify-content: space-between;
}
.video-clips-row {
  margin-block: 0.75rem;
}
.video-clips-row-header span {
  flex: 1;
}
.video-clips textarea {
  display: block;
  width: 100%;
  min-height: 5rem;
  font: inherit;
  font-size: max(1rem, var(--mold-fs-body, 1rem));
  padding: 0.5rem;
  color: inherit;
  background: var(--mold-bg, transparent);
  border: 1px solid var(--mold-border, currentColor);
  border-radius: var(--mold-radius-2, 0.5rem);
}
.video-clips button,
.video-clips-door {
  min-height: 2.75rem;
  padding: 0.25rem 0.75rem;
  font: inherit;
  color: inherit;
  border: 1px solid var(--mold-border, currentColor);
  border-radius: var(--mold-radius-2, 0.5rem);
  background: var(--mold-surface, transparent);
  cursor: pointer;
}
.video-clips button:disabled {
  opacity: 0.5;
  cursor: default;
}
.video-clips-length input {
  min-width: 0;
  flex: 1;
  min-height: 2.75rem;
}
.video-clips-note {
  opacity: 0.75;
}
.video-clips footer {
  justify-content: flex-end;
  margin-top: 0.75rem;
}
.video-clips-error {
  color: var(--mold-error, #b33);
}
</style>
