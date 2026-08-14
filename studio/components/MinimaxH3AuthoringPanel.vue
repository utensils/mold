<script setup lang="ts">
import { computed, ref } from "vue";
import {
  MINIMAX_H3_MAX_REFERENCE_AUDIOS,
  MINIMAX_H3_MAX_REFERENCE_IMAGES,
  MINIMAX_H3_MAX_REFERENCE_VIDEOS,
  MINIMAX_H3_MAX_REFERENCES,
  MINIMAX_H3_RESYNTHESIS_GUIDANCE,
  MINIMAX_H3_RESYNTHESIS_TITLE,
  cloneMinimaxH3AuthoringState,
  minimaxH3Mode,
  minimaxH3ReferenceBudget,
  minimaxH3ReferenceDurationMs,
  minimaxH3ReferenceName,
  minimaxH3ReferenceNeedsMedia,
  moveMinimaxH3Reference,
  type MinimaxH3AuthoringState,
  type MinimaxH3BoundaryImage,
  type MinimaxH3ReferenceDraft,
  type MinimaxH3Task,
} from "../lib/minimaxH3Authoring";
import {
  probeMinimaxH3Mp4,
  probeMinimaxH3Wav,
} from "../lib/minimaxH3MediaProbe";
import ImageDropWell from "./ImageDropWell.vue";

const props = withDefaults(
  defineProps<{
    modelValue: MinimaxH3AuthoringState;
    task: MinimaxH3Task;
    requiredEndpoint?: "first" | null;
    disabled?: boolean;
    touchFriendly?: boolean;
  }>(),
  {
    requiredEndpoint: null,
    disabled: false,
    touchFriendly: false,
  },
);

const emit = defineEmits<{
  "update:modelValue": [state: MinimaxH3AuthoringState];
  "request-first-frame": [];
  "request-last-frame": [];
}>();

const error = ref("");
const busy = ref(false);
const budget = computed(() =>
  minimaxH3ReferenceBudget(props.modelValue.references),
);
const mode = computed(() =>
  props.requiredEndpoint === "first"
    ? props.modelValue.firstFrame
      ? "first-frame-to-audio-video"
      : "first-frame-required"
    : minimaxH3Mode(props.task, props.modelValue),
);

function patch(next: Partial<MinimaxH3AuthoringState>): void {
  emit("update:modelValue", {
    ...cloneMinimaxH3AuthoringState(props.modelValue),
    ...next,
  });
}

function base64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(reader.error ?? new Error("Could not read media."));
    reader.onload = () => {
      const result = String(reader.result ?? "");
      const comma = result.indexOf(",");
      if (comma < 0)
        reject(new Error("The selected media could not be encoded."));
      else resolve(result.slice(comma + 1));
    };
    reader.readAsDataURL(file);
  });
}

async function digestBytes(bytes: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Secure media hashing is unavailable on this device.");
  }
  const value = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(value)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function digest(file: File): Promise<string> {
  return digestBytes(await file.arrayBuffer());
}

async function imageDimensions(
  file: File,
): Promise<{ width: number; height: number }> {
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(file);
    const result = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return result;
  }
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not decode the selected image."));
    };
    image.src = url;
  });
}

async function boundaryImage(file: File): Promise<MinimaxH3BoundaryImage> {
  if (!file.type.toLowerCase().startsWith("image/")) {
    throw new Error("FL2VA endpoints must be still images.");
  }
  const [data, sha256, dimensions] = await Promise.all([
    base64(file),
    digest(file),
    imageDimensions(file),
  ]);
  return {
    filename: file.name,
    mimeType: file.type,
    data,
    sha256,
    ...dimensions,
  };
}

async function referenceDraft(file: File): Promise<MinimaxH3ReferenceDraft> {
  const mime = file.type.toLowerCase();
  if (mime.startsWith("image/")) {
    const [data, sha256] = await Promise.all([base64(file), digest(file)]);
    const dimensions = await imageDimensions(file);
    return {
      reference: {
        kind: "image",
        media: { authority: "inline", data },
        provenance: { name: file.name, sha256 },
        mime_type: file.type,
        ...dimensions,
      },
    };
  }
  const isMp4 = mime === "video/mp4" || /\.mp4$/i.test(file.name);
  const isWav =
    ["audio/wav", "audio/x-wav", "audio/wave"].includes(mime) ||
    /\.wav$/i.test(file.name);
  if (!isMp4 && !isWav) {
    throw new Error(`${file.name}: choose an image, video, or audio file.`);
  }
  const bytes = await file.arrayBuffer();
  try {
    const facts = isMp4 ? probeMinimaxH3Mp4(bytes) : probeMinimaxH3Wav(bytes);
    if (
      facts.mimeType === "audio/wav" &&
      (facts.channels < 1 || facts.channels > 2)
    ) {
      throw new Error("H3 accepts mono or stereo audio.");
    }
    const [data, sha256] = await Promise.all([
      base64(file),
      digestBytes(bytes),
    ]);
    const media = { authority: "inline" as const, data };
    const provenance = { name: file.name, sha256 };
    if (facts.mimeType === "video/mp4") {
      return {
        reference: {
          kind: "video",
          media,
          provenance,
          mime_type: facts.mimeType,
          width: facts.width,
          height: facts.height,
          frame_count: facts.frameCount,
          duration_ms: facts.durationMs,
          fps: facts.fps,
          has_audio: facts.hasAudio,
          audio_duration_ms: facts.audioDurationMs,
          audio_sample_count: facts.audioSampleCount,
          audio_sample_rate: facts.audioSampleRate,
          audio_channels: facts.audioChannels,
        },
      };
    }
    return {
      reference: {
        kind: "audio",
        media,
        provenance,
        mime_type: facts.mimeType,
        duration_ms: facts.durationMs,
        sample_rate: facts.sampleRate,
        channels: facts.channels,
        sample_count: facts.sampleCount,
      },
    };
  } catch (reason) {
    const message = reason instanceof Error ? reason.message : String(reason);
    throw new Error(`${file.name}: ${message}`);
  }
}

async function attachBoundary(
  endpoint: "firstFrame" | "lastFrame",
  file: File,
): Promise<void> {
  error.value = "";
  busy.value = true;
  try {
    patch({ [endpoint]: await boundaryImage(file) });
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    busy.value = false;
  }
}

async function addReferences(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const files = [...(input.files ?? [])];
  input.value = "";
  if (files.length === 0) return;
  error.value = "";
  busy.value = true;
  try {
    const additions: MinimaxH3ReferenceDraft[] = [];
    for (const file of files) additions.push(await referenceDraft(file));
    patch({ references: [...props.modelValue.references, ...additions] });
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    busy.value = false;
  }
}

function referenceAccept(
  reference: MinimaxH3ReferenceDraft["reference"],
): string {
  return reference.kind === "audio"
    ? ".wav,audio/wav,audio/x-wav,audio/wave"
    : reference.kind === "video"
      ? ".mp4,video/mp4"
      : "image/*";
}

async function reattachReference(index: number, event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  error.value = "";
  busy.value = true;
  try {
    const replacement = await referenceDraft(file);
    const current = props.modelValue.references[index];
    if (!current || replacement.reference.kind !== current.reference.kind) {
      throw new Error(
        `Reference ${index + 1} must be reattached as ${current?.reference.kind ?? "the same media kind"}.`,
      );
    }
    const references = [...props.modelValue.references];
    references[index] = replacement;
    patch({ references });
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    busy.value = false;
  }
}

function move(index: number, delta: -1 | 1): void {
  patch({
    references: moveMinimaxH3Reference(
      props.modelValue.references,
      index,
      index + delta,
    ),
  });
}

function removeReference(index: number): void {
  patch({
    references: props.modelValue.references.filter((_, item) => item !== index),
  });
}

function durationLabel(
  reference: MinimaxH3ReferenceDraft["reference"],
): string | null {
  const duration = minimaxH3ReferenceDurationMs(reference);
  return duration == null ? null : `${(duration / 1_000).toFixed(1)}s`;
}
</script>

<template>
  <section
    class="h3-authoring"
    :class="{ 'h3-authoring--touch': touchFriendly }"
    data-test="h3-authoring"
  >
    <template v-if="task === 'fl2va'">
      <header class="h3-authoring__header">
        <div>
          <strong>Frame endpoints</strong>
          <p>
            <template v-if="requiredEndpoint === 'first'">
              Attach the required opening frame. This reviewed runtime accepts
              no other endpoint mode.
            </template>
            <template v-else>
              Use neither, first, last, or both endpoints. The final endpoint
              follows clip duration.
            </template>
          </p>
        </div>
        <span class="h3-authoring__mode" data-test="h3-mode">{{ mode }}</span>
      </header>
      <div class="h3-authoring__endpoints">
        <div class="h3-authoring__endpoint">
          <span>First frame · frame 0</span>
          <small>{{
            modelValue.firstFrame?.filename ??
            (requiredEndpoint === "first" ? "Required" : "Optional")
          }}</small>
          <ImageDropWell
            :image="modelValue.firstFrame?.data || null"
            :mime-type="modelValue.firstFrame?.mimeType"
            :filename="modelValue.firstFrame?.filename"
            placeholder="Drop the opening frame or click to pick"
            :disabled="disabled"
            :pick-disabled="busy"
            :required="requiredEndpoint === 'first'"
            gallery
            alt="First frame"
            :touch-friendly="touchFriendly"
            test-id="h3-first"
            @file="attachBoundary('firstFrame', $event)"
            @gallery="emit('request-first-frame')"
            @clear="patch({ firstFrame: null })"
          />
        </div>
        <div
          v-if="requiredEndpoint !== 'first' || modelValue.lastFrame"
          class="h3-authoring__endpoint"
          :class="{
            'h3-authoring__endpoint--incompatible':
              requiredEndpoint === 'first',
          }"
        >
          <span>Last frame · final frame</span>
          <small>{{
            requiredEndpoint === "first"
              ? `${modelValue.lastFrame?.filename ?? "Last frame"} · incompatible; remove`
              : (modelValue.lastFrame?.filename ?? "Optional")
          }}</small>
          <ImageDropWell
            :image="modelValue.lastFrame?.data || null"
            :mime-type="modelValue.lastFrame?.mimeType"
            :filename="modelValue.lastFrame?.filename"
            placeholder="Drop the closing frame or click to pick"
            :disabled="disabled"
            :pick-disabled="busy || requiredEndpoint === 'first'"
            gallery
            alt="Last frame"
            :touch-friendly="touchFriendly"
            test-id="h3-last"
            @file="attachBoundary('lastFrame', $event)"
            @gallery="emit('request-last-frame')"
            @clear="patch({ lastFrame: null })"
          />
        </div>
      </div>
    </template>

    <template v-else>
      <header class="h3-authoring__header">
        <div>
          <strong>{{ MINIMAX_H3_RESYNTHESIS_TITLE }}</strong>
          <p>{{ MINIMAX_H3_RESYNTHESIS_GUIDANCE }}</p>
        </div>
      </header>

      <ol class="h3-authoring__references" aria-label="Ordered H3 references">
        <li
          v-for="(draft, index) in modelValue.references"
          :key="`${draft.reference.provenance?.sha256 ?? draft.reference.provenance?.name ?? index}:${index}`"
          class="h3-authoring__reference"
          :data-test="`h3-reference-${index}`"
        >
          <span class="h3-authoring__order" aria-hidden="true">{{
            index + 1
          }}</span>
          <div class="h3-authoring__reference-copy">
            <strong>{{
              minimaxH3ReferenceName(draft.reference, index)
            }}</strong>
            <span>
              {{ draft.reference.kind }}
              <template v-if="durationLabel(draft.reference)">
                · {{ durationLabel(draft.reference) }}</template
              >
              <template
                v-if="
                  draft.reference.kind === 'video' && draft.reference.has_audio
                "
              >
                · soundtrack attached
              </template>
            </span>
            <small v-if="minimaxH3ReferenceNeedsMedia(draft)" role="status">
              Reattach original media before generating.
            </small>
          </div>
          <div
            class="h3-authoring__actions"
            role="group"
            :aria-label="`Reference ${index + 1} controls`"
          >
            <label
              v-if="minimaxH3ReferenceNeedsMedia(draft)"
              class="h3-authoring__reattach"
              :aria-label="`Reattach reference ${index + 1}`"
            >
              Reattach
              <input
                type="file"
                :accept="referenceAccept(draft.reference)"
                :disabled="disabled || busy"
                :data-test="`h3-reference-reattach-${index}`"
                @change="reattachReference(index, $event)"
              />
            </label>
            <button
              type="button"
              :disabled="disabled || index === 0"
              :aria-label="`Move reference ${index + 1} earlier`"
              :data-test="`h3-reference-up-${index}`"
              @click="move(index, -1)"
            >
              ↑
            </button>
            <button
              type="button"
              :disabled="disabled || index === modelValue.references.length - 1"
              :aria-label="`Move reference ${index + 1} later`"
              :data-test="`h3-reference-down-${index}`"
              @click="move(index, 1)"
            >
              ↓
            </button>
            <button
              type="button"
              :disabled="disabled"
              :aria-label="`Remove reference ${index + 1}`"
              :data-test="`h3-reference-remove-${index}`"
              @click="removeReference(index)"
            >
              ×
            </button>
          </div>
        </li>
      </ol>

      <label class="h3-authoring__add">
        <span>Add references in semantic order</span>
        <small
          >Up to {{ MINIMAX_H3_MAX_REFERENCES }} total ·
          {{ MINIMAX_H3_MAX_REFERENCE_IMAGES }} images ·
          {{ MINIMAX_H3_MAX_REFERENCE_VIDEOS }} videos ·
          {{ MINIMAX_H3_MAX_REFERENCE_AUDIOS }} audio</small
        >
        <input
          type="file"
          multiple
          accept="image/*,.mp4,video/mp4,.wav,audio/wav,audio/x-wav,audio/wave"
          :disabled="
            disabled ||
            busy ||
            modelValue.references.length >= MINIMAX_H3_MAX_REFERENCES
          "
          data-test="h3-reference-files"
          @change="addReferences"
        />
      </label>

      <p class="h3-authoring__budget" data-test="h3-reference-budget">
        {{ budget.total }}/{{ MINIMAX_H3_MAX_REFERENCES }} files ·
        {{ (budget.videoDurationMs / 1_000).toFixed(1) }}/15s video ·
        {{ (budget.audioDurationMs / 1_000).toFixed(1) }}/15s audio +
        soundtracks
      </p>
      <ul v-if="budget.errors.length" class="h3-authoring__errors" role="alert">
        <li v-for="message in budget.errors" :key="message">{{ message }}</li>
      </ul>
    </template>

    <p
      v-if="error"
      class="h3-authoring__errors"
      role="alert"
      data-test="h3-media-error"
    >
      {{ error }}
    </p>
  </section>
</template>

<style scoped>
.h3-authoring {
  display: grid;
  gap: 12px;
  color: var(--ink, currentColor);
}
.h3-authoring__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.h3-authoring__header p,
.h3-authoring__endpoint small,
.h3-authoring__reference-copy span,
.h3-authoring__reference-copy small,
.h3-authoring__add small,
.h3-authoring__budget {
  margin: 3px 0 0;
  color: var(--ink-3, #737373);
  font-size: 12px;
  line-height: 1.45;
}
.h3-authoring__mode {
  border: 1px solid var(--edge, #bbb);
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 10px;
  white-space: nowrap;
}
.h3-authoring__endpoints {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.h3-authoring__endpoint,
.h3-authoring__add {
  display: grid;
  gap: 7px;
  border-radius: 10px;
  padding: 12px;
}
/* The well inside draws its own dashed drop target. */
.h3-authoring__endpoint {
  border: 1px solid var(--edge, #bbb);
}
.h3-authoring__add {
  border: 1px dashed var(--edge, #bbb);
}
.h3-authoring__add input {
  max-width: 100%;
  font-size: 16px;
}
.h3-authoring__actions button,
.h3-authoring__reattach {
  min-width: 44px;
  min-height: 44px;
  border: 1px solid var(--edge, #bbb);
  border-radius: 8px;
  background: var(--bench, transparent);
  color: inherit;
}
.h3-authoring__reattach {
  position: relative;
  display: inline-grid;
  place-items: center;
  box-sizing: border-box;
  padding: 0 9px;
  cursor: pointer;
  font-size: 12px;
  overflow: hidden;
}
.h3-authoring__reattach input {
  position: absolute;
  inset: 0;
  width: 100%;
  opacity: 0;
  cursor: pointer;
}
.h3-authoring__references {
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.h3-authoring__reference {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) auto;
  align-items: center;
  gap: 9px;
  border: 1px solid var(--edge, #bbb);
  border-radius: 10px;
  padding: 8px;
}
.h3-authoring__order {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--well, rgba(128, 128, 128, 0.14));
  font-size: 12px;
  font-weight: 700;
}
.h3-authoring__reference-copy {
  display: grid;
  min-width: 0;
}
.h3-authoring__reference-copy strong {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.h3-authoring__actions {
  display: flex;
  gap: 4px;
}
.h3-authoring__errors {
  margin: 0;
  color: var(--stop, #b42318);
  font-size: 12px;
  line-height: 1.45;
}
.h3-authoring--touch button,
.h3-authoring--touch input {
  min-height: 44px;
}
@media (max-width: 520px) {
  .h3-authoring__endpoints {
    grid-template-columns: 1fr;
  }
  .h3-authoring__reference {
    grid-template-columns: 28px minmax(0, 1fr);
  }
  .h3-authoring__actions {
    grid-column: 1 / -1;
    justify-content: flex-end;
  }
}
</style>
