<script setup lang="ts">
import type { VideoNormalizationCapabilities } from "../api/referenceNormalizationTypes";
import OriginalMediaButton from "./OriginalMediaButton.vue";
import { computed, ref, watch } from "vue";
import { minimaxH3PickedMediaReference as referenceDraft } from "../lib/minimaxH3PickedMedia";
import {
  MINIMAX_H3_MAX_REFERENCE_AUDIOS,
  MINIMAX_H3_MAX_REFERENCE_IMAGES,
  MINIMAX_H3_MAX_REFERENCE_VIDEOS,
  MINIMAX_H3_MAX_REFERENCES,
  MINIMAX_H3_RESYNTHESIS_GUIDANCE,
  MINIMAX_H3_RESYNTHESIS_TITLE,
  cloneMinimaxH3AuthoringState,
  hasFreshVideoReferences,
  minimaxH3ReferenceBudget,
  minimaxH3ReferenceDurationMs,
  minimaxH3ReferenceName,
  minimaxH3ReferenceNeedsMedia,
  moveMinimaxH3Reference,
  reattachMinimaxH3Reference,
  type MinimaxH3AuthoringState,
  type MinimaxH3ReferenceDraft,
} from "../lib/minimaxH3Authoring";

// FL2VA first/last boundaries render through the shared SourceMediaWells in
// every surface's primary form; this panel owns only the Ref2VA ordered
// mixed-media references.
const props = withDefaults(
  defineProps<{
    modelValue: MinimaxH3AuthoringState;
    disabled?: boolean;
    touchFriendly?: boolean;
    imagePickerAvailable?: boolean;
    normalizationCapabilities?:
      VideoNormalizationCapabilities | null | undefined;
  }>(),
  {
    disabled: false,
    touchFriendly: false,
    imagePickerAvailable: false,
  },
);

const emit = defineEmits<{
  "update:modelValue": [state: MinimaxH3AuthoringState];
  "open-image-picker": [];
  /** Open the surface's crop editor (dialog / bottom sheet) for this row. */
  "crop-reference": [index: number];
}>();

const error = ref("");
/** Non-error disclosure, e.g. a saved crop that could not follow a reattach. */
const notice = ref("");
const busy = ref(false);
const imagePreviews = ref(
  new Map<MinimaxH3ReferenceDraft["reference"], string>(),
);
const budget = computed(() =>
  minimaxH3ReferenceBudget(props.modelValue.references),
);
function patch(next: Partial<MinimaxH3AuthoringState>): void {
  emit("update:modelValue", {
    ...cloneMinimaxH3AuthoringState(props.modelValue),
    ...next,
  });
}

function attachmentSnapshot() {
  const state = props.modelValue;
  const capabilityStamp = JSON.stringify(
    props.normalizationCapabilities ?? null,
  );
  const capability = props.normalizationCapabilities
    ? (JSON.parse(capabilityStamp) as VideoNormalizationCapabilities)
    : null;
  return {
    state,
    capability,
    check() {
      if (
        props.modelValue !== state ||
        JSON.stringify(props.normalizationCapabilities ?? null) !==
          capabilityStamp
      )
        throw new Error(
          "References or machine changed while reading the original. Choose it again.",
        );
    },
  };
}

async function addReferences(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const files = [...(input.files ?? [])];
  input.value = "";
  if (files.length === 0) return;
  error.value = "";
  busy.value = true;
  const captured = attachmentSnapshot();
  try {
    const additions: MinimaxH3ReferenceDraft[] = [];
    for (const file of files)
      additions.push(await referenceDraft(file, captured.capability));
    captured.check();
    patch({ references: [...captured.state.references, ...additions] });
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    busy.value = false;
  }
}

const videoAccept = computed(() =>
  props.normalizationCapabilities?.available &&
  props.normalizationCapabilities.protocol_version === 1 &&
  props.normalizationCapabilities.input_roles.includes("reference_video")
    ? [
        ...props.normalizationCapabilities.accepted_mime_types,
        ...props.normalizationCapabilities.accepted_extensions.map(
          (extension) => `.${extension}`,
        ),
      ].join(",")
    : ".mp4,video/mp4",
);
function referenceAccept(
  reference: MinimaxH3ReferenceDraft["reference"],
): string {
  return reference.kind === "fresh_video" || reference.kind === "video"
    ? videoAccept.value
    : reference.kind === "audio"
      ? ".wav,audio/wav,audio/x-wav,audio/wave"
      : "image/*";
}

async function reattachReference(index: number, event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  error.value = "";
  busy.value = true;
  const captured = attachmentSnapshot();
  try {
    const replacement = await referenceDraft(file, captured.capability);
    captured.check();
    const result = reattachMinimaxH3Reference(
      props.modelValue,
      index,
      replacement,
    );
    notice.value = result.notice ?? "";
    patch({ references: result.state.references });
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

/** Only an attached image can be cropped; a redacted row needs its bytes first. */
function canCrop(draft: MinimaxH3ReferenceDraft): boolean {
  return (
    draft.reference.kind === "image" &&
    draft.reference.media.authority === "inline"
  );
}

/** The pending crop as thumbnail percentages (`object-fit: cover` keeps the
 * thumbnail's own aspect, so the outline is drawn in the same frame). */
function cropOutlineStyle(
  draft: MinimaxH3ReferenceDraft,
): Record<string, string> | null {
  if (!draft.crop || draft.reference.kind !== "image") return null;
  const { width, height } = draft.reference;
  return {
    left: `${(draft.crop.x / width) * 100}%`,
    top: `${(draft.crop.y / height) * 100}%`,
    width: `${(draft.crop.width / width) * 100}%`,
    height: `${(draft.crop.height / height) * 100}%`,
  };
}

function durationLabel(
  reference: MinimaxH3ReferenceDraft["reference"],
): string | null {
  const duration = minimaxH3ReferenceDurationMs(reference);
  return duration == null ? null : `${(duration / 1_000).toFixed(1)}s`;
}

async function boundedImagePreview(
  reference: MinimaxH3ReferenceDraft["reference"],
): Promise<string | null> {
  if (
    reference.kind !== "image" ||
    reference.media.authority !== "inline" ||
    typeof createImageBitmap !== "function"
  ) {
    return null;
  }
  let bitmap: ImageBitmap | null = null;
  try {
    // Decode in bounded chunks so several-megabyte references never produce
    // both a full binary string and a second full byte copy at once.
    const parts: ArrayBuffer[] = [];
    const chunkSize = 32_768;
    for (let offset = 0; offset < reference.media.data.length;) {
      let end = Math.min(offset + chunkSize, reference.media.data.length);
      if (end < reference.media.data.length) end -= (end - offset) % 4;
      const binary = atob(reference.media.data.slice(offset, end));
      const buffer = new ArrayBuffer(binary.length);
      const bytes = new Uint8Array(buffer);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      parts.push(buffer);
      offset = end;
    }
    const blob = new Blob(parts, { type: reference.mime_type });
    const scale = Math.min(
      1,
      112 / Math.max(reference.width, reference.height),
    );
    const width = Math.max(1, Math.round(reference.width * scale));
    const height = Math.max(1, Math.round(reference.height * scale));
    bitmap = await createImageBitmap(blob, {
      resizeWidth: width,
      resizeHeight: height,
      resizeQuality: "high",
    });
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.78);
  } catch {
    return null;
  } finally {
    bitmap?.close();
  }
}

// A phone can hold several large references. Decode only one thumbnail at a
// time so their transient compressed/decode buffers cannot stack up.
let previewQueue = Promise.resolve();

watch(
  () => props.modelValue.references.map((draft) => draft.reference),
  (references) => {
    const retained = new Map(
      [...imagePreviews.value].filter(([reference]) =>
        references.includes(reference),
      ),
    );
    imagePreviews.value = retained;
    for (const reference of references) {
      if (
        reference.kind !== "image" ||
        reference.media.authority !== "inline" ||
        retained.has(reference)
      ) {
        continue;
      }
      previewQueue = previewQueue.then(async () => {
        const preview = await boundedImagePreview(reference);
        if (!preview) return;
        const current = props.modelValue.references.map(
          (draft) => draft.reference,
        );
        if (!current.includes(reference)) return;
        imagePreviews.value = new Map(imagePreviews.value).set(
          reference,
          preview,
        );
      });
    }
  },
  { immediate: true },
);

function imagePreview(
  reference: MinimaxH3ReferenceDraft["reference"],
): string | null {
  return imagePreviews.value.get(reference) ?? null;
}
</script>

<template>
  <section
    class="h3-authoring"
    :class="{ 'h3-authoring--touch': touchFriendly }"
    data-test="h3-authoring"
    data-drop-target="h3-reference"
  >
    <header class="h3-authoring__header">
      <div>
        <strong>{{ MINIMAX_H3_RESYNTHESIS_TITLE }}</strong>
        <p>{{ MINIMAX_H3_RESYNTHESIS_GUIDANCE }}</p>
      </div>
    </header>

    <ol class="h3-authoring__references" aria-label="Ordered H3 references">
      <li
        v-for="(draft, index) in modelValue.references"
        :key="`${draft.reference.kind === 'fresh_video' ? draft.reference.original.sha256 : (draft.reference.provenance?.sha256 ?? draft.reference.provenance?.name ?? index)}:${index}`"
        class="h3-authoring__reference"
        :data-test="`h3-reference-${index}`"
      >
        <span class="h3-authoring__order" aria-hidden="true">{{
          index + 1
        }}</span>
        <div class="h3-authoring__preview" :data-kind="draft.reference.kind">
          <img
            v-if="imagePreview(draft.reference)"
            :src="imagePreview(draft.reference) ?? ''"
            :alt="minimaxH3ReferenceName(draft.reference, index)"
          />
          <span v-else aria-hidden="true">{{
            draft.reference.kind === "video" ||
            draft.reference.kind === "fresh_video"
              ? "VID"
              : draft.reference.kind === "audio"
                ? "AUD"
                : "IMG"
          }}</span>
          <span
            v-if="cropOutlineStyle(draft)"
            class="h3-authoring__crop-outline"
            :style="cropOutlineStyle(draft) ?? undefined"
            :data-test="`h3-reference-crop-outline-${index}`"
            aria-hidden="true"
          />
        </div>
        <div class="h3-authoring__reference-copy">
          <strong>{{ minimaxH3ReferenceName(draft.reference, index) }}</strong>
          <span>
            {{
              draft.reference.kind === "fresh_video"
                ? "video · verified on submission"
                : draft.reference.kind
            }}
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
            <template v-if="draft.crop">
              · cropped to {{ draft.crop.width }}×{{ draft.crop.height }}
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
          <OriginalMediaButton
            v-if="
              draft.reference.kind !== 'fresh_video' &&
              draft.reference.media.authority === 'inline'
            "
            :data="draft.reference.media.data"
            :mime-type="
              draft.reference.kind === 'audio'
                ? 'audio/wav'
                : draft.reference.kind === 'video'
                  ? 'video/mp4'
                  : draft.reference.mime_type
            "
            :label="minimaxH3ReferenceName(draft.reference, index)"
          />
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
            v-if="draft.reference.kind === 'image'"
            type="button"
            :disabled="disabled || busy || !canCrop(draft)"
            :aria-label="`Crop reference ${index + 1}`"
            :aria-pressed="draft.crop ? 'true' : 'false'"
            :data-test="`h3-reference-crop-${index}`"
            @click="emit('crop-reference', index)"
          >
            Crop
          </button>
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

    <div class="h3-authoring__add">
      <span>Add references in semantic order</span>
      <small
        >Up to {{ MINIMAX_H3_MAX_REFERENCES }} total ·
        {{ MINIMAX_H3_MAX_REFERENCE_IMAGES }} images ·
        {{ MINIMAX_H3_MAX_REFERENCE_VIDEOS }} videos ·
        {{ MINIMAX_H3_MAX_REFERENCE_AUDIOS }} audio</small
      >
      <div class="h3-authoring__add-actions">
        <label class="h3-authoring__choose-files">
          Choose local files
          <input
            type="file"
            multiple
            :accept="`image/*,${videoAccept},.wav,audio/wav,audio/x-wav,audio/wave`"
            :disabled="
              disabled ||
              busy ||
              modelValue.references.length >= MINIMAX_H3_MAX_REFERENCES
            "
            data-test="h3-reference-files"
            @change="addReferences"
          />
        </label>
        <button
          v-if="imagePickerAvailable"
          type="button"
          class="h3-authoring__choose-library"
          :disabled="
            disabled ||
            busy ||
            modelValue.references.length >= MINIMAX_H3_MAX_REFERENCES
          "
          data-test="h3-reference-library"
          @click="emit('open-image-picker')"
        >
          Choose from Library
        </button>
      </div>
    </div>

    <p class="h3-authoring__budget" data-test="h3-reference-budget">
      {{ budget.total }}/{{ MINIMAX_H3_MAX_REFERENCES }} files ·
      <template v-if="hasFreshVideoReferences(modelValue)"
        >Video duration and soundtrack budget verified on submission</template
      >
      <template v-else
        >{{ (budget.videoDurationMs / 1_000).toFixed(1) }}/15s video ·
        {{ (budget.audioDurationMs / 1_000).toFixed(1) }}/15s audio +
        soundtracks</template
      >
    </p>
    <ul v-if="budget.errors.length" class="h3-authoring__errors" role="alert">
      <li v-for="message in budget.errors" :key="message">{{ message }}</li>
    </ul>

    <p
      v-if="notice"
      class="h3-authoring__notice"
      role="status"
      data-test="h3-reference-notice"
    >
      {{ notice }}
    </p>
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
  container-type: inline-size;
  display: grid;
  min-width: 0;
  max-width: 100%;
  gap: 12px;
  color: var(--mold-text);
}
.h3-authoring__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.h3-authoring__header p,
.h3-authoring__reference-copy span,
.h3-authoring__reference-copy small,
.h3-authoring__add small,
.h3-authoring__budget {
  margin: 3px 0 0;
  color: var(--mold-text-dim, #737373);
  font-size: 0.75rem;
  line-height: 1.45;
}
.h3-authoring__add {
  display: grid;
  gap: 7px;
  border: 1px dashed var(--mold-border, #bbb);
  border-radius: 10px;
  padding: 12px;
}
.h3-authoring__actions button,
.h3-authoring__reattach,
.h3-authoring__choose-files,
.h3-authoring__choose-library {
  min-width: 44px;
  min-height: 44px;
  border: 1px solid var(--mold-border, #bbb);
  border-radius: 8px;
  background: var(--mold-bg, transparent);
  color: inherit;
}
.h3-authoring__add-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.h3-authoring__choose-files,
.h3-authoring__choose-library {
  display: inline-grid;
  place-items: center;
  box-sizing: border-box;
  padding: 0 12px;
  cursor: pointer;
  font: inherit;
}
.h3-authoring__choose-files {
  position: relative;
  overflow: hidden;
}
.h3-authoring__choose-files input {
  position: absolute;
  inset: 0;
  width: 100%;
  opacity: 0;
  cursor: pointer;
}
.h3-authoring__reattach {
  position: relative;
  display: inline-grid;
  place-items: center;
  box-sizing: border-box;
  padding: 0 9px;
  cursor: pointer;
  font-size: 0.75rem;
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
  grid-template-columns: 28px 56px minmax(0, 1fr) auto;
  min-width: 0;
  max-width: 100%;
  align-items: center;
  gap: 9px;
  border: 1px solid var(--mold-border, #bbb);
  border-radius: 10px;
  padding: 8px;
}
.h3-authoring__preview {
  display: grid;
  place-items: center;
  width: 56px;
  height: 56px;
  overflow: hidden;
  border: 1px solid var(--mold-border, #bbb);
  border-radius: 8px;
  background: var(--well, rgba(128, 128, 128, 0.14));
  color: var(--mold-text-dim, #737373);
  font-size: 0.625rem;
  font-weight: 700;
  letter-spacing: 0.08em;
}
.h3-authoring__preview {
  position: relative;
}
.h3-authoring__preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.h3-authoring__crop-outline {
  position: absolute;
  box-sizing: border-box;
  border: 1px solid var(--accent, #fff);
  box-shadow: 0 0 0 999px rgba(0, 0, 0, 0.45);
  pointer-events: none;
}
.h3-authoring__notice {
  margin: 0;
  color: var(--mold-text-dim, #737373);
  font-size: 0.75rem;
  line-height: 1.45;
}
.h3-authoring__order {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--well, rgba(128, 128, 128, 0.14));
  font-size: 0.75rem;
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
  flex-wrap: wrap;
  min-width: 0;
  gap: 4px;
}
.h3-authoring__errors {
  margin: 0;
  color: var(--mold-error, #b42318);
  font-size: 0.75rem;
  line-height: 1.45;
}
.h3-authoring--touch button,
.h3-authoring--touch input {
  min-height: 44px;
}
/* The desktop Create inspector is narrow even when the app window is wide, so
 * this must follow the component's own width rather than the viewport. */
@container (max-width: 520px) {
  .h3-authoring__reference {
    grid-template-columns: 28px 56px minmax(0, 1fr);
  }
  .h3-authoring__actions {
    grid-column: 2 / -1;
    grid-row: 2;
    justify-content: flex-start;
  }
}
</style>
