<script setup lang="ts">
import { computed, ref } from "vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import SourceMediaWells, {
  type SourceMediaSlot,
} from "@studio/components/SourceMediaWells.vue";
import MinimaxH3AuthoringPanel from "@studio/components/MinimaxH3AuthoringPanel.vue";
import { sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import { generationCapabilitiesForFamily } from "../../lib/generateCapabilities";
import { resolveExclusiveWells } from "@studio/lib/sourceMediaPlan";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import { sourceImageValidationError } from "@studio/lib/sourceImageCapability";
import { submitsExtend } from "@studio/lib/extend";
import {
  coerceSourceFitForMaskless,
  defaultSourceFitPolicy,
  MASKLESS_SOURCE_FIT_OPTIONS,
  SOURCE_FIT_OPTIONS,
  sourceFitHelp,
} from "@studio/lib/sourceFit";
import { strengthSemantics } from "@studio/lib/strengthSemantics";
import { blobToBase64 } from "../../lib/base64";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { sourceConditioningLimitLabel } from "@studio/lib/sourceResolution";
import {
  emptyMinimaxH3AuthoringState,
  setMinimaxH3BoundaryFile,
  type MinimaxH3GalleryImageResult,
  type MinimaxH3BoundaryEndpoint,
} from "@studio/lib/minimaxH3Authoring";
import type {
  GenerateFormState,
  ModelInfoExtended,
  SourceFitPolicy,
  SourceImageState,
} from "../../types";

/**
 * The primary-form image conditioning card: the model dictates whether (and
 * how) it renders, exactly like resolutions — one shared `sourceMediaPlan`
 * policy, never a local heuristic. Lives beside the essentials on tablet+
 * and in the phone control stack; the Advanced groups no longer own it.
 */
const props = withDefaults(
  defineProps<{
    modelValue: GenerateFormState;
    family: string;
    /** Installed models on the selected generation route. */
    models?: ModelInfoExtended[];
  }>(),
  { models: () => [] },
);

const emit = defineEmits<{
  "update:modelValue": [value: GenerateFormState];
  "open-picker": [];
  "open-target-picker": [];
  "clear-source": [];
  "open-end-frame-picker": [];
  "clear-end-frame": [];
  "open-mask": [];
  "open-h3-first-frame-picker": [];
  "open-h3-last-frame-picker": [];
  "open-h3-reference-picker": [];
  /** Open the picker for an EXCLUSIVE recipe's reference strip. */
  "open-reference-picker": [];
  /** Open the page-level crop editor for ordered reference `index`. */
  "crop-h3-reference": [index: number];
}>();

function patch(next: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...next });
}

const selectedModel = computed(
  () =>
    props.models.find((model) => model.name === props.modelValue.model) ?? null,
);
// The advertised per-model source-image contract (#772) rides as the fifth
// argument; the shared kit owns the absent-field fallback for older servers.
const caps = computed(() =>
  generationCapabilitiesForFamily(
    props.family,
    props.modelValue.model,
    props.modelValue.pipeline,
    selectedModel.value?.guidance_capabilities,
    selectedModel.value?.source_image ?? props.modelValue.sourceImageCapability,
    effectiveGenerationRecipe(selectedModel.value, props.modelValue.pipeline),
  ),
);
/** The model's own image-attachment shape — the single shared policy. */
const plan = computed(() => sourceMediaPlan(caps.value));
/** Family-scoped label for the shared `strength` wire field (#1055). */
const strength = computed(() => strengthSemantics(props.family));
/** A strip with no Target is a pure reference strip; the ceiling and the
 * roles come from the advertised recipe, never from a model name. */
const referencesOnly = computed(
  () =>
    (plan.value.kind === "attachments" && plan.value.primary === null) ||
    plan.value.kind === "single-or-references",
);
const referenceMax = computed(() => caps.value.referenceImages?.max ?? null);
/** The exclusive (Klein) references live in their own store, because the
 * source well keeps `imageAttachments[0]`. */
const referenceImages = computed(() => props.modelValue.referenceImages ?? []);
/**
 * The parking rule: whichever well holds media is the active one, the other
 * parks with an inline note and KEEPS its media, and Generate stays enabled.
 */
const exclusive = computed(() =>
  plan.value.kind === "single-or-references"
    ? resolveExclusiveWells({
        hasSource: Boolean(props.modelValue.imageAttachments[0]?.base64),
        referenceCount: referenceImages.value.length,
        lastWrite: props.modelValue.exclusiveWell ?? null,
      })
    : null,
);
/** Fit, strength and the repaint mask describe a SOURCE image. */
const sourceRefinements = computed(
  () =>
    plan.value.kind === "single" || exclusive.value?.active !== "references",
);
const sourceLimitLabel = computed(() =>
  sourceConditioningLimitLabel(
    selectedModel.value ?? props.family,
    props.modelValue.pipeline,
  ),
);

const kicker = computed(() =>
  plan.value.kind === "h3-boundaries"
    ? "Frame endpoints"
    : plan.value.kind === "h3-references"
      ? "Ordered references"
      : plan.value.kind === "single-or-references"
        ? "Source or references"
        : plan.value.kind === "attachments"
          ? referencesOnly.value
            ? "Reference images"
            : "Edit images"
          : "Source image",
);

const hasSource = computed(() => props.modelValue.imageAttachments.length > 0);
const sourceAttachment = computed<SourceImageState | null>(
  () => props.modelValue.imageAttachments[0] ?? null,
);
const hasEndFrame = computed(
  () => caps.value.supportsEndFrame && props.modelValue.endFrame != null,
);
/** Why the attached conditioning would be refused, in the server's own order.
 * A continuation carries its own first frames (#783), so the well's notice
 * has to read it the way submit and admission do or the two disagree. */
const sourceConditioningError = computed(() =>
  plan.value.kind === "single"
    ? sourceImageValidationError({
        capability: caps.value.sourceImageCapability,
        hasSourceImage: hasSource.value,
        isExtend: submitsExtend({
          family: props.family,
          extendVideo: props.modelValue.extendVideo,
          extendVideoPath: props.modelValue.extendVideoPath,
        }),
        hasEndFrame: hasEndFrame.value,
        frames: caps.value.supportsVideo ? props.modelValue.frames : null,
        model: props.modelValue.model,
      })
    : null,
);

const uploadError = ref<string | null>(null);

/** Decode the PNG/JPEG header for dimensions — the same facts a gallery pick
 * carries, and a format gate (the engine accepts nothing else) that also
 * covers drag-and-drop, which bypasses the file input's accept filter. */
async function fileToSourceImage(file: File): Promise<SourceImageState | null> {
  const base64 = await blobToBase64(file);
  const dimensions = imageDimensionsFromBase64(base64);
  if (!dimensions) {
    uploadError.value = "Only PNG or JPEG images can be used here.";
    return null;
  }
  uploadError.value = null;
  return {
    kind: "upload",
    filename: file.name,
    base64,
    width: dimensions.width,
    height: dimensions.height,
    mime: file.type || null,
  };
}

async function onWellFile(slot: SourceMediaSlot, file: File) {
  const image = await fileToSourceImage(file);
  if (!image) return;
  if (slot === "source") {
    // Every newly selected source starts from the shared crop-fill policy.
    patch({
      imageAttachments:
        plan.value.kind === "attachments"
          ? [image, ...props.modelValue.imageAttachments.slice(1)]
          : [image],
      sourceFitPolicy: defaultSourceFitPolicy(),
      // Last write wins on an exclusive recipe: the references park, kept.
      exclusiveWell: "source",
    });
  } else {
    patch({ endFrame: image });
  }
}
/** Drop the exclusive strip; the parked source becomes active again. */
function clearReferences() {
  patch({ referenceImages: [], exclusiveWell: "source" });
}

/**
 * The References strip is a DROP TARGET in its own right.
 *
 * It renders `data-drop-target="references"` so a shell that hit-tests the
 * cursor (`elementFromPoint(...).closest(…)`) can name it, and it handles the
 * HTML5 drop itself — which both writes through the SAME fields the strip's
 * picker writes and marks the event handled, so `CreatePage.vue`'s
 * window-level listener (`event.defaultPrevented`) leaves it alone. Without
 * this a file dragged onto the strip fell through to the window and landed on
 * the SOURCE well, which on an exclusive (Klein) recipe parks the very strip
 * the user was adding to.
 */
async function onStripDrop(event: DragEvent) {
  const files = Array.from(event.dataTransfer?.files ?? []);
  if (files.length === 0) return;
  const images: SourceImageState[] = [];
  for (const file of files) {
    const image = await fileToSourceImage(file);
    if (!image) return;
    images.push(image);
  }
  // The ceiling is the RECIPE's, never a client constant, and the strip is
  // APPENDED to — a drop that replaced it lost every earlier picture.
  const max = referenceMax.value ?? undefined;
  if (plan.value.kind === "single-or-references") {
    patch({
      referenceImages: [...referenceImages.value, ...images].slice(0, max),
      // Last write wins on an exclusive recipe: the source parks, kept.
      exclusiveWell: "references",
    });
    return;
  }
  patch({
    imageAttachments: [...props.modelValue.imageAttachments, ...images].slice(
      0,
      max,
    ),
  });
}
function onWellGallery(slot: SourceMediaSlot) {
  if (slot === "source") {
    if (plan.value.kind === "attachments") emit("open-target-picker");
    else emit("open-picker");
  } else emit("open-end-frame-picker");
}
function onWellClear(slot: SourceMediaSlot) {
  if (slot === "source") {
    if (plan.value.kind === "attachments") {
      patch({ imageAttachments: props.modelValue.imageAttachments.slice(1) });
    } else emit("clear-source");
  } else emit("clear-end-frame");
}

// ── MiniMax H3 FL2VA boundaries ───────────────────────────────────────
// The same standard wells, backed by the dedicated H3 authoring state and the
// shared studio setters so a file and a gallery pick produce identical facts.
const h3Authoring = computed(
  () => props.modelValue.h3Authoring ?? emptyMinimaxH3AuthoringState(),
);
const h3Error = ref<string | null>(null);
function h3Endpoint(slot: SourceMediaSlot): MinimaxH3BoundaryEndpoint {
  return slot === "source" ? "firstFrame" : "lastFrame";
}
function applyH3(result: MinimaxH3GalleryImageResult) {
  if (!result.ok) {
    h3Error.value = result.error;
    return;
  }
  h3Error.value = null;
  patch({ h3Authoring: result.state });
}
async function onH3File(slot: SourceMediaSlot, file: File) {
  applyH3(
    await setMinimaxH3BoundaryFile(
      props.modelValue.h3Authoring,
      h3Endpoint(slot),
      file,
    ),
  );
}
function onH3Gallery(slot: SourceMediaSlot) {
  if (slot === "source") emit("open-h3-first-frame-picker");
  else emit("open-h3-last-frame-picker");
}
function onH3Clear(slot: SourceMediaSlot) {
  h3Error.value = null;
  patch({
    h3Authoring: { ...h3Authoring.value, [h3Endpoint(slot)]: null },
  });
}
function setH3Authoring(value: typeof h3Authoring.value) {
  patch({ h3Authoring: value });
}

// ── Fit / strength / mask (single-source refinement) ──────────────────
const fitOptions = SOURCE_FIT_OPTIONS;
const fitMode = computed(
  () => props.modelValue.sourceFitPolicy?.mode ?? defaultSourceFitPolicy().mode,
);
/** H3 has no repaint mask; pad-repaint is never offered for its boundaries. */
const masklessFitOptions = MASKLESS_SOURCE_FIT_OPTIONS;
const masklessFitMode = computed(
  () =>
    coerceSourceFitForMaskless(
      props.modelValue.sourceFitPolicy ?? { mode: "crop-fill" },
    ).mode,
);
function setFit(mode: string) {
  if (mode === "upscale-then-fit") {
    patch({
      sourceFitPolicy: {
        mode,
        upscalerModel:
          props.modelValue.upscaleModel || "real-esrgan-x4plus:fp16",
        fit: { mode: "crop-fill" },
      },
    });
    return;
  }
  patch({ sourceFitPolicy: { mode } as SourceFitPolicy });
}

// ── ControlNet (guidance image + model + scale, sd15 today) ───────────
const showControlNet = computed(() => caps.value.supportsControlNet);
const hasControl = computed(() => props.modelValue.controlImage != null);
const controlModels = computed(() =>
  props.models.filter(
    (model) => model.downloaded && model.family === "controlnet",
  ),
);
async function onControlImage(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0] ?? null;
  input.value = "";
  if (!file) return;
  const image = await fileToSourceImage(file);
  if (image) patch({ controlImage: image });
}
function clearControl() {
  patch({ controlImage: null });
}
</script>

<template>
  <section
    v-if="plan.kind !== 'none'"
    class="smp"
    data-test="source-media-panel"
  >
    <div class="smp__head">
      <span class="smp__kicker">{{ kicker }}</span>
    </div>

    <MinimaxH3AuthoringPanel
      v-if="plan.kind === 'h3-references'"
      :model-value="h3Authoring"
      image-picker-available
      @update:model-value="setH3Authoring"
      @open-image-picker="emit('open-h3-reference-picker')"
      @crop-reference="emit('crop-h3-reference', $event)"
    />

    <!-- Ordered picture strip (Qwen edit / FLUX.2 references). -->
    <template v-else-if="plan.kind === 'attachments'">
      <SourceMediaWells
        v-if="plan.primary === 'target'"
        :plan="plan"
        :source="
          sourceAttachment
            ? {
                data: sourceAttachment.base64,
                mimeType: sourceAttachment.mime,
                filename: sourceAttachment.filename,
              }
            : null
        "
        :error="uploadError"
        @file="onWellFile"
        @gallery="onWellGallery"
        @clear="onWellClear"
      />
      <!-- The strip is its own drop target: a file dragged here appends to the
           references, and the handled event keeps the window listener off it. -->
      <div
        class="smp__strip"
        data-test="reference-strip"
        data-drop-target="references"
        @dragover.prevent
        @drop.prevent="onStripDrop"
      >
        <p
          v-if="plan.required && plan.primary === null"
          class="smp__required"
          data-test="source-required-badge"
        >
          Required — this checkpoint renders from an image.
        </p>
        <button
          v-if="!hasSource && plan.primary === null"
          type="button"
          class="smp__dropzone"
          data-test="source-attach"
          :aria-required="plan.required || undefined"
          @click="emit('open-picker')"
        >
          {{ referencesOnly ? "Attach references or " : "Attach images or "
          }}<span class="smp__accent">browse</span>
        </button>
        <div v-else-if="plan.primary === null">
          <div class="smp__source-row">
            <span class="smp__source-name">
              {{ modelValue.imageAttachments[0]?.filename }}
              <template v-if="modelValue.imageAttachments.length > 1">
                +{{ modelValue.imageAttachments.length - 1 }} more
              </template>
            </span>
            <button
              type="button"
              class="smp__remove"
              data-test="source-remove"
              @click="emit('clear-source')"
            >
              Remove
            </button>
          </div>
          <button
            type="button"
            class="smp__dropzone smp__dropzone--compact"
            data-test="source-attach-more"
            @click="emit('open-picker')"
          >
            Add more or <span class="smp__accent">browse</span>
          </button>
        </div>
        <button
          v-else-if="hasSource"
          type="button"
          class="smp__dropzone smp__dropzone--compact"
          data-test="source-attach-more"
          @click="emit('open-picker')"
        >
          Add references or <span class="smp__accent">browse</span>
        </button>
      </div>
      <div v-if="plan.primary === 'target' && hasSource" class="smp__field">
        <label class="smp__label">Fit to canvas</label>
        <SegmentedControl
          :model-value="masklessFitMode"
          :options="masklessFitOptions"
          label="Fit to canvas"
          @update:model-value="setFit"
        />
        <p class="smp__hint" data-test="source-fit-help">
          {{ sourceFitHelp(masklessFitMode) }} Qwen conditioning limit:
          {{ sourceLimitLabel }} from this model; Output size is separate.
        </p>
      </div>
      <p class="smp__hint">
        {{
          referencesOnly
            ? referenceMax === null
              ? "Ordered references."
              : `Up to ${referenceMax} ordered references.`
            : "First picture is the edit Target; the rest are References."
        }}
      </p>
    </template>

    <!-- One source image (+ optional end frame), or — on an EXCLUSIVE recipe
         (FLUX.2 [klein]) — the same well plus the reference strip below it,
         mutually exclusive. -->
    <template
      v-else-if="plan.kind === 'single' || plan.kind === 'single-or-references'"
    >
      <SourceMediaWells
        :plan="plan"
        :parked="exclusive?.parked === 'source'"
        :note="exclusive?.parked === 'source' ? exclusive.note : null"
        :source="
          sourceAttachment
            ? {
                data: sourceAttachment.base64,
                mimeType: sourceAttachment.mime,
                filename: sourceAttachment.filename,
              }
            : null
        "
        :end-frame="
          modelValue.endFrame
            ? {
                data: modelValue.endFrame.base64,
                mimeType: modelValue.endFrame.mime,
                filename: modelValue.endFrame.filename,
              }
            : null
        "
        :error="uploadError ?? sourceConditioningError"
        @file="onWellFile"
        @gallery="onWellGallery"
        @clear="onWellClear"
      />

      <!-- The exclusive reference strip: the SAME picker the strip-only
           layouts use, driven by the plan. -->
      <div
        v-if="plan.kind === 'single-or-references'"
        class="smp__strip"
        data-test="reference-strip"
        data-drop-target="references"
        @dragover.prevent
        @drop.prevent="onStripDrop"
      >
        <div class="smp__subhead">References</div>
        <button
          v-if="referenceImages.length === 0"
          type="button"
          class="smp__dropzone smp__dropzone--compact"
          data-test="reference-attach"
          @click="emit('open-reference-picker')"
        >
          Attach references or <span class="smp__accent">browse</span>
        </button>
        <div v-else>
          <div class="smp__source-row">
            <span class="smp__source-name" data-test="reference-names">
              {{ referenceImages[0]?.filename }}
              <template v-if="referenceImages.length > 1">
                +{{ referenceImages.length - 1 }} more
              </template>
            </span>
            <button
              type="button"
              class="smp__remove"
              data-test="reference-remove"
              @click="clearReferences"
            >
              Remove
            </button>
          </div>
          <button
            type="button"
            class="smp__dropzone smp__dropzone--compact"
            data-test="reference-attach-more"
            @click="emit('open-reference-picker')"
          >
            Add more or <span class="smp__accent">browse</span>
          </button>
        </div>
        <p
          v-if="exclusive?.parked === 'references'"
          class="smp__hint"
          data-test="references-parked-note"
        >
          {{ exclusive.note }}
        </p>
        <p v-else class="smp__hint">
          {{
            referenceMax === null
              ? "Ordered references."
              : `Up to ${referenceMax} ordered references.`
          }}
        </p>
      </div>

      <template v-if="sourceRefinements && hasSource">
        <!-- A canvasless recipe (a 3-D mesh) has no canvas to fit onto, and
             `toRequest` sends no `source_fit` for one. -->
        <div v-if="!caps.canvasless" class="smp__field">
          <label class="smp__label">Fit to canvas</label>
          <SegmentedControl
            :model-value="fitMode"
            :options="fitOptions"
            label="Fit to canvas"
            @update:model-value="setFit"
          />
        </div>
        <SliderRow
          v-if="caps.supportsStrength"
          :label="strength.label"
          :model-value="modelValue.strength"
          :min="0"
          :max="1"
          :step="0.01"
          :value-label="modelValue.strength.toFixed(2)"
          @update:model-value="patch({ strength: $event })"
        />
        <p v-if="caps.supportsStrength" class="smp__hint">
          {{ strength.hint }}
        </p>
        <button
          v-if="caps.supportsMask"
          type="button"
          class="smp__mask"
          data-test="source-mask"
          @click="emit('open-mask')"
        >
          {{
            modelValue.maskImage ? "Mask applied · edit" : "Edit inpaint mask"
          }}
        </button>
      </template>

      <!-- ControlNet (guidance image + model + scale). -->
      <div
        v-if="showControlNet"
        class="smp__controlnet"
        data-test="controlnet-block"
      >
        <div class="smp__subhead">ControlNet</div>
        <label
          v-if="!hasControl"
          class="smp__filezone"
          data-test="control-attach"
        >
          <span
            >Attach a control image or
            <span class="smp__accent">browse</span></span
          >
          <input
            type="file"
            accept="image/png,image/jpeg"
            class="smp__file-input"
            @change="onControlImage"
          />
        </label>
        <template v-else>
          <div class="smp__source-row">
            <span class="smp__source-name">{{
              modelValue.controlImage?.filename
            }}</span>
            <button
              type="button"
              class="smp__remove"
              data-test="control-remove"
              @click="clearControl"
            >
              Remove
            </button>
          </div>
          <div class="smp__field">
            <label class="smp__label">Control model</label>
            <input
              class="smp__input"
              data-test="control-model"
              list="installed-controlnet-models"
              placeholder="e.g. control_v11p_sd15_canny"
              :value="modelValue.controlModel"
              @input="
                patch({
                  controlModel: ($event.target as HTMLInputElement).value,
                })
              "
            />
            <datalist id="installed-controlnet-models">
              <option
                v-for="model in controlModels"
                :key="model.name"
                :value="model.name"
              />
            </datalist>
          </div>
          <SliderRow
            label="Control scale"
            data-test="control-scale"
            :model-value="modelValue.controlScale"
            :min="0"
            :max="2"
            :step="0.05"
            :value-label="modelValue.controlScale.toFixed(2)"
            @update:model-value="patch({ controlScale: $event })"
          />
        </template>
      </div>
    </template>

    <!-- MiniMax H3 FL2VA boundaries: the exact same wells, H3-owned state. -->
    <template v-else-if="plan.kind === 'h3-boundaries'">
      <SourceMediaWells
        :plan="plan"
        :source="h3Authoring.firstFrame"
        :end-frame="h3Authoring.lastFrame"
        :error="uploadError ?? h3Error"
        @file="onH3File"
        @gallery="onH3Gallery"
        @clear="onH3Clear"
      />
      <!-- The same client-side fit as an ordinary source, coerced maskless
           and applied to both boundaries at submit. -->
      <div
        v-if="h3Authoring.firstFrame || h3Authoring.lastFrame"
        class="smp__field"
      >
        <label class="smp__label">Fit to canvas</label>
        <SegmentedControl
          :model-value="masklessFitMode"
          :options="masklessFitOptions"
          label="Fit to canvas"
          @update:model-value="setFit"
        />
      </div>
    </template>
  </section>
</template>

<style scoped>
.smp {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 18px;
}
.smp__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 9px;
  margin-bottom: 12px;
}
.smp__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.smp__dropzone {
  width: 100%;
  border: 1.5px dashed var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-card);
  padding: 26px;
  font-size: 13px;
  cursor: pointer;
}
.smp__dropzone--compact {
  margin-top: 10px;
  padding: 12px;
}
.smp__accent {
  color: var(--safelight);
  font-weight: 600;
}
.smp__source-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.smp__source-name {
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.smp__remove {
  border: 0;
  background: transparent;
  color: var(--stop);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.smp__mask {
  margin-top: 12px;
  width: 100%;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: var(--radius-control-lg);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}
.smp__hint {
  font-size: 10.5px;
  color: var(--ink-3);
  margin-top: 6px;
}
/* Textual, never colour alone: the requirement has to survive a monochrome
 * or high-contrast rendering. */
.smp__required {
  font-size: 11px;
  font-weight: 600;
  color: var(--safelight);
  margin: 0 0 10px;
}
.smp__field {
  margin-top: 12px;
}
.smp__label {
  display: block;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--ink-2);
  margin-bottom: 6px;
}
.smp__input {
  width: 100%;
  border: 1px solid var(--ce);
  background: var(--bath);
  color: var(--ink);
  border-radius: var(--radius-control);
  padding: 8px 10px;
  font-size: 13px;
}
.smp__subhead {
  font-size: 11.5px;
  font-weight: 700;
  color: var(--ink-2);
  margin: 14px 0 8px;
}
.smp__subhead::before {
  content: "";
}
.smp__filezone {
  position: relative;
  display: block;
  border: 1.5px dashed var(--ce);
  border-radius: var(--radius-card);
  padding: 14px;
  font-size: 12.5px;
  color: var(--ink-2);
  text-align: center;
  cursor: pointer;
}
.smp__file-input {
  position: absolute;
  inset: 0;
  width: 100%;
  opacity: 0;
  cursor: pointer;
}
</style>
