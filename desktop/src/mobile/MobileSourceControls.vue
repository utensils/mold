<script setup lang="ts">
import { computed, ref, watch } from "vue";
import MaskEditorModal from "../components/generate/MaskEditorModal.vue";
import { fetchCatalogInstalled } from "../lib/api/catalog";
import type { ApiTarget } from "../lib/api/client";
import type { CatalogEntry, ModelEntry } from "../lib/api/types";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import { buildControlNetOptions } from "../lib/controlNetOptions";
import { attachmentRoleLabel, attachmentTitleLabel, moveAttachment } from "../lib/editAttachments";
import type { GenerateForm } from "../lib/generateForm";
import {
  inlineGenerationMediaBytes,
  decodedBase64Bytes,
  MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
  MOBILE_MEDIA_BUDGET_ERROR,
  sourceConditioningValidationError,
  type InlineGenerationMediaField,
} from "../lib/generateValidation";
import { base64ToDataUrl, fileToBase64, isStillImageFile } from "../lib/image";
import {
  coerceSourceFitForMaskless,
  defaultSourceFitPolicy,
  MASKLESS_SOURCE_FIT_OPTIONS,
  sourceFitHelp,
  sourceFitPolicyForMode,
  type SourceFitMode,
} from "@studio/lib/sourceFit";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import { strengthSemantics } from "@studio/lib/strengthSemantics";
import { sourceConditioningLimitLabel } from "@studio/lib/sourceResolution";
import MobileImagePickerSheet, {
  type MobileGallerySource,
  type MobilePickedImage,
} from "./MobileImagePickerSheet.vue";
import MobileReferenceCropSheet from "./MobileReferenceCropSheet.vue";
import type { ReferenceCrop } from "@studio/lib/referenceCrop";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import SourceMediaWells, { type SourceMediaSlot } from "@studio/components/SourceMediaWells.vue";
import MinimaxH3AuthoringPanel from "@studio/components/MinimaxH3AuthoringPanel.vue";
import { resolveExclusiveWells, sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import {
  appendMinimaxH3PickedImageReferences,
  emptyMinimaxH3AuthoringState,
  minimaxH3ReferenceCropTarget,
  setMinimaxH3BoundaryFile,
  setMinimaxH3PickedImageBoundary,
  setMinimaxH3ReferenceCrop,
  type MinimaxH3BoundaryEndpoint,
  type MinimaxH3GalleryImageResult,
} from "@studio/lib/minimaxH3Authoring";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    target?: ApiTarget | null;
    gallerySources?: MobileGallerySource[];
    controlModels?: ModelEntry[];
    upscalers?: ModelEntry[];
    model?: ModelEntry | null;
  }>(),
  {
    target: null,
    gallerySources: () => [],
    controlModels: () => [],
    upscalers: () => [],
    model: null,
  },
);

const emit = defineEmits<{
  "validity-change": [valid: boolean];
}>();

// The selected checkpoint's own advertised source-image contract (#772) rides
// as the fifth argument, exactly as it does in the desktop well: wan's
// checkpoints split T2V / I2V-optional / I2V-required and only the server can
// tell them apart. Never read `source_image` (or a family set) here — the
// helper owns the absent-field fallback that keeps older servers on today's
// behaviour, and `supportsEndFrame` (#779) rides the same resolution.
const caps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    props.model?.guidance_capabilities ?? props.form.guidanceCapabilities,
    props.model?.source_image ?? props.form.sourceImageCapability,
    effectiveGenerationRecipe(props.model, props.form.pipeline),
  ),
);
// Family-scoped label for the shared `strength` wire field (#1055).
const strength = computed(() => strengthSemantics(props.form.family));
/**
 * A canvasless recipe (a 3-D mesh) renders no pixel canvas, so there is
 * nothing for the source to be cropped, padded or upscaled toward: the fit
 * policy is not a choice the user has here. `buildRequest` records no
 * `source_fit` for such a request either, so offering the control would
 * promise a setting the wire drops.
 */
// The family is the legacy fallback for a form restored before the profile
// landed — the same reading `buildRequest` takes, so the control and the
// wire cannot disagree about whether a fit policy exists.
const canvasless = computed(() => caps.value.canvasless || isMeshFamily(props.form.family));
/** The model's own image-attachment shape — the single shared policy. */
const plan = computed(() => sourceMediaPlan(caps.value));
const isAttachmentMode = computed(() => plan.value.kind === "attachments");
const editFitMode = computed(() => coerceSourceFitForMaskless(props.form.sourceFit).mode);
const sourceLimitLabel = computed(() =>
  sourceConditioningLimitLabel(props.model ?? props.form.family, props.form.pipeline),
);

// ── MiniMax H3 FL2VA boundaries ─────────────────────────────────────────────
// The same standard wells, backed by the dedicated H3 authoring state and the
// shared studio setters so a file and a gallery pick produce identical facts.
const h3Authoring = computed(() => props.form.h3Authoring ?? emptyMinimaxH3AuthoringState());
function setH3Authoring(value: typeof h3Authoring.value): void {
  props.form.h3Authoring = value;
}
const h3PickerTarget = ref<MinimaxH3BoundaryEndpoint | null>(null);
const h3ReferencePickerOpen = ref(false);
/** Which ordered reference the crop sheet is editing; null when closed. */
const h3CropIndex = ref<number | null>(null);
const h3CropTarget = computed(() =>
  minimaxH3ReferenceCropTarget(props.form.h3Authoring, h3CropIndex.value),
);
function applyH3ReferenceCrop(crop: ReferenceCrop | null): void {
  if (h3CropIndex.value === null) return;
  props.form.h3Authoring = setMinimaxH3ReferenceCrop(
    props.form.h3Authoring ?? emptyMinimaxH3AuthoringState(),
    h3CropIndex.value,
    crop,
  );
  h3CropIndex.value = null;
}
const h3Error = ref<string | null>(null);
const h3PickerMaxBytes = computed(() =>
  Math.max(
    0,
    MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES -
      inlineGenerationMediaBytes(
        props.form,
        h3PickerTarget.value === "lastFrame" ? "h3LastFrame" : "h3FirstFrame",
      ),
  ),
);
function h3Endpoint(slot: SourceMediaSlot): MinimaxH3BoundaryEndpoint {
  return slot === "source" ? "firstFrame" : "lastFrame";
}
function applyH3(result: MinimaxH3GalleryImageResult): void {
  if (!result.ok) {
    h3Error.value = result.error;
    return;
  }
  h3Error.value = null;
  props.form.h3Authoring = result.state;
}
async function onH3File(slot: SourceMediaSlot, file: File): Promise<void> {
  // The same 45 MiB request budget the picker sheet enforces — checked before
  // the file is read so an oversized base64 never lands in the WebView.
  const endpoint = h3Endpoint(slot);
  const budget = Math.max(
    0,
    MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES -
      inlineGenerationMediaBytes(
        props.form,
        endpoint === "lastFrame" ? "h3LastFrame" : "h3FirstFrame",
      ),
  );
  if (file.size > budget) {
    h3Error.value = MOBILE_MEDIA_BUDGET_ERROR;
    return;
  }
  applyH3(await setMinimaxH3BoundaryFile(props.form.h3Authoring, endpoint, file));
}
function onH3Gallery(slot: SourceMediaSlot): void {
  h3PickerTarget.value = h3Endpoint(slot);
}
function onH3Clear(slot: SourceMediaSlot): void {
  h3Error.value = null;
  props.form.h3Authoring = { ...h3Authoring.value, [h3Endpoint(slot)]: null };
}
function onH3Picked(image: MobilePickedImage): void {
  const endpoint = h3PickerTarget.value;
  if (!endpoint) return;
  applyH3(setMinimaxH3PickedImageBoundary(props.form.h3Authoring, endpoint, image));
}
async function onH3ReferenceImagesPicked(images: MobilePickedImage[]): Promise<void> {
  applyH3(await appendMinimaxH3PickedImageReferences(props.form.h3Authoring, images));
}
const h3ReferencePickerMaxBytes = computed(() =>
  Math.max(0, MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES - inlineGenerationMediaBytes(props.form)),
);
/** A strip with no Target is a pure reference strip; the ceiling and the
 * roles come from the advertised recipe, never from a model name. */
const referencesOnly = computed(
  () =>
    (plan.value.kind === "attachments" && plan.value.primary === null) ||
    plan.value.kind === "single-or-references",
);
const referenceMax = computed(() =>
  plan.value.kind === "attachments"
    ? plan.value.max
    : plan.value.kind === "single-or-references"
      ? plan.value.references.max
      : null,
);
/** The exclusive (Klein) parking rule — see `resolveExclusiveWells`. */
const exclusive = computed(() =>
  plan.value.kind === "single-or-references"
    ? resolveExclusiveWells({
        hasSource: Boolean(props.form.sourceImage),
        referenceCount: props.form.imageAttachments.length,
        lastWrite: props.form.exclusiveWell ?? null,
      })
    : null,
);
/** Klein renders both: the ordered strip AND the source well. */
const showStrip = computed(
  () => plan.value.kind === "attachments" || plan.value.kind === "single-or-references",
);
const showSourceWell = computed(
  () => plan.value.kind === "single" || plan.value.kind === "single-or-references",
);
/** Fit, strength and the mask describe a SOURCE image. */
const sourceRefinements = computed(
  () => plan.value.kind === "single" || exclusive.value?.active !== "references",
);
const error = ref("");
const maskOpen = ref(false);
const sourcePickerOpen = ref(false);
const endFramePickerOpen = ref(false);
const sourcePickerMaxBytes = computed(() =>
  Math.max(
    0,
    MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES -
      (isAttachmentMode.value
        ? inlineGenerationMediaBytes(props.form) -
          decodedBase64Bytes(props.form.imageAttachments[0])
        : inlineGenerationMediaBytes(props.form, "sourceImage")),
  ),
);
const endFramePickerMaxBytes = computed(() =>
  Math.max(
    0,
    MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES - inlineGenerationMediaBytes(props.form, "endFrame"),
  ),
);

const editInput = ref<HTMLInputElement | null>(null);
const maskInput = ref<HTMLInputElement | null>(null);
const controlInput = ref<HTMLInputElement | null>(null);

const CUSTOM_CONTROL_MODEL = "__custom__";
const controlCustomMode = ref(false);
const catalogControlModels = ref<CatalogEntry[]>([]);
const controlOptions = computed(() =>
  buildControlNetOptions(
    props.controlModels,
    catalogControlModels.value,
    controlCustomMode.value ? "" : props.form.controlModel,
  ),
);
const controlSelectValue = computed(() =>
  controlCustomMode.value ? CUSTOM_CONTROL_MODEL : props.form.controlModel,
);
const validationError = computed(() => {
  if (caps.value.sourceImageMode === "qwen-edit" && props.form.imageAttachments.length === 0) {
    return "Add a Target photo before generating an edit.";
  }
  if (caps.value.supportsControlNet && props.form.controlImage && !props.form.controlModel.trim()) {
    return "Choose an installed ControlNet model for the control photo.";
  }
  return null;
});
/**
 * Why the attached conditioning would be refused, in the order admission
 * checks it. Its own slot rather than a branch of `validationError`, because
 * the ControlNet paragraph must keep naming the ControlNet problem.
 */
const conditioningError = computed(() => sourceConditioningValidationError(props.form));
const valid = computed(() => !validationError.value && !conditioningError.value);

watch(valid, (next) => emit("validity-change", next), { immediate: true });

// `/api/models` covers built-in ControlNet manifests, while ControlNet weights
// pulled from Catalog are discoverable through the installed-sidecar endpoint.
// Fetch against the explicitly selected mobile host so a path from one remote
// machine can never leak into another host's generation request. Including the
// model in the key also invalidates an in-flight response when a same-family
// model selection changes. The cleanup token makes those invalidations race
// safe, including component unmounts.
let controlCatalogFetchToken = 0;
watch(
  () =>
    [
      props.form.family,
      props.form.model,
      props.target?.baseUrl ?? "",
      props.target?.apiKey ?? "",
    ] as const,
  async ([family], _previous, onCleanup) => {
    const token = ++controlCatalogFetchToken;
    onCleanup(() => {
      if (controlCatalogFetchToken === token) controlCatalogFetchToken += 1;
    });
    catalogControlModels.value = [];
    const target = props.target;
    if (!target || !generationCapabilitiesForFamily(family).supportsControlNet) return;

    try {
      const response = await fetchCatalogInstalled({ family, kind: "control-net" }, target);
      if (token !== controlCatalogFetchToken) return;
      catalogControlModels.value = response.entries.filter(
        (entry) => entry.installed && entry.primary_path,
      );
    } catch {
      if (token === controlCatalogFetchToken) catalogControlModels.value = [];
    }
  },
  { immediate: true },
);

watch(
  () => [caps.value.supportsMask, props.form.sourceImage] as const,
  ([supportsMask, source]) => {
    if (!supportsMask && source) {
      props.form.sourceFit = coerceSourceFitForMaskless(props.form.sourceFit);
    }
  },
  { immediate: true },
);

watch(
  () => props.form.controlModel,
  (value) => {
    if (controlCustomMode.value && value.trim() === "") return;
    if (controlOptions.value.some((option) => option.value === value)) {
      controlCustomMode.value = false;
    }
  },
);

watch(
  () =>
    [
      props.form.controlImage,
      controlOptions.value
        .filter((option) => !option.disabled)
        .map((option) => option.value)
        .join("\u0000"),
    ] as const,
  ([controlImage]) => {
    if (!controlImage || props.form.controlModel.trim()) return;
    const firstAvailable = controlOptions.value.find((option) => !option.disabled);
    if (firstAvailable) props.form.controlModel = firstAvailable.value;
  },
  { immediate: true },
);

function isAcceptedImage(file: File): boolean {
  return (
    file.type === "image/png" ||
    file.type === "image/jpeg" ||
    (!file.type && isStillImageFile(file.name))
  );
}

async function readImages(
  event: Event,
  multiple: boolean,
  replacing: InlineGenerationMediaField | null = null,
): Promise<Array<{ file: File; b64: string }>> {
  const input = event.target as HTMLInputElement;
  const files = Array.from(input.files ?? []);
  input.value = "";
  if (files.length === 0) return [];
  if (files.some((file) => file.size === 0)) {
    error.value = "Empty photos can’t be used here.";
    return [];
  }
  if (
    inlineGenerationMediaBytes(props.form, replacing) +
      files.reduce((sum, file) => sum + file.size, 0) >
    MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES
  ) {
    error.value = "Combined generation media must be 45 MiB or smaller on this phone.";
    return [];
  }
  if (files.some((file) => !isAcceptedImage(file))) {
    error.value = "Only PNG or JPEG photos can be used here.";
    return [];
  }

  error.value = "";
  const selected = multiple ? files : files.slice(0, 1);
  try {
    return await Promise.all(
      selected.map(async (file) => ({ file, b64: await fileToBase64(file) })),
    );
  } catch {
    error.value = "Couldn’t read that photo. Try choosing it again.";
    return [];
  }
}

function pickSource(image: MobilePickedImage): void {
  error.value = "";
  props.form.sourceImage = image.base64;
  props.form.sourceImageName = image.filename || null;
  props.form.sourceFit = defaultSourceFitPolicy();
  sourcePickerOpen.value = false;
}

function replaceEditTarget(base64: string): void {
  props.form.imageAttachments = [base64, ...props.form.imageAttachments.slice(1)];
  props.form.sourceFit = defaultSourceFitPolicy();
}

function pickEditTarget(image: MobilePickedImage): void {
  error.value = "";
  replaceEditTarget(image.base64);
  sourcePickerOpen.value = false;
}

async function onEditTargetFile(_slot: SourceMediaSlot, file: File): Promise<void> {
  if (!isAcceptedImage(file)) {
    error.value = "Only PNG or JPEG photos can be used here.";
    return;
  }
  if (file.size > sourcePickerMaxBytes.value) {
    error.value = MOBILE_MEDIA_BUDGET_ERROR;
    return;
  }
  try {
    replaceEditTarget(await fileToBase64(file));
    error.value = "";
  } catch {
    error.value = "Couldn’t read that photo. Try choosing it again.";
  }
}

function clearEditTarget(): void {
  props.form.imageAttachments = props.form.imageAttachments.slice(1);
}

async function onSingleSourceFile(slot: SourceMediaSlot, file: File): Promise<void> {
  if (!isAcceptedImage(file)) {
    error.value = "Only PNG or JPEG photos can be used here.";
    return;
  }
  const maxBytes = slot === "source" ? sourcePickerMaxBytes.value : endFramePickerMaxBytes.value;
  if (file.size > maxBytes) {
    error.value = MOBILE_MEDIA_BUDGET_ERROR;
    return;
  }
  try {
    const base64 = await fileToBase64(file);
    error.value = "";
    if (slot === "source") {
      props.form.sourceImage = base64;
      props.form.sourceImageName = file.name;
      props.form.sourceFit = defaultSourceFitPolicy();
      // Last write wins on an exclusive recipe: the references park, kept.
      props.form.exclusiveWell = "source";
    } else {
      props.form.endFrame = { filename: file.name, base64 };
    }
  } catch {
    error.value = "Couldn’t read that photo. Try choosing it again.";
  }
}

function openSingleSourcePicker(slot: SourceMediaSlot): void {
  if (slot === "source") sourcePickerOpen.value = true;
  else endFramePickerOpen.value = true;
}

function clearSingleSource(slot: SourceMediaSlot): void {
  if (slot === "source") removeSource();
  else removeEndFrame();
}

/** The closing still of a wan first/last-frame render (#779). It keeps its own
 * name because that name — with the digest — is all saved metadata will ever
 * hold of it. */
function pickEndFrame(image: MobilePickedImage): void {
  error.value = "";
  props.form.endFrame = { filename: image.filename, base64: image.base64 };
  endFramePickerOpen.value = false;
}

function removeEndFrame(): void {
  props.form.endFrame = null;
}

async function pickEditImages(event: Event): Promise<void> {
  const picked = await readImages(event, true);
  if (picked.length === 0) return;
  const establishesTarget =
    plan.value.kind === "attachments" &&
    plan.value.primary === "target" &&
    props.form.imageAttachments.length === 0;
  const next = [...props.form.imageAttachments, ...picked.map((image) => image.b64)];
  props.form.imageAttachments =
    referenceMax.value === null ? next : next.slice(0, referenceMax.value);
  props.form.exclusiveWell = "references";
  if (establishesTarget) props.form.sourceFit = defaultSourceFitPolicy();
}

async function pickMask(event: Event): Promise<void> {
  const picked = (await readImages(event, false, "maskImage"))[0];
  if (picked) props.form.maskImage = picked.b64;
}

async function pickControl(event: Event): Promise<void> {
  const picked = (await readImages(event, false, "controlImage"))[0];
  if (!picked) return;
  props.form.controlImage = picked.b64;
  if (!props.form.controlModel.trim()) {
    const firstAvailable = controlOptions.value.find((option) => !option.disabled);
    if (firstAvailable) props.form.controlModel = firstAvailable.value;
  }
}

function removeSource(): void {
  props.form.sourceImage = null;
  props.form.sourceImageName = null;
  // A mask is defined in the source image's coordinate space; retaining it
  // after the source disappears would silently apply stale pixels later.
  props.form.maskImage = null;
}

function removeEditImage(index: number): void {
  const next = props.form.imageAttachments.slice();
  next.splice(index, 1);
  props.form.imageAttachments = next;
}

function moveEditImage(index: number, delta: -1 | 1): void {
  props.form.imageAttachments = moveAttachment(props.form.imageAttachments, index, delta);
}

function setSourceFit(event: Event): void {
  props.form.sourceFit = sourceFitPolicyForMode(
    (event.target as HTMLSelectElement).value as SourceFitMode,
    {
      supportsMask: caps.value.supportsMask,
      upscalerModel: props.form.upscaleModel || props.upscalers[0]?.name || "",
    },
  );
}

function setControlModel(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  if (value === CUSTOM_CONTROL_MODEL) {
    controlCustomMode.value = true;
    return;
  }
  controlCustomMode.value = false;
  props.form.controlModel = value;
}

function removeControl(): void {
  props.form.controlImage = null;
  props.form.controlModel = "";
  controlCustomMode.value = false;
}

function applyMask(mask: string): void {
  props.form.maskImage = mask;
  maskOpen.value = false;
}
</script>

<template>
  <template v-if="plan.kind === 'h3-references'">
    <fieldset class="mobile-source-controls" data-test="mobile-h3-authoring">
      <legend class="mobile-source-legend">Ordered references · Required</legend>
      <MinimaxH3AuthoringPanel
        :model-value="h3Authoring"
        touch-friendly
        image-picker-available
        @update:model-value="setH3Authoring"
        @open-image-picker="h3ReferencePickerOpen = true"
        @crop-reference="h3CropIndex = $event"
      />
    </fieldset>
    <MobileReferenceCropSheet
      :open="h3CropTarget !== null"
      :title="`Crop reference ${(h3CropIndex ?? 0) + 1}`"
      :image="h3CropTarget?.image ?? null"
      :crop="h3CropTarget?.crop ?? null"
      @apply="applyH3ReferenceCrop"
      @close="h3CropIndex = null"
    />
    <MobileImagePickerSheet
      :open="h3ReferencePickerOpen"
      :target="target"
      :gallery-sources="gallerySources"
      title="Add ordered reference images"
      multiple
      :max-bytes="h3ReferencePickerMaxBytes"
      :oversize-message="MOBILE_MEDIA_BUDGET_ERROR"
      @pick-many="onH3ReferenceImagesPicked"
      @close="h3ReferencePickerOpen = false"
    />
  </template>

  <!-- MiniMax H3 FL2VA boundaries: the exact same wells, H3-owned state. -->
  <template v-else-if="plan.kind === 'h3-boundaries'">
    <fieldset class="mobile-source-controls" data-test="mobile-h3-boundaries">
      <SourceMediaWells
        :plan="plan"
        touch-friendly
        :source="h3Authoring.firstFrame"
        :end-frame="h3Authoring.lastFrame"
        :error="h3Error"
        @file="onH3File"
        @gallery="onH3Gallery"
        @clear="onH3Clear"
      />
      <!-- The same client-side fit as an ordinary source, coerced maskless
           and applied to both boundaries at submit. -->
      <label v-if="h3Authoring.firstFrame || h3Authoring.lastFrame" class="field">
        <span>Source fit</span>
        <select
          class="control"
          :value="coerceSourceFitForMaskless(form.sourceFit).mode"
          data-test="mobile-h3-source-fit"
          @change="setSourceFit"
        >
          <option
            v-for="option in MASKLESS_SOURCE_FIT_OPTIONS"
            :key="option.value"
            :value="option.value"
          >
            {{ option.label }}
          </option>
        </select>
      </label>
    </fieldset>
    <MobileImagePickerSheet
      :open="h3PickerTarget !== null"
      :target="target"
      :gallery-sources="gallerySources"
      :title="h3PickerTarget === 'lastFrame' ? 'Last frame' : 'First frame'"
      :max-bytes="h3PickerMaxBytes"
      :oversize-message="MOBILE_MEDIA_BUDGET_ERROR"
      @pick="onH3Picked"
      @close="h3PickerTarget = null"
    />
  </template>

  <template
    v-else-if="
      plan.kind === 'attachments' || plan.kind === 'single' || plan.kind === 'single-or-references'
    "
  >
    <fieldset v-if="showStrip" class="mobile-source-controls" data-test="mobile-source-controls">
      <legend class="mobile-source-legend">{{ referencesOnly ? "References" : "Pictures" }}</legend>
      <SourceMediaWells
        v-if="plan.kind === 'attachments' && plan.primary === 'target'"
        :plan="plan"
        touch-friendly
        :source="form.imageAttachments[0] ? { data: form.imageAttachments[0] } : null"
        @file="onEditTargetFile"
        @gallery="sourcePickerOpen = true"
        @clear="clearEditTarget"
      />
      <label
        v-if="plan.kind === 'attachments' && plan.primary === 'target' && form.imageAttachments[0]"
        class="field"
      >
        <span>Source fit</span>
        <select
          class="control"
          :value="editFitMode"
          data-test="mobile-source-fit"
          @change="setSourceFit"
        >
          <option
            v-for="option in MASKLESS_SOURCE_FIT_OPTIONS"
            :key="option.value"
            :value="option.value"
          >
            {{ option.label }}
          </option>
        </select>
      </label>
      <p class="mobile-source-note" data-test="mobile-source-fit-help">
        {{ sourceFitHelp(editFitMode) }} Qwen conditioning limit: {{ sourceLimitLabel }} from this
        model; Output size is separate.
      </p>
      <p class="mobile-source-note">
        {{
          referencesOnly
            ? referenceMax === null
              ? "Optional ordered references. Their order is preserved."
              : `Add up to  optional references. Their order is preserved.`
            : "The first picture is the edit Target. Additional pictures are References."
        }}
      </p>
      <p
        v-if="exclusive?.parked === 'references'"
        class="mobile-source-note"
        data-test="mobile-references-parked-note"
      >
        {{ exclusive.note }}
      </p>
      <p
        v-if="validationError"
        class="mobile-source-error"
        role="alert"
        data-test="mobile-source-validation"
      >
        {{ validationError }}
      </p>

      <input
        ref="editInput"
        hidden
        type="file"
        accept="image/png,image/jpeg"
        multiple
        data-test="mobile-edit-input"
        tabindex="-1"
        @change="pickEditImages"
      />
      <button
        type="button"
        class="secondary-button mobile-source-pick"
        data-test="mobile-edit-add"
        @click="editInput?.click()"
      >
        Add photos
      </button>

      <div
        v-if="form.imageAttachments.length"
        class="mobile-attachment-grid"
        data-test="mobile-edit-grid"
      >
        <article
          v-for="(image, index) in form.imageAttachments"
          :key="`${index}-${image.slice(0, 16)}`"
          class="mobile-attachment-card"
          :data-test="`mobile-edit-card-${index}`"
        >
          <img
            :src="base64ToDataUrl(image)"
            :alt="`${attachmentRoleLabel(index)} ${attachmentTitleLabel(index)}`"
          />
          <div class="mobile-attachment-copy">
            <strong :data-test="`mobile-edit-role-${index}`">{{
              referencesOnly ? `Reference ${index + 1}` : attachmentRoleLabel(index)
            }}</strong>
            <span :data-test="`mobile-edit-title-${index}`">{{ attachmentTitleLabel(index) }}</span>
          </div>
          <div
            class="mobile-attachment-actions"
            :aria-label="`${attachmentTitleLabel(index)} actions`"
          >
            <button
              type="button"
              class="mobile-media-tile-action"
              :disabled="index === 0"
              :aria-label="`Move ${attachmentTitleLabel(index)} earlier`"
              :data-test="`mobile-edit-earlier-${index}`"
              @click="moveEditImage(index, -1)"
            >
              ‹
            </button>
            <button
              type="button"
              class="mobile-media-tile-action"
              :disabled="index === form.imageAttachments.length - 1"
              :aria-label="`Move ${attachmentTitleLabel(index)} later`"
              :data-test="`mobile-edit-later-${index}`"
              @click="moveEditImage(index, 1)"
            >
              ›
            </button>
            <button
              type="button"
              class="mobile-media-tile-action is-danger"
              :aria-label="`Remove ${attachmentTitleLabel(index)}`"
              :data-test="`mobile-edit-remove-${index}`"
              @click="removeEditImage(index)"
            >
              Remove
            </button>
          </div>
        </article>
      </div>
    </fieldset>

    <fieldset
      v-if="showSourceWell"
      class="mobile-source-controls"
      data-test="mobile-source-controls"
    >
      <SourceMediaWells
        :plan="plan"
        touch-friendly
        test-id-prefix="mobile-"
        :source="
          form.sourceImage ? { data: form.sourceImage, filename: form.sourceImageName } : null
        "
        :end-frame="
          form.endFrame ? { data: form.endFrame.base64, filename: form.endFrame.filename } : null
        "
        :error="conditioningError"
        @file="onSingleSourceFile"
        @gallery="openSingleSourcePicker"
        @clear="clearSingleSource"
      />
      <template v-if="sourceRefinements && form.sourceImage">
        <!-- Wan pins the first frame exactly and never reads strength. -->
        <label v-if="caps.supportsStrength" class="mobile-range-field">
          <span
            >{{ strength.label }} <output>{{ form.strength.toFixed(2) }}</output></span
          >
          <input
            v-model.number="form.strength"
            type="range"
            min="0.05"
            max="1"
            step="0.05"
            :aria-label="strength.label"
            :title="strength.hint"
            data-test="mobile-source-strength"
          />
        </label>

        <!-- A canvasless (3-D) recipe fits the source to no canvas at all. -->
        <label v-if="!canvasless" class="field">
          <span>Source fit</span>
          <select
            class="control"
            :value="form.sourceFit.mode"
            data-test="mobile-source-fit"
            @change="setSourceFit"
          >
            <option v-if="caps.supportsMask" value="pad-repaint">Pad repaint</option>
            <option value="crop-fill">Crop fill</option>
            <option value="pad-fit">Pad fit</option>
            <option value="lanczos-resize">Lanczos resize</option>
            <option value="upscale-then-fit">Upscale then fit</option>
          </select>
        </label>
        <p
          v-if="!canvasless && form.sourceFit.mode === 'upscale-then-fit'"
          class="mobile-source-note"
        >
          {{
            form.sourceFit.upscalerModel
              ? `Preprocesses with ${form.sourceFit.upscalerModel}.`
              : "No upscaler is available; the source will be fit without upscaling."
          }}
        </p>
      </template>

      <fieldset
        v-if="sourceRefinements && caps.supportsMask && form.sourceImage"
        class="mobile-source-subsection"
      >
        <legend>Mask</legend>
        <p class="mobile-source-note">White repaints. Black preserves the source.</p>
        <input
          ref="maskInput"
          hidden
          type="file"
          accept="image/png,image/jpeg"
          data-test="mobile-mask-input"
          tabindex="-1"
          @change="pickMask"
        />
        <img
          v-if="form.maskImage"
          class="mobile-source-preview is-mask"
          :src="base64ToDataUrl(form.maskImage)"
          alt="Current source mask"
          data-test="mobile-mask-preview"
        />
        <div class="mobile-media-actions">
          <button
            type="button"
            class="secondary-button"
            data-test="mobile-mask-upload"
            @click="maskInput?.click()"
          >
            {{ form.maskImage ? "Replace mask" : "Upload mask" }}
          </button>
          <button
            type="button"
            class="secondary-button"
            data-test="mobile-mask-edit"
            @click="maskOpen = true"
          >
            Edit mask
          </button>
          <button
            v-if="form.maskImage"
            type="button"
            class="secondary-button"
            data-test="mobile-mask-remove"
            @click="form.maskImage = null"
          >
            Remove mask
          </button>
        </div>
      </fieldset>

      <fieldset v-if="caps.supportsControlNet" class="mobile-source-subsection">
        <legend>ControlNet</legend>
        <input
          ref="controlInput"
          hidden
          type="file"
          accept="image/png,image/jpeg"
          data-test="mobile-control-input"
          tabindex="-1"
          @change="pickControl"
        />
        <button
          v-if="!form.controlImage"
          type="button"
          class="secondary-button mobile-source-pick"
          data-test="mobile-control-add"
          @click="controlInput?.click()"
        >
          Choose control photo
        </button>
        <template v-else>
          <img
            class="mobile-source-preview"
            :src="base64ToDataUrl(form.controlImage)"
            alt="ControlNet conditioning photo"
            data-test="mobile-control-preview"
          />
          <div class="mobile-media-actions">
            <button type="button" class="secondary-button" @click="controlInput?.click()">
              Replace photo
            </button>
            <button
              type="button"
              class="secondary-button"
              data-test="mobile-control-remove"
              @click="removeControl"
            >
              Remove
            </button>
          </div>

          <label class="field">
            <span>Control model</span>
            <select
              class="control"
              :value="controlSelectValue"
              data-test="mobile-control-model"
              @change="setControlModel"
            >
              <option value="">None</option>
              <option
                v-for="option in controlOptions"
                :key="option.value"
                :value="option.value"
                :disabled="option.disabled"
              >
                {{ option.label }}
              </option>
              <option :value="CUSTOM_CONTROL_MODEL">Custom…</option>
            </select>
          </label>
          <p
            v-if="validationError"
            class="mobile-source-error"
            role="alert"
            data-test="mobile-source-validation"
          >
            {{ validationError }}
          </p>
          <p v-if="controlOptions.some((option) => option.disabled)" class="mobile-source-note">
            Missing ControlNet models can be pulled from Catalog.
          </p>
          <label v-if="controlCustomMode" class="field">
            <span>Custom model ID</span>
            <input
              v-model="form.controlModel"
              class="control"
              type="text"
              autocapitalize="none"
              autocomplete="off"
              spellcheck="false"
              data-test="mobile-control-custom"
            />
          </label>

          <label class="mobile-range-field">
            <span
              >Control scale <output>{{ form.controlScale.toFixed(2) }}</output></span
            >
            <input
              v-model.number="form.controlScale"
              type="range"
              min="0"
              max="2"
              step="0.05"
              aria-label="ControlNet scale"
              data-test="mobile-control-scale"
            />
          </label>
        </template>
      </fieldset>
    </fieldset>

    <p v-if="error" class="mobile-source-error" role="alert" data-test="mobile-source-error">
      {{ error }}
    </p>

    <MaskEditorModal
      :open="maskOpen"
      :source="form.sourceImage"
      :filename="form.sourceImageName"
      :initial-mask="form.maskImage"
      @apply="applyMask"
      @close="maskOpen = false"
    />
    <MobileImagePickerSheet
      v-if="!isAttachmentMode || (plan.kind === 'attachments' && plan.primary === 'target')"
      :open="sourcePickerOpen"
      :target="target"
      :gallery-sources="gallerySources"
      :title="isAttachmentMode ? 'Edit target' : 'Source image'"
      :max-bytes="sourcePickerMaxBytes"
      :oversize-message="MOBILE_MEDIA_BUDGET_ERROR"
      @pick="isAttachmentMode ? pickEditTarget($event) : pickSource($event)"
      @close="sourcePickerOpen = false"
    />
    <MobileImagePickerSheet
      v-if="!isAttachmentMode && caps.supportsEndFrame"
      :open="endFramePickerOpen"
      :target="target"
      :gallery-sources="gallerySources"
      title="End frame"
      :max-bytes="endFramePickerMaxBytes"
      :oversize-message="MOBILE_MEDIA_BUDGET_ERROR"
      @pick="pickEndFrame"
      @close="endFramePickerOpen = false"
    />
  </template>
</template>
