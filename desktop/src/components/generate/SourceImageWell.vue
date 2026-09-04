<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { GenerateForm, PickedImage } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";
import { generationCapabilitiesForFamily } from "../../lib/capabilities";
import { base64ToDataUrl, fileToBase64, isStillImageFile } from "../../lib/image";
import {
  attachmentRoleLabel,
  attachmentTitleLabel,
  moveAttachment,
  reorderAttachment,
} from "../../lib/editAttachments";
import { buildControlNetOptions } from "../../lib/controlNetOptions";
import { sourceConditioningValidationError } from "../../lib/generateValidation";
import { fetchCatalogInstalled } from "../../lib/api/catalog";
import type { CatalogEntry } from "../../lib/api/types";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useModelStore } from "../../stores/models";
import { useToastStore } from "../../stores/toasts";
import { attachPickedImage } from "../../lib/sourceAttachment";
import ImageDropWell from "@studio/components/ImageDropWell.vue";
import SourceMediaWells, { type SourceMediaSlot } from "@studio/components/SourceMediaWells.vue";
import MinimaxH3AuthoringPanel from "@studio/components/MinimaxH3AuthoringPanel.vue";
import { resolveExclusiveWells, sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import type { ReferenceCrop } from "@studio/lib/referenceCrop";
import SliderRow from "@ui/components/SliderRow.vue";
import { strengthSemantics } from "@studio/lib/strengthSemantics";
import { sourceConditioningLimitLabel } from "@studio/lib/sourceResolution";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import {
  coerceSourceFitForMaskless,
  defaultSourceFitPolicy,
  MASKLESS_SOURCE_FIT_OPTIONS,
  sourceFitHelp,
  sourceFitPolicyForMode,
  type SourceFitMode,
} from "@studio/lib/sourceFit";
import {
  appendMinimaxH3PickedImageReferences,
  emptyMinimaxH3AuthoringState,
  minimaxH3ReferenceCropTarget,
  setMinimaxH3BoundaryFile,
  setMinimaxH3ReferenceCrop,
  setMinimaxH3PickedImageBoundary,
  type MinimaxH3BoundaryEndpoint,
  type MinimaxH3GalleryImageResult,
} from "@studio/lib/minimaxH3Authoring";
import ImagePickerModal from "./ImagePickerModal.vue";
import MaskEditorModal from "./MaskEditorModal.vue";
import ReferenceCropModal from "./ReferenceCropModal.vue";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    /**
     * The row that answers for the CHECKPOINT'S CONTRACT; its advertised
     * recipe wins over the form's snapshot so this component and its mount
     * gate share one derivation.
     *
     * The parent resolves it from whichever machine has the checkpoint, not
     * only from the one Create is aimed at — otherwise aiming at a machine
     * that must download it first put a Denoise slider and an Edit-mask
     * control on a 3-D print. `null` means no machine has it, and the family
     * rules answer instead.
     */
    selectedModel?: ModelEntry | null;
  }>(),
  { selectedModel: null },
);
const toasts = useToastStore();
const models = useModelStore();
const appPrefs = useAppPrefsStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();

// The selected checkpoint's own advertised source-image contract (#772) rides
// as the fifth argument: wan's three checkpoints split T2V / I2V-optional /
// I2V-required and only the server can tell them apart. Never read
// `source_image` (or a family set) directly here — the helper owns the
// absent-field fallback that keeps older servers on today's behaviour.
const caps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    props.selectedModel?.guidance_capabilities ?? props.form.guidanceCapabilities,
    props.selectedModel?.source_image ?? props.form.sourceImageCapability,
    // The recipe is the authority on strength, the repaint mask, and whether
    // there is a canvas to fit onto at all. Without it the pre-profile family
    // rules answer, and they say a 3-D mesh takes all three — so a Hunyuan3D
    // source would show a denoise slider, a mask well, and a fit policy the
    // request cannot carry.
    effectiveGenerationRecipe(props.selectedModel, props.form.pipeline),
  ),
);
// Family-scoped label for the shared `strength` wire field (#1055).
const strength = computed(() => strengthSemantics(props.form.family));
/** The model's own image-attachment shape — the single policy every surface
 * renders (`@studio/lib/sourceMediaPlan`). */
const plan = computed(() => sourceMediaPlan(caps.value));
/** A strip with no primary Target is a pure reference strip (FLUX.2), which
 * changes only the wording — the ceiling and the roles come from the plan. */
const referencesOnly = computed(
  () =>
    (plan.value.kind === "attachments" && plan.value.primary === null) ||
    plan.value.kind === "single-or-references",
);
/** The advertised strip ceiling; `null` is unbounded (Qwen edit). */
const referenceMax = computed(() =>
  plan.value.kind === "attachments"
    ? plan.value.max
    : plan.value.kind === "single-or-references"
      ? plan.value.references.max
      : null,
);
/**
 * The exclusive (Klein) parking rule: whichever well holds media is the
 * active one, the other parks with an inline note and keeps its media, and
 * Generate stays enabled. Strength and the repaint mask belong to the source
 * well, so they render only while it is active.
 */
const exclusive = computed(() =>
  plan.value.kind === "single-or-references"
    ? resolveExclusiveWells({
        hasSource: Boolean(props.form.sourceImage),
        referenceCount: props.form.imageAttachments.length,
        lastWrite: props.form.exclusiveWell ?? null,
      })
    : null,
);
const sourceActive = computed(() => !exclusive.value || exclusive.value.active !== "references");
/** Qwen's layout: the strip's first picture IS the primary (Target) well. */
const targetLayout = computed(
  () => plan.value.kind === "attachments" && plan.value.primary === "target",
);
/** What the shared primary well shows — the Target for a target-first strip,
 * the source image everywhere else. */
const primaryWellImage = computed(() =>
  targetLayout.value
    ? props.form.imageAttachments[0]
      ? { data: props.form.imageAttachments[0] }
      : null
    : props.form.sourceImage
      ? { data: props.form.sourceImage, filename: props.form.sourceImageName }
      : null,
);
/** Fit, strength and the repaint mask describe a SOURCE image. */
const sourceRefinements = computed(
  () =>
    plan.value.kind === "single" ||
    (plan.value.kind === "single-or-references" && sourceActive.value),
);
/** Why the attached conditioning would be refused, in the server's own order. */
const conditioningError = computed(() => sourceConditioningValidationError(props.form));
const editFitMode = computed(() => coerceSourceFitForMaskless(props.form.sourceFit).mode);
const sourceLimitLabel = computed(() =>
  sourceConditioningLimitLabel(props.selectedModel ?? props.form.family, props.form.pipeline),
);

const pickerOpen = ref(false);
const endPickerOpen = ref(false);
const maskOpen = ref(false);
/** Whether this recipe and this attachment can take a painted mask at all —
 * the one answer behind the group's "Paint a mask" door. */
const maskAvailable = computed(
  () => sourceRefinements.value && caps.value.supportsMask && Boolean(props.form.sourceImage),
);
function openMaskEditor() {
  if (maskAvailable.value) maskOpen.value = true;
}
defineExpose({ maskAvailable, openMaskEditor });
const h3ReferencePickerOpen = ref(false);
/** Which ordered reference the crop dialog is editing; null when closed. */
const h3CropIndex = ref<number | null>(null);
const h3CropTarget = computed(() =>
  minimaxH3ReferenceCropTarget(props.form.h3Authoring, h3CropIndex.value),
);
function applyH3ReferenceCrop(crop: ReferenceCrop | null) {
  if (h3CropIndex.value === null) return;
  props.form.h3Authoring = setMinimaxH3ReferenceCrop(
    props.form.h3Authoring ?? emptyMinimaxH3AuthoringState(),
    h3CropIndex.value,
    crop,
  );
  h3CropIndex.value = null;
}

function onSourcePicked(picked: PickedImage[]) {
  const first = picked[0];
  if (first) attachPickedImage(props.form, first);
}
function onEndFramePicked(picked: PickedImage[]) {
  const first = picked[0];
  if (first) setSlot("end", first.base64, first.filename || null);
}
function onMaskApplied(mask: string) {
  props.form.maskImage = mask;
}

async function onH3ReferenceImagesPicked(picked: PickedImage[]) {
  const result = await appendMinimaxH3PickedImageReferences(props.form.h3Authoring, picked);
  applyH3(result);
}

// ── Qwen-edit Target + Reference strip ──────────────────────────────────────
// Ordered base64 attachments: index 0 is the primary edit Target, the rest
// are References (web Composer parity — the order ships as `edit_images`).

const editPickerOpen = ref(false);
const targetPickerOpen = ref(false);
const dragIndex = ref<number | null>(null);

function onEditPicked(picked: PickedImage[]) {
  if (picked.length === 0) return;
  const establishesTarget =
    plan.value.kind === "attachments" &&
    plan.value.primary === "target" &&
    props.form.imageAttachments.length === 0;
  const next = [...props.form.imageAttachments, ...picked.map((p) => p.base64)];
  const max = referenceMax.value;
  props.form.imageAttachments = max === null ? next : next.slice(0, max);
  // On an exclusive recipe this write parks the source well; the source
  // itself is kept and comes back when the strip empties.
  props.form.exclusiveWell = "references";
  if (establishesTarget) props.form.sourceFit = defaultSourceFitPolicy();
}
function replaceEditTarget(base64: string) {
  props.form.imageAttachments = [base64, ...props.form.imageAttachments.slice(1)];
  props.form.sourceFit = defaultSourceFitPolicy();
}
function onTargetPicked(picked: PickedImage[]) {
  if (picked[0]) replaceEditTarget(picked[0].base64);
}
async function onTargetFile(_slot: SourceMediaSlot, file: File) {
  if (
    file.type !== "image/png" &&
    file.type !== "image/jpeg" &&
    !(!file.type && isStillImageFile(file.name))
  ) {
    toasts.push("Only PNG or JPEG images can be used here.", "error");
    return;
  }
  try {
    replaceEditTarget(await fileToBase64(file));
  } catch {
    toasts.push("Couldn't read the image.", "error");
  }
}
function clearEditTarget() {
  props.form.imageAttachments = props.form.imageAttachments.slice(1);
}
function removeAttachmentAt(index: number) {
  const next = props.form.imageAttachments.slice();
  next.splice(index, 1);
  props.form.imageAttachments = next;
}
function moveAttachmentBy(index: number, delta: -1 | 1) {
  props.form.imageAttachments = moveAttachment(props.form.imageAttachments, index, delta);
}
function onTileDragStart(index: number, event: DragEvent) {
  dragIndex.value = index;
  event.dataTransfer?.setData("text/plain", String(index));
  if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
}
function onTileDrop(index: number, event: DragEvent) {
  const raw = event.dataTransfer?.getData("text/plain");
  const from = raw && !Number.isNaN(Number(raw)) ? Number(raw) : dragIndex.value;
  dragIndex.value = null;
  if (from == null) return;
  props.form.imageAttachments = reorderAttachment(props.form.imageAttachments, from, index);
}

// ── ControlNet model picker ─────────────────────────────────────────────────
// Installed `controlnet`-family models (scoped to the pinned generation host's
// inventory when one is known) ∪ catalog-installed control-net entries, with a
// Custom… escape hatch that reveals the legacy free-text input.

const CUSTOM_CONTROL_MODEL = "__custom__";
const controlCustomMode = ref(false);
const catalogControlNet = ref<CatalogEntry[]>([]);

/** The sticky generation host (an explicit pick, not Auto / Most capable). */
const stickyHost = computed(() => {
  const sel = appPrefs.settings?.generateTargetHost ?? null;
  if (!sel || sel === "capable") return null;
  return hosts.all.find((h) => h.id === sel && h.status === "ready" && h.baseUrl) ?? null;
});

/** Installed-model source: the pinned host's inventory when known, else the
 * primary's canonical list. "Known" is a successful fetch (`fetchedAt > 0`),
 * not a non-empty list — a host that genuinely has no models must scope to its
 * own (empty) inventory rather than borrow the primary's. */
const installedControlNetSource = computed(() => {
  const host = stickyHost.value;
  if (host && !host.primary) {
    const record = hostModels.byHost[host.id];
    if (record && record.fetchedAt > 0) return record.entries;
  }
  return models.all;
});

const controlNetOptions = computed(() =>
  buildControlNetOptions(
    installedControlNetSource.value,
    catalogControlNet.value,
    controlCustomMode.value ? "" : props.form.controlModel,
  ),
);

const controlSelectValue = computed(() =>
  controlCustomMode.value ? CUSTOM_CONTROL_MODEL : props.form.controlModel,
);

function onControlModelChange(e: Event) {
  const value = (e.target as HTMLSelectElement).value;
  if (value === CUSTOM_CONTROL_MODEL) {
    // Keep the current text so the user can edit rather than retype it.
    controlCustomMode.value = true;
    return;
  }
  controlCustomMode.value = false;
  props.form.controlModel = value;
}

// Refetch catalog-installed control-net entries when the family or the pinned
// host changes; a token guards against stale async writes.
let controlFetchToken = 0;
watch(
  () => [props.form.family, stickyHost.value?.id] as const,
  async ([family]) => {
    const token = ++controlFetchToken;
    catalogControlNet.value = [];
    if (!generationCapabilitiesForFamily(family).supportsControlNet) return;
    try {
      const host = stickyHost.value;
      const target = host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : undefined;
      const res = await fetchCatalogInstalled({ family, kind: "control-net" }, target);
      if (token !== controlFetchToken) return;
      catalogControlNet.value = res.entries.filter((e) => e.installed && e.primary_path);
    } catch {
      if (token === controlFetchToken) catalogControlNet.value = [];
    }
  },
  { immediate: true },
);

type Slot = "source" | "end" | "mask" | "control";

function setSlot(slot: Slot, b64: string | null, name: string | null = null) {
  if (slot === "source") {
    props.form.sourceImage = b64;
    // The label lives and dies with the image (Reuse-settings restore).
    props.form.sourceImageName = b64 ? name : null;
    if (b64) {
      props.form.sourceFit = defaultSourceFitPolicy();
      // Last write wins on an exclusive recipe: this parks the references
      // without discarding them.
      props.form.exclusiveWell = "source";
    }
  } else if (slot === "end") {
    // The closing still keeps its own name: it ships as the second keyframe,
    // whose provenance is all saved metadata will ever hold of it.
    props.form.endFrame = b64 ? { filename: name ?? "", base64: b64 } : null;
  } else if (slot === "mask") props.form.maskImage = b64;
  else props.form.controlImage = b64;
}
async function ingest(slot: Slot, file: File | undefined | null) {
  if (!file) return;
  // Same constraint as the picker modal: the engine only accepts PNG/JPEG
  // for source_image / mask / keyframes — dropped files bypass the input's
  // accept filter, so gate by MIME with a filename fallback.
  if (
    file.type !== "image/png" &&
    file.type !== "image/jpeg" &&
    !(!file.type && isStillImageFile(file.name))
  ) {
    toasts.push("Only PNG or JPEG images can be used here.", "error");
    return;
  }
  try {
    setSlot(slot, await fileToBase64(file), file.name || null);
  } catch {
    toasts.push("Couldn't read the image.", "error");
  }
}
function clearSlot(slot: Slot) {
  setSlot(slot, null);
}

function onWellFile(slot: SourceMediaSlot, file: File) {
  void ingest(slot === "source" ? "source" : "end", file);
}
function onWellGallery(slot: SourceMediaSlot) {
  if (slot === "source") pickerOpen.value = true;
  else endPickerOpen.value = true;
}
function onWellClear(slot: SourceMediaSlot) {
  clearSlot(slot === "source" ? "source" : "end");
}

// The primary well is ONE component for every layout; only what a write means
// differs — a target-first strip edits attachment 0, everything else the
// source image.
function onPrimaryFile(slot: SourceMediaSlot, file: File) {
  if (targetLayout.value && slot === "source") void onTargetFile(slot, file);
  else onWellFile(slot, file);
}
function onPrimaryGallery(slot: SourceMediaSlot) {
  if (targetLayout.value && slot === "source") targetPickerOpen.value = true;
  else onWellGallery(slot);
}
function onPrimaryClear(slot: SourceMediaSlot) {
  if (targetLayout.value && slot === "source") clearEditTarget();
  else onWellClear(slot);
}

// ── MiniMax H3 FL2VA boundaries ─────────────────────────────────────────────
// The same two wells, backed by the dedicated H3 authoring state and the
// shared studio setters so a file and a gallery pick produce identical facts.

const h3Authoring = computed(() => props.form.h3Authoring ?? emptyMinimaxH3AuthoringState());
function setH3Authoring(value: typeof h3Authoring.value): void {
  props.form.h3Authoring = value;
}
const h3PickerTarget = ref<MinimaxH3BoundaryEndpoint | null>(null);
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
  props.form.h3Authoring = result.state;
}
async function onH3File(slot: SourceMediaSlot, file: File) {
  applyH3(await setMinimaxH3BoundaryFile(props.form.h3Authoring, h3Endpoint(slot), file));
}
function onH3Gallery(slot: SourceMediaSlot) {
  h3PickerTarget.value = h3Endpoint(slot);
}
function onH3Clear(slot: SourceMediaSlot) {
  h3Error.value = null;
  props.form.h3Authoring = { ...h3Authoring.value, [h3Endpoint(slot)]: null };
}
function onH3Picked(images: PickedImage[]) {
  const endpoint = h3PickerTarget.value;
  const image = images[0];
  if (!endpoint || !image) return;
  applyH3(setMinimaxH3PickedImageBoundary(props.form.h3Authoring, endpoint, image));
}

/** Port of the web SPA's `setSourceFitPolicy` mode→policy mapping. */
function setSourceFitMode(e: Event) {
  props.form.sourceFit = sourceFitPolicyForMode(
    (e.target as HTMLSelectElement).value as SourceFitMode,
    {
      supportsMask: caps.value.supportsMask,
      upscalerModel: props.form.upscaleModel || models.upscalers[0]?.name || "",
    },
  );
}
</script>

<template>
  <div v-if="plan.kind === 'h3-references'" data-test="h3-reference-controls">
    <div class="mb-2 flex items-center gap-2">
      <span class="font-mono text-micro text-fg-dim whitespace-nowrap">Ordered references</span>
      <div class="border-border h-px flex-1 border-t" />
    </div>
    <MinimaxH3AuthoringPanel
      :model-value="h3Authoring"
      image-picker-available
      @update:model-value="setH3Authoring"
      @open-image-picker="h3ReferencePickerOpen = true"
      @crop-reference="h3CropIndex = $event"
    />
    <ReferenceCropModal
      :open="h3CropTarget !== null"
      :title="`Crop reference ${(h3CropIndex ?? 0) + 1}`"
      :image="h3CropTarget?.image ?? null"
      :crop="h3CropTarget?.crop ?? null"
      @apply="applyH3ReferenceCrop"
      @close="h3CropIndex = null"
    />
    <ImagePickerModal
      :open="h3ReferencePickerOpen"
      :multiple="true"
      title="Add ordered reference images"
      @pick="onH3ReferenceImagesPicked"
      @close="h3ReferencePickerOpen = false"
    />
  </div>

  <!-- One source well, an ordered picture strip, or (Klein) both, mutually
       exclusive. The wells and the strip are the SAME ones every other
       layout renders; the plan decides which of them appear. -->
  <div
    v-else-if="
      plan.kind === 'attachments' || plan.kind === 'single' || plan.kind === 'single-or-references'
    "
    :data-test="plan.kind === 'attachments' ? undefined : 'source-media-controls'"
  >
    <SourceMediaWells
      v-if="plan.kind !== 'attachments' || plan.primary === 'target'"
      :plan="plan"
      :source="primaryWellImage"
      :end-frame="
        !targetLayout && form.endFrame
          ? { data: form.endFrame.base64, filename: form.endFrame.filename }
          : null
      "
      :error="conditioningError"
      :parked="exclusive?.parked === 'source'"
      :note="exclusive?.parked === 'source' ? exclusive.note : null"
      @file="onPrimaryFile"
      @gallery="onPrimaryGallery"
      @clear="onPrimaryClear"
    />
    <template v-if="targetLayout && form.imageAttachments[0]">
      <label class="mt-3 block text-micro text-fg-2" for="edit-source-fit-policy">
        Source fit
      </label>
      <select
        id="edit-source-fit-policy"
        :value="editFitMode"
        data-test="source-fit-policy"
        class="border-border mt-1 h-7 w-full rounded-control border bg-bg-deep px-1.5 text-sm text-fg"
        @change="setSourceFitMode"
      >
        <option
          v-for="option in MASKLESS_SOURCE_FIT_OPTIONS"
          :key="option.value"
          :value="option.value"
        >
          {{ option.label }}
        </option>
      </select>
      <p class="mt-1 text-micro text-fg-dim" data-test="source-fit-help">
        {{ sourceFitHelp(editFitMode) }} Qwen conditioning limit: {{ sourceLimitLabel }} from this
        model; Output size is separate.
      </p>
    </template>
    <!-- The ordered picture strip. Qwen's Target + References, FLUX.2 [dev]'s
         references, and Klein's second (exclusive) well are all THIS strip. -->
    <div
      v-if="plan.kind === 'attachments' || plan.kind === 'single-or-references'"
      class="mb-2 flex items-center gap-2"
      :class="{ 'mt-3': plan.kind === 'single-or-references' }"
    >
      <span class="font-mono text-micro text-fg-dim whitespace-nowrap">{{
        referencesOnly ? "References" : "Pictures"
      }}</span>
      <div class="border-border h-px flex-1 border-t" />
    </div>

    <div
      v-if="plan.kind === 'attachments' || plan.kind === 'single-or-references'"
      class="flex gap-2 overflow-x-auto pb-1"
      data-test="attachment-strip"
      data-drop-target="references"
    >
      <div
        v-for="(image, index) in form.imageAttachments"
        :key="`${index}-${image.slice(0, 16)}`"
        class="relative w-20 shrink-0 overflow-hidden rounded-inner border border-border-control bg-bg-deep"
        draggable="true"
        :data-test="`attachment-card-${index}`"
        @dragstart="onTileDragStart(index, $event)"
        @dragend="dragIndex = null"
        @dragover.prevent
        @drop.prevent="onTileDrop(index, $event)"
      >
        <img
          :src="base64ToDataUrl(image)"
          class="h-12 w-20 object-cover"
          :alt="`${attachmentRoleLabel(index)} ${attachmentTitleLabel(index)}`"
        />
        <div class="px-1.5 py-1 leading-tight">
          <div
            class="font-mono text-micro text-fg-dim whitespace-nowrap"
            :data-test="`attachment-role-${index}`"
          >
            {{ referencesOnly ? `Reference ${index + 1}` : attachmentRoleLabel(index) }}
          </div>
          <div class="truncate text-micro text-fg" :data-test="`attachment-title-${index}`">
            {{ attachmentTitleLabel(index) }}
          </div>
        </div>
        <button
          v-if="index > 0"
          type="button"
          class="absolute top-1 left-1 h-5 w-5 rounded-control bg-bg-deep/90 text-micro text-fg-2 hover:text-fg"
          :aria-label="`Move ${attachmentTitleLabel(index)} left`"
          :data-test="`move-attachment-up-${index}`"
          @click="moveAttachmentBy(index, -1)"
        >
          ‹
        </button>
        <button
          v-if="index < form.imageAttachments.length - 1"
          type="button"
          class="absolute top-1 left-7 h-5 w-5 rounded-control bg-bg-deep/90 text-micro text-fg-2 hover:text-fg"
          :aria-label="`Move ${attachmentTitleLabel(index)} right`"
          :data-test="`move-attachment-down-${index}`"
          @click="moveAttachmentBy(index, 1)"
        >
          ›
        </button>
        <button
          type="button"
          class="border-border absolute top-1 right-1 h-5 w-5 rounded-control border bg-bg-deep text-fg-2 hover:text-error"
          :aria-label="`Remove ${attachmentTitleLabel(index)}`"
          :data-test="`remove-attachment-${index}`"
          @click="removeAttachmentAt(index)"
        >
          ✕
        </button>
      </div>

      <button
        type="button"
        class="flex h-[4.75rem] w-20 shrink-0 cursor-pointer items-center justify-center rounded-inner border border-dashed border-border-control text-base text-fg-dim transition-colors hover:border-accent hover:text-accent focus-visible:outline-2 focus-visible:outline-accent"
        data-test="add-edit-image"
        aria-label="Add pictures"
        @click="editPickerOpen = true"
      >
        ＋
      </button>
    </div>
    <p
      v-if="plan.kind === 'attachments' || plan.kind === 'single-or-references'"
      class="mt-1 text-micro text-fg-dim"
    >
      {{
        referencesOnly
          ? referenceMax === null
            ? "Ordered references. Drag (or ‹ ›) to reorder."
            : `Up to ${referenceMax} ordered references. Drag (or ‹ ›) to reorder.`
          : "First picture is the edit Target; the rest are References. Drag (or ‹ ›) to reorder."
      }}
    </p>
    <!-- The exclusive parking note, on whichever well is not shipping. -->
    <p
      v-if="exclusive?.parked === 'references'"
      class="mt-1 text-micro text-fg-dim"
      data-test="references-parked-note"
    >
      {{ exclusive.note }}
    </p>

    <ImagePickerModal
      v-if="targetLayout"
      :open="targetPickerOpen"
      :multiple="false"
      title="Edit target"
      gallery-only
      @pick="onTargetPicked"
      @close="targetPickerOpen = false"
    />
    <ImagePickerModal
      v-if="plan.kind === 'attachments' || plan.kind === 'single-or-references'"
      :open="editPickerOpen"
      :multiple="true"
      :title="referencesOnly ? 'Add references' : 'Add pictures'"
      @pick="onEditPicked"
      @close="editPickerOpen = false"
    />

    <!-- Source fit (how a mismatched source maps onto the target canvas;
         applied client-side on submit — labels mirror the web SPA). A
         canvasless recipe (a 3-D mesh) has no canvas to fit onto. On an
         exclusive recipe these belong to the Source well, so they render
         only while it is the active one. -->
    <template v-if="sourceRefinements && form.sourceImage && !caps.canvasless">
      <label class="mt-3 block text-micro text-fg-2" for="source-fit-policy">Source fit</label>
      <select
        id="source-fit-policy"
        :value="form.sourceFit?.mode ?? defaultSourceFitPolicy().mode"
        data-test="source-fit-policy"
        class="border-border mt-1 h-7 w-full rounded-control border bg-bg-deep px-1.5 text-sm text-fg"
        @change="setSourceFitMode"
      >
        <option v-if="caps.supportsMask" value="pad-repaint">Pad repaint</option>
        <option value="crop-fill">Crop fill</option>
        <option value="pad-fit">Pad fit</option>
        <option value="lanczos-resize">Lanczos resize</option>
        <option value="upscale-then-fit">Upscale then fit</option>
      </select>
      <p
        v-if="form.sourceFit?.mode === 'upscale-then-fit'"
        class="mt-1 text-micro text-fg-dim"
        data-test="source-fit-upscaler-hint"
      >
        {{
          form.sourceFit.upscalerModel
            ? `Runs ${form.sourceFit.upscalerModel} on the source first`
            : "No upscaler model available — the source is fit without upscaling"
        }}
      </p>
    </template>

    <!-- Strength (wan pins the first frame exactly and never reads it) -->
    <template v-if="sourceRefinements && form.sourceImage && caps.supportsStrength">
      <SliderRow
        class="mt-3"
        :model-value="form.strength"
        :min="0.05"
        :max="1"
        :step="0.05"
        :label="strength.label"
        :value-label="form.strength.toFixed(2)"
        low="Keep the photo"
        high="Start fresh"
        @update:model-value="form.strength = $event"
      />
      <p class="mt-1 text-micro text-fg-dim">{{ strength.hint }}</p>
    </template>

    <!-- Mask well (inpaint families). Painting is opened by the Start-from-a-
         photo group's own door, which asks this component whether it has one. -->
    <template v-if="maskAvailable">
      <label class="mt-3 block text-micro text-fg-2">Mask</label>
      <div class="mt-1">
        <ImageDropWell
          :image="form.maskImage"
          alt="Mask image"
          placeholder="White repaints, black preserves"
          test-id="mask"
          @file="ingest('mask', $event)"
          @clear="clearSlot('mask')"
        />
      </div>
    </template>

    <!-- Control well + model + scale (sd15 only) -->
    <template v-if="caps.supportsControlNet">
      <label class="mt-3 block text-micro text-fg-2">Control image</label>
      <div class="mt-1">
        <ImageDropWell
          :image="form.controlImage"
          alt="Control image"
          placeholder="Drop a control image"
          test-id="control"
          @file="ingest('control', $event)"
          @clear="clearSlot('control')"
        />
      </div>
      <template v-if="form.controlImage">
        <label class="mt-3 block text-micro text-fg-2" for="controlnet-select">
          Control model
        </label>
        <select
          id="controlnet-select"
          data-test="controlnet-select"
          :value="controlSelectValue"
          class="border-border mt-1 h-7 w-full rounded-control border bg-bg-deep px-1.5 text-sm text-fg"
          @change="onControlModelChange"
        >
          <option value="">None</option>
          <option
            v-for="opt in controlNetOptions"
            :key="opt.value"
            :value="opt.value"
            :disabled="opt.disabled"
          >
            {{ opt.label }}
          </option>
          <option :value="CUSTOM_CONTROL_MODEL">Custom…</option>
        </select>
        <p
          v-if="controlNetOptions.some((o) => o.disabled)"
          class="mt-1 text-micro text-fg-dim"
          data-test="controlnet-missing-hint"
        >
          Greyed-out models aren't downloaded yet — pull them from the Catalog.
        </p>
        <input
          v-if="controlCustomMode"
          v-model="form.controlModel"
          data-test="controlnet-custom-input"
          data-selectable
          type="text"
          aria-label="Custom control model id"
          placeholder="controlnet-canny-sd15"
          class="border-border mt-1 h-7 w-full rounded-control border bg-bg-deep px-1.5 text-sm text-fg placeholder:text-fg-dim"
        />
        <label class="mt-3 flex items-center justify-between text-micro text-fg-2">
          Control scale
          <span class="font-mono text-xs text-fg">{{ form.controlScale.toFixed(2) }}</span>
        </label>
        <input
          v-model.number="form.controlScale"
          type="range"
          min="0"
          max="2"
          step="0.05"
          class="mt-1 w-full accent-[var(--mold-blue)]"
        />
      </template>
    </template>

    <ImagePickerModal
      :open="pickerOpen"
      :multiple="false"
      gallery-only
      @pick="onSourcePicked"
      @close="pickerOpen = false"
    />
    <ImagePickerModal
      v-if="caps.supportsEndFrame"
      :open="endPickerOpen"
      :multiple="false"
      title="Pick an end frame"
      gallery-only
      @pick="onEndFramePicked"
      @close="endPickerOpen = false"
    />
    <MaskEditorModal
      :open="maskOpen"
      :source="form.sourceImage"
      :initial-mask="form.maskImage"
      @apply="onMaskApplied"
      @close="maskOpen = false"
    />
  </div>

  <!-- MiniMax H3 FL2VA boundaries: the exact same wells, H3-owned state. -->
  <div v-else-if="plan.kind === 'h3-boundaries'" data-test="source-media-controls">
    <SourceMediaWells
      :plan="plan"
      :source="h3Authoring.firstFrame"
      :end-frame="h3Authoring.lastFrame"
      :error="h3Error"
      @file="onH3File"
      @gallery="onH3Gallery"
      @clear="onH3Clear"
    />
    <!-- The same client-side fit as an ordinary source, coerced maskless and
         applied to both boundaries at submit. -->
    <template v-if="h3Authoring.firstFrame || h3Authoring.lastFrame">
      <label class="mt-3 block text-micro text-fg-2" for="source-fit-policy">Source fit</label>
      <select
        id="source-fit-policy"
        :value="coerceSourceFitForMaskless(form.sourceFit ?? { mode: 'crop-fill' }).mode"
        data-test="source-fit-policy"
        class="border-border mt-1 h-7 w-full rounded-control border bg-bg-deep px-1.5 text-sm text-fg"
        @change="setSourceFitMode"
      >
        <option value="crop-fill">Crop fill</option>
        <option value="pad-fit">Pad fit</option>
        <option value="lanczos-resize">Lanczos resize</option>
        <option value="upscale-then-fit">Upscale then fit</option>
      </select>
    </template>
    <ImagePickerModal
      :open="h3PickerTarget !== null"
      :title="h3PickerTarget === 'lastFrame' ? 'Last frame' : 'First frame'"
      :multiple="false"
      gallery-only
      @pick="onH3Picked"
      @close="h3PickerTarget = null"
    />
  </div>
</template>
