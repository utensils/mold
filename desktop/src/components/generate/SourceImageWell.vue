<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { GenerateForm, PickedImage } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";
import { generationCapabilitiesForFamily, isFlux2DevModel } from "../../lib/capabilities";
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
import { sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import { strengthSemantics } from "@studio/lib/strengthSemantics";
import { coerceSourceFitForMaskless } from "@studio/lib/sourceFit";
import {
  emptyMinimaxH3AuthoringState,
  setMinimaxH3BoundaryFile,
  setMinimaxH3PickedImageBoundary,
  type MinimaxH3BoundaryEndpoint,
  type MinimaxH3GalleryImageResult,
} from "@studio/lib/minimaxH3Authoring";
import ImagePickerModal from "./ImagePickerModal.vue";
import MaskEditorModal from "./MaskEditorModal.vue";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    /** The picked model row; its advertised contract wins over the form's
     * snapshot so this component and its mount gate share one derivation. */
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
  ),
);
// Family-scoped label for the shared `strength` wire field (#1055).
const strength = computed(() => strengthSemantics(props.form.family));
/** The model's own image-attachment shape — the single policy every surface
 * renders (`@studio/lib/sourceMediaPlan`). `none` and `h3-references` render
 * nothing here. */
const plan = computed(() => sourceMediaPlan(caps.value));
const flux2Dev = computed(() => isFlux2DevModel(props.form.model));
/** Why the attached conditioning would be refused, in the server's own order. */
const conditioningError = computed(() => sourceConditioningValidationError(props.form));

const pickerOpen = ref(false);
const endPickerOpen = ref(false);
const maskOpen = ref(false);

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

// ── Qwen-edit Target + Reference strip ──────────────────────────────────────
// Ordered base64 attachments: index 0 is the primary edit Target, the rest
// are References (web Composer parity — the order ships as `edit_images`).

const editPickerOpen = ref(false);
const dragIndex = ref<number | null>(null);

function onEditPicked(picked: PickedImage[]) {
  if (picked.length === 0) return;
  const next = [...props.form.imageAttachments, ...picked.map((p) => p.base64)];
  props.form.imageAttachments = flux2Dev.value ? next.slice(0, 4) : next;
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
    if (b64) props.form.sourceFit = { mode: "lanczos-resize" };
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

// ── MiniMax H3 FL2VA boundaries ─────────────────────────────────────────────
// The same two wells, backed by the dedicated H3 authoring state and the
// shared studio setters so a file and a gallery pick produce identical facts.

const h3Authoring = computed(() => props.form.h3Authoring ?? emptyMinimaxH3AuthoringState());
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
  const raw = (e.target as HTMLSelectElement).value;
  if (raw === "crop-fill") {
    props.form.sourceFit = { mode: "crop-fill", alignX: "center", alignY: "center" };
    return;
  }
  if (raw === "lanczos-resize") {
    props.form.sourceFit = { mode: "lanczos-resize" };
    return;
  }
  if (raw === "upscale-then-fit") {
    props.form.sourceFit = {
      mode: "upscale-then-fit",
      upscalerModel: props.form.upscaleModel || models.upscalers[0]?.name || "",
      // Maskless families (video img2img) can't repaint pad bands — fill the
      // canvas instead of padding it.
      fit: caps.value.supportsMask
        ? { mode: "pad-repaint" }
        : { mode: "crop-fill", alignX: "center", alignY: "center" },
    };
    return;
  }
  props.form.sourceFit = { mode: raw === "pad-fit" ? "pad-fit" : "pad-repaint" };
}
</script>

<template>
  <!-- Ordered Qwen edit pictures or FLUX.2 reference images. -->
  <div v-if="plan.kind === 'attachments'">
    <div class="mb-2 flex items-center gap-2">
      <span class="edge-code">Pictures</span>
      <div class="border-edge h-px flex-1 border-t" />
    </div>

    <div class="flex gap-2 overflow-x-auto pb-1" data-test="attachment-strip">
      <div
        v-for="(image, index) in form.imageAttachments"
        :key="`${index}-${image.slice(0, 16)}`"
        class="relative w-20 shrink-0 overflow-hidden rounded-media border border-control-edge bg-bath"
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
          <div class="edge-code" :data-test="`attachment-role-${index}`">
            {{ flux2Dev ? `Reference ${index + 1}` : attachmentRoleLabel(index) }}
          </div>
          <div class="truncate text-caption text-ink" :data-test="`attachment-title-${index}`">
            {{ attachmentTitleLabel(index) }}
          </div>
        </div>
        <button
          v-if="index > 0"
          type="button"
          class="absolute top-1 left-1 h-5 w-5 rounded-control bg-bath/90 text-caption text-ink-2 hover:text-ink"
          :aria-label="`Move ${attachmentTitleLabel(index)} left`"
          :data-test="`move-attachment-up-${index}`"
          @click="moveAttachmentBy(index, -1)"
        >
          ‹
        </button>
        <button
          v-if="index < form.imageAttachments.length - 1"
          type="button"
          class="absolute top-1 left-7 h-5 w-5 rounded-control bg-bath/90 text-caption text-ink-2 hover:text-ink"
          :aria-label="`Move ${attachmentTitleLabel(index)} right`"
          :data-test="`move-attachment-down-${index}`"
          @click="moveAttachmentBy(index, 1)"
        >
          ›
        </button>
        <button
          type="button"
          class="border-edge absolute top-1 right-1 h-5 w-5 rounded-control border bg-bath text-ink-2 hover:text-stop"
          :aria-label="`Remove ${attachmentTitleLabel(index)}`"
          :data-test="`remove-attachment-${index}`"
          @click="removeAttachmentAt(index)"
        >
          ✕
        </button>
      </div>

      <button
        type="button"
        class="flex h-[4.75rem] w-20 shrink-0 cursor-pointer items-center justify-center rounded-media border border-dashed border-control-edge text-body-lg text-ink-3 transition-colors hover:border-safelight hover:text-safelight focus-visible:outline-2 focus-visible:outline-safelight"
        data-test="add-edit-image"
        aria-label="Add pictures"
        @click="editPickerOpen = true"
      >
        ＋
      </button>
    </div>
    <p class="mt-1 text-caption text-ink-3">
      {{
        flux2Dev
          ? "Up to four ordered references. Drag (or ‹ ›) to reorder."
          : "First picture is the edit Target; the rest are References. Drag (or ‹ ›) to reorder."
      }}
    </p>

    <ImagePickerModal
      :open="editPickerOpen"
      :multiple="true"
      title="Add pictures"
      @pick="onEditPicked"
      @close="editPickerOpen = false"
    />
  </div>

  <div v-else-if="plan.kind === 'single'" data-test="source-media-controls">
    <SourceMediaWells
      :plan="plan"
      :source="form.sourceImage ? { data: form.sourceImage, filename: form.sourceImageName } : null"
      :end-frame="
        form.endFrame ? { data: form.endFrame.base64, filename: form.endFrame.filename } : null
      "
      :error="conditioningError"
      @file="onWellFile"
      @gallery="onWellGallery"
      @clear="onWellClear"
    />

    <!-- Source fit (how a mismatched source maps onto the target canvas;
         applied client-side on submit — labels mirror the web SPA) -->
    <template v-if="form.sourceImage">
      <label class="mt-3 block text-caption text-ink-2" for="source-fit-policy">Source fit</label>
      <select
        id="source-fit-policy"
        :value="form.sourceFit?.mode ?? 'pad-repaint'"
        data-test="source-fit-policy"
        class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
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
        class="mt-1 text-caption text-ink-3"
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
    <template v-if="form.sourceImage && caps.supportsStrength">
      <label class="mt-3 flex items-center justify-between text-caption text-ink-2">
        {{ strength.label }}
        <span class="data-mono text-ink">{{ form.strength.toFixed(2) }}</span>
      </label>
      <input
        v-model.number="form.strength"
        type="range"
        min="0.05"
        max="1"
        step="0.05"
        class="mt-1 w-full accent-[var(--safelight)]"
        :aria-label="strength.label"
        :title="strength.hint"
      />
    </template>

    <!-- Mask well (inpaint families) -->
    <template v-if="caps.supportsMask && form.sourceImage">
      <div class="mt-3 flex items-center justify-between">
        <label class="text-caption text-ink-2">Mask</label>
        <button
          type="button"
          class="text-caption text-safelight underline-offset-2 hover:underline"
          data-test="source-edit-mask"
          @click="maskOpen = true"
        >
          Edit mask…
        </button>
      </div>
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
      <label class="mt-3 block text-caption text-ink-2">Control image</label>
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
        <label class="mt-3 block text-caption text-ink-2" for="controlnet-select">
          Control model
        </label>
        <select
          id="controlnet-select"
          data-test="controlnet-select"
          :value="controlSelectValue"
          class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
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
          class="mt-1 text-caption text-ink-3"
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
          class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink placeholder:text-ink-3"
        />
        <label class="mt-3 flex items-center justify-between text-caption text-ink-2">
          Control scale <span class="data-mono text-ink">{{ form.controlScale.toFixed(2) }}</span>
        </label>
        <input
          v-model.number="form.controlScale"
          type="range"
          min="0"
          max="2"
          step="0.05"
          class="mt-1 w-full accent-[var(--safelight)]"
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
      <label class="mt-3 block text-caption text-ink-2" for="source-fit-policy">Source fit</label>
      <select
        id="source-fit-policy"
        :value="coerceSourceFitForMaskless(form.sourceFit ?? { mode: 'crop-fill' }).mode"
        data-test="source-fit-policy"
        class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
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
