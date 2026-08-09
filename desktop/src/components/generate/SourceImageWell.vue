<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { GenerateForm, PickedImage } from "../../lib/generateForm";
import { generationCapabilitiesForFamily, isFlux2DevModel } from "../../lib/capabilities";
import { base64ToDataUrl, fileToBase64 } from "../../lib/image";
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
import ImagePickerModal from "./ImagePickerModal.vue";
import MaskEditorModal from "./MaskEditorModal.vue";

const props = defineProps<{ form: GenerateForm }>();
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
    props.form.guidanceCapabilities,
    props.form.sourceImageCapability,
  ),
);
const flux2Dev = computed(() => isFlux2DevModel(props.form.model));
/** Why the attached conditioning would be refused, in the server's own order. */
const conditioningError = computed(() => sourceConditioningValidationError(props.form));

const pickerOpen = ref(false);
const endPickerOpen = ref(false);
const maskOpen = ref(false);

function onSourcePicked(picked: PickedImage[]) {
  const first = picked[0];
  if (first) {
    props.form.sourceImage = first.base64;
    // Provenance label for Reuse-settings source restore.
    props.form.sourceImageName = first.filename || null;
    props.form.sourceFit = { mode: "lanczos-resize" };
  }
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
const dragOver = ref<Slot | null>(null);
const inputEls = ref<Record<Slot, HTMLInputElement | null>>({
  source: null,
  end: null,
  mask: null,
  control: null,
});

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
  if (!file.type.startsWith("image/")) {
    toasts.push("That file isn't an image.", "error");
    return;
  }
  try {
    setSlot(slot, await fileToBase64(file), file.name || null);
  } catch {
    toasts.push("Couldn't read the image.", "error");
  }
}

function onDrop(slot: Slot, e: DragEvent) {
  dragOver.value = null;
  void ingest(slot, e.dataTransfer?.files?.[0]);
}
function onPick(slot: Slot, e: Event) {
  void ingest(slot, (e.target as HTMLInputElement).files?.[0]);
}
function pick(slot: Slot) {
  inputEls.value[slot]?.click();
}
function clearSlot(slot: Slot) {
  setSlot(slot, null);
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
  <div v-if="caps.supportsImg2img && caps.sourceImageMode !== 'single'">
    <div class="mt-5 mb-2 flex items-center gap-2">
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

  <div v-else-if="caps.supportsImg2img && caps.sourceImageMode === 'single'">
    <div class="mt-5 mb-2 flex items-center gap-2">
      <span class="edge-code">Source</span>
      <span
        v-if="caps.requiresSourceImage"
        class="edge-code text-safelight"
        data-test="source-required-badge"
      >
        Required
      </span>
      <div class="border-edge h-px flex-1 border-t" />
    </div>

    <!-- Source well -->
    <div data-test="source-media-controls" class="flex flex-col items-start">
      <input
        :ref="(el) => (inputEls.source = el as HTMLInputElement | null)"
        type="file"
        accept="image/*"
        class="hidden"
        @change="onPick('source', $event)"
      />
      <div
        v-if="!form.sourceImage"
        role="button"
        tabindex="0"
        :aria-required="caps.requiresSourceImage || undefined"
        class="flex h-24 cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors focus-visible:outline-2 focus-visible:outline-safelight"
        :class="
          dragOver === 'source'
            ? 'border-safelight text-safelight'
            : 'border-control-edge text-ink-3'
        "
        @click="pick('source')"
        @keydown.enter.prevent="pick('source')"
        @keydown.space.prevent="pick('source')"
        @dragover.prevent="dragOver = 'source'"
        @dragleave="dragOver = null"
        @drop.prevent="onDrop('source', $event)"
      >
        Drop an image or click to pick
      </div>
      <div v-else class="relative inline-block max-w-full">
        <img
          :src="base64ToDataUrl(form.sourceImage)"
          alt="source"
          class="max-h-40 max-w-full rounded-media border border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] p-px"
        />
        <button
          type="button"
          class="border-edge absolute top-1 right-1 h-5 w-5 rounded-control border bg-bath text-ink-2 hover:text-stop"
          title="Clear source"
          aria-label="Clear source image"
          @click="clearSlot('source')"
        >
          ✕
        </button>
      </div>
      <button
        type="button"
        class="mt-2 text-caption text-ink-3 underline-offset-2 hover:text-ink hover:underline"
        data-test="source-choose-gallery"
        @click="pickerOpen = true"
      >
        Choose from gallery…
      </button>
    </div>

    <!-- Why the attached conditioning would be refused. One message at a time,
         in the same order admission checks them. -->
    <p
      v-if="conditioningError"
      class="mt-2 text-caption text-stop"
      role="alert"
      data-test="source-conditioning-error"
    >
      {{ conditioningError }}
    </p>

    <!-- End frame well (wan first/last-frame conditioning). Offered only when
         the server advertised a source-image contract for this checkpoint —
         an older server rejects wan keyframes outright. -->
    <template v-if="caps.supportsEndFrame">
      <div class="mt-4 mb-2 flex items-center gap-2">
        <span class="edge-code">End frame</span>
        <span class="edge-code text-ink-3">Optional</span>
        <div class="border-edge h-px flex-1 border-t" />
      </div>
      <div data-test="end-frame-media-controls" class="flex flex-col items-start">
        <input
          :ref="(el) => (inputEls.end = el as HTMLInputElement | null)"
          type="file"
          accept="image/*"
          class="hidden"
          @change="onPick('end', $event)"
        />
        <div
          v-if="!form.endFrame"
          role="button"
          tabindex="0"
          class="flex h-24 cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors focus-visible:outline-2 focus-visible:outline-safelight"
          :class="
            dragOver === 'end'
              ? 'border-safelight text-safelight'
              : 'border-control-edge text-ink-3'
          "
          @click="pick('end')"
          @keydown.enter.prevent="pick('end')"
          @keydown.space.prevent="pick('end')"
          @dragover.prevent="dragOver = 'end'"
          @dragleave="dragOver = null"
          @drop.prevent="onDrop('end', $event)"
        >
          Drop the closing image or click to pick
        </div>
        <div v-else class="relative inline-block max-w-full">
          <img
            :src="base64ToDataUrl(form.endFrame.base64)"
            alt="end frame"
            class="max-h-40 max-w-full rounded-media border border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] p-px"
          />
          <button
            type="button"
            class="border-edge absolute top-1 right-1 h-5 w-5 rounded-control border bg-bath text-ink-2 hover:text-stop"
            title="Clear end frame"
            aria-label="Clear end frame"
            @click="clearSlot('end')"
          >
            ✕
          </button>
        </div>
        <button
          type="button"
          class="mt-2 text-caption text-ink-3 underline-offset-2 hover:text-ink hover:underline"
          data-test="end-frame-choose-gallery"
          @click="endPickerOpen = true"
        >
          Choose from gallery…
        </button>
      </div>
      <p class="mt-1 text-caption text-ink-3">
        Renders a first/last-frame clip: the source opens it, this closes it.
      </p>
    </template>

    <!-- Strength (wan pins the first frame exactly and never reads it) -->
    <template v-if="form.sourceImage && caps.supportsStrength">
      <label class="mt-3 flex items-center justify-between text-caption text-ink-2">
        Strength <span class="data-mono text-ink">{{ form.strength.toFixed(2) }}</span>
      </label>
      <input
        v-model.number="form.strength"
        type="range"
        min="0.05"
        max="1"
        step="0.05"
        class="mt-1 w-full accent-[var(--safelight)]"
      />
    </template>

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
      <input
        :ref="(el) => (inputEls.mask = el as HTMLInputElement | null)"
        type="file"
        accept="image/*"
        class="hidden"
        @change="onPick('mask', $event)"
      />
      <div class="mt-1">
        <div
          v-if="!form.maskImage"
          role="button"
          tabindex="0"
          class="flex h-16 cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors focus-visible:outline-2 focus-visible:outline-safelight"
          :class="
            dragOver === 'mask'
              ? 'border-safelight text-safelight'
              : 'border-control-edge text-ink-3'
          "
          @click="pick('mask')"
          @keydown.enter.prevent="pick('mask')"
          @keydown.space.prevent="pick('mask')"
          @dragover.prevent="dragOver = 'mask'"
          @dragleave="dragOver = null"
          @drop.prevent="onDrop('mask', $event)"
        >
          White repaints, black preserves
        </div>
        <div v-else class="relative inline-block">
          <img
            :src="base64ToDataUrl(form.maskImage)"
            alt="mask"
            class="max-h-28 rounded-media border border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] p-px"
          />
          <button
            type="button"
            class="border-edge absolute top-1 right-1 h-5 w-5 rounded-control border bg-bath text-ink-2 hover:text-stop"
            title="Clear mask"
            aria-label="Clear mask image"
            @click="clearSlot('mask')"
          >
            ✕
          </button>
        </div>
      </div>
    </template>

    <!-- Control well + model + scale (sd15 only) -->
    <template v-if="caps.supportsControlNet">
      <label class="mt-3 text-caption text-ink-2">Control image</label>
      <input
        :ref="(el) => (inputEls.control = el as HTMLInputElement | null)"
        type="file"
        accept="image/*"
        class="hidden"
        @change="onPick('control', $event)"
      />
      <div class="mt-1">
        <div
          v-if="!form.controlImage"
          role="button"
          tabindex="0"
          class="flex h-16 cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors focus-visible:outline-2 focus-visible:outline-safelight"
          :class="
            dragOver === 'control'
              ? 'border-safelight text-safelight'
              : 'border-control-edge text-ink-3'
          "
          @click="pick('control')"
          @keydown.enter.prevent="pick('control')"
          @keydown.space.prevent="pick('control')"
          @dragover.prevent="dragOver = 'control'"
          @dragleave="dragOver = null"
          @drop.prevent="onDrop('control', $event)"
        >
          Drop a control image
        </div>
        <div v-else class="relative inline-block">
          <img
            :src="base64ToDataUrl(form.controlImage)"
            alt="control"
            class="max-h-28 rounded-media border border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] p-px"
          />
          <button
            type="button"
            class="border-edge absolute top-1 right-1 h-5 w-5 rounded-control border bg-bath text-ink-2 hover:text-stop"
            title="Clear control image"
            aria-label="Clear control image"
            @click="clearSlot('control')"
          >
            ✕
          </button>
        </div>
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
      @pick="onSourcePicked"
      @close="pickerOpen = false"
    />
    <ImagePickerModal
      v-if="caps.supportsEndFrame"
      :open="endPickerOpen"
      :multiple="false"
      title="Pick an end frame"
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
</template>
