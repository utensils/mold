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
  MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
  type InlineGenerationMediaField,
} from "../lib/generateValidation";
import { base64ToDataUrl, fileToBase64, isStillImageFile } from "../lib/image";
import { coerceSourceFitForMaskless } from "@studio/lib/sourceFit";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    target?: ApiTarget | null;
    controlModels?: ModelEntry[];
    upscalers?: ModelEntry[];
  }>(),
  {
    target: null,
    controlModels: () => [],
    upscalers: () => [],
  },
);

const emit = defineEmits<{
  "validity-change": [valid: boolean];
}>();

const caps = computed(() => generationCapabilitiesForFamily(props.form.family));
const isQwenEdit = computed(() => caps.value.sourceImageMode === "qwen-edit");
const error = ref("");
const maskOpen = ref(false);

const sourceInput = ref<HTMLInputElement | null>(null);
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
  if (isQwenEdit.value && props.form.imageAttachments.length === 0) {
    return "Add a Target photo before generating an edit.";
  }
  if (caps.value.supportsControlNet && props.form.controlImage && !props.form.controlModel.trim()) {
    return "Choose an installed ControlNet model for the control photo.";
  }
  return null;
});

watch(validationError, (next) => emit("validity-change", !next), { immediate: true });

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
    error.value = "Combined generation media must be 45 MiB or smaller on iPhone.";
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

async function pickSource(event: Event): Promise<void> {
  const picked = (await readImages(event, false, "sourceImage"))[0];
  if (!picked) return;
  props.form.sourceImage = picked.b64;
  props.form.sourceImageName = picked.file.name || null;
}

async function pickEditImages(event: Event): Promise<void> {
  const picked = await readImages(event, true);
  if (picked.length === 0) return;
  props.form.imageAttachments = [
    ...props.form.imageAttachments,
    ...picked.map((image) => image.b64),
  ];
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
  const mode = (event.target as HTMLSelectElement).value;
  if (mode === "crop-fill") {
    props.form.sourceFit = { mode: "crop-fill", alignX: "center", alignY: "center" };
  } else if (mode === "pad-fit") {
    props.form.sourceFit = { mode: "pad-fit" };
  } else if (mode === "lanczos-resize") {
    props.form.sourceFit = { mode: "lanczos-resize" };
  } else if (mode === "upscale-then-fit") {
    props.form.sourceFit = {
      mode: "upscale-then-fit",
      upscalerModel: props.form.upscaleModel || props.upscalers[0]?.name || "",
      fit: caps.value.supportsMask
        ? { mode: "pad-repaint" }
        : { mode: "crop-fill", alignX: "center", alignY: "center" },
    };
  } else {
    props.form.sourceFit = { mode: "pad-repaint" };
  }
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
  <template v-if="caps.supportsImg2img">
    <fieldset v-if="isQwenEdit" class="mobile-source-controls" data-test="mobile-source-controls">
      <legend class="mobile-source-legend">Pictures</legend>
      <p class="mobile-source-note">
        The first picture is the edit Target. Additional pictures are References.
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
              attachmentRoleLabel(index)
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

    <fieldset v-else class="mobile-source-controls" data-test="mobile-source-controls">
      <legend class="mobile-source-legend">Source</legend>
      <input
        ref="sourceInput"
        hidden
        type="file"
        accept="image/png,image/jpeg"
        data-test="mobile-source-input"
        tabindex="-1"
        @change="pickSource"
      />

      <button
        v-if="!form.sourceImage"
        type="button"
        class="secondary-button mobile-source-pick"
        data-test="mobile-source-add"
        @click="sourceInput?.click()"
      >
        Choose photo
      </button>
      <template v-else>
        <figure class="mobile-source-preview-wrap">
          <img
            class="mobile-source-preview"
            data-test="mobile-source-preview"
            :src="base64ToDataUrl(form.sourceImage)"
            :alt="form.sourceImageName || 'Source photo'"
          />
          <figcaption v-if="form.sourceImageName">{{ form.sourceImageName }}</figcaption>
        </figure>
        <div class="mobile-media-actions">
          <button
            type="button"
            class="secondary-button"
            data-test="mobile-source-replace"
            @click="sourceInput?.click()"
          >
            Replace photo
          </button>
          <button
            type="button"
            class="secondary-button"
            data-test="mobile-source-remove"
            @click="removeSource"
          >
            Remove
          </button>
        </div>

        <label class="mobile-range-field">
          <span
            >Strength <output>{{ form.strength.toFixed(2) }}</output></span
          >
          <input
            v-model.number="form.strength"
            type="range"
            min="0.05"
            max="1"
            step="0.05"
            aria-label="Source strength"
            data-test="mobile-source-strength"
          />
        </label>

        <label class="field">
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
        <p v-if="form.sourceFit.mode === 'upscale-then-fit'" class="mobile-source-note">
          {{
            form.sourceFit.upscalerModel
              ? `Preprocesses with ${form.sourceFit.upscalerModel}.`
              : "No upscaler is available; the source will be fit without upscaling."
          }}
        </p>
      </template>

      <fieldset v-if="caps.supportsMask && form.sourceImage" class="mobile-source-subsection">
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
  </template>
</template>
