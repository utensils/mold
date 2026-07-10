<script setup lang="ts">
import { computed, ref } from "vue";
import type { GenerateForm, PickedImage } from "../../lib/generateForm";
import { generationCapabilitiesForFamily } from "../../lib/capabilities";
import { base64ToDataUrl, fileToBase64 } from "../../lib/image";
import { useToastStore } from "../../stores/toasts";
import ImagePickerModal from "./ImagePickerModal.vue";
import MaskEditorModal from "./MaskEditorModal.vue";

const props = defineProps<{ form: GenerateForm }>();
const toasts = useToastStore();

const caps = computed(() => generationCapabilitiesForFamily(props.form.family));

const pickerOpen = ref(false);
const maskOpen = ref(false);

function onSourcePicked(picked: PickedImage[]) {
  const first = picked[0];
  if (first) props.form.sourceImage = first.base64;
}
function onMaskApplied(mask: string) {
  props.form.maskImage = mask;
}

type Slot = "source" | "mask" | "control";
const dragOver = ref<Slot | null>(null);
const inputEls = ref<Record<Slot, HTMLInputElement | null>>({
  source: null,
  mask: null,
  control: null,
});

function setSlot(slot: Slot, b64: string | null) {
  if (slot === "source") props.form.sourceImage = b64;
  else if (slot === "mask") props.form.maskImage = b64;
  else props.form.controlImage = b64;
}
async function ingest(slot: Slot, file: File | undefined | null) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    toasts.push("That file isn't an image.", "error");
    return;
  }
  try {
    setSlot(slot, await fileToBase64(file));
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
</script>

<template>
  <div v-if="caps.supportsImg2img && caps.sourceImageMode === 'single'">
    <div class="mt-5 mb-2 flex items-center gap-2">
      <span class="edge-code">Source</span>
      <div class="border-edge h-px flex-1 border-t" />
    </div>

    <!-- Source well -->
    <div>
      <input
        :ref="(el) => (inputEls.source = el as HTMLInputElement | null)"
        type="file"
        accept="image/*"
        class="hidden"
        @change="onPick('source', $event)"
      />
      <div
        v-if="!form.sourceImage"
        class="flex h-24 cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors"
        :class="
          dragOver === 'source'
            ? 'border-safelight text-safelight'
            : 'border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] text-ink-3'
        "
        @click="pick('source')"
        @dragover.prevent="dragOver = 'source'"
        @dragleave="dragOver = null"
        @drop.prevent="onDrop('source', $event)"
      >
        Drop an image or click to pick
      </div>
      <div v-else class="relative inline-block">
        <img
          :src="base64ToDataUrl(form.sourceImage)"
          alt="source"
          class="max-h-40 rounded-media border border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] p-px"
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

    <!-- Strength -->
    <template v-if="form.sourceImage">
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
          class="flex h-16 cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors"
          :class="
            dragOver === 'mask'
              ? 'border-safelight text-safelight'
              : 'border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] text-ink-3'
          "
          @click="pick('mask')"
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
          class="flex h-16 cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors"
          :class="
            dragOver === 'control'
              ? 'border-safelight text-safelight'
              : 'border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] text-ink-3'
          "
          @click="pick('control')"
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
        <label class="mt-3 text-caption text-ink-2">Control model</label>
        <input
          v-model="form.controlModel"
          data-selectable
          type="text"
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
    <MaskEditorModal
      :open="maskOpen"
      :source="form.sourceImage"
      :initial-mask="form.maskImage"
      @apply="onMaskApplied"
      @close="maskOpen = false"
    />
  </div>
</template>
