<script setup lang="ts">
import { computed, ref } from "vue";
import IdentityPhotoWell from "@studio/components/IdentityPhotoWell.vue";
import {
  identityImageError,
  identityValidationError,
  supportsIdentity,
} from "@studio/lib/identityConditioning";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { blobToBase64 } from "../../lib/base64";
import type { GenerateFormState, ModelInfoExtended } from "../../types";

/**
 * Face-identity (PuLID) conditioning in the primary form (#1224) — the photo
 * well sits directly below the source-media card because it is media the user
 * attaches, not a setting: only the two knobs live in Advanced.
 *
 * Every rule comes from `@studio/lib/identityConditioning`; this component
 * owns only what a picked File means on this surface. Upload and drop only in
 * this pass — no gallery picker, so the shared well's gallery action is off.
 */
const props = withDefaults(
  defineProps<{
    modelValue: GenerateFormState;
    /** Installed models on the selected generation route. */
    models?: ModelInfoExtended[];
    /** One-line page-owned disclosure, e.g. a reuse whose photo is gone. */
    notice?: string | null;
  }>(),
  { models: () => [], notice: null },
);

const emit = defineEmits<{ "update:modelValue": [value: GenerateFormState] }>();

function patch(next: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...next });
}

const selectedModel = computed(
  () =>
    props.models.find((model) => model.name === props.modelValue.model) ?? null,
);
/**
 * The server-authored recipe is the authority; the additive `/api/models`
 * flag answers for a host that sends no profile. With no resolved row at all
 * the snapshot taken on model change stands in, exactly as it does in
 * `toRequest` — an inventory blip must not blank a well the user is using.
 */
const supported = computed(() =>
  selectedModel.value
    ? supportsIdentity(
        effectiveGenerationRecipe(
          selectedModel.value,
          props.modelValue.pipeline,
        ),
        selectedModel.value,
      )
    : (props.modelValue.identitySupported ?? false),
);

/** Only real bytes are an attachment; a reattach descriptor carries none. */
const stagedPhoto = computed(() =>
  props.modelValue.identityImage?.base64
    ? {
        base64: props.modelValue.identityImage.base64,
        filename: props.modelValue.identityImage.filename,
      }
    : null,
);

/**
 * A staged photo keeps the card on screen even where the checkpoint cannot
 * use it. `toRequest` already keeps it off the wire, but hiding the well
 * outright would point the inline refusal at a control the user cannot see —
 * and leave no way to remove the photo that is blocking Generate.
 */
const visible = computed(
  () => supported.value || props.modelValue.identityImage != null,
);

const uploadError = ref<string | null>(null);

/** Why this identity partition would be refused, in the server's own order. */
const conditioningError = computed(() =>
  identityValidationError({
    supported: supported.value,
    image: stagedPhoto.value,
    weight: props.modelValue.identityWeight ?? null,
    startStep: props.modelValue.identityStartStep ?? null,
    steps: props.modelValue.steps,
    hasLora: props.modelValue.loras.length > 0,
    hasSourceImage: props.modelValue.imageAttachments.length > 0,
    model: props.modelValue.model,
  }),
);

async function onFile(file: File) {
  const base64 = await blobToBase64(file);
  // The same header-only pre-checks admission runs — format, encoded size,
  // declared dimensions — so a refusal is immediate instead of a failed job.
  // Drag-and-drop bypasses the file input's accept filter, so this is the
  // only gate on that path.
  const rejection = identityImageError(base64);
  if (rejection) {
    uploadError.value = rejection;
    return;
  }
  uploadError.value = null;
  const dimensions = imageDimensionsFromBase64(base64);
  // No fit policy and no resize: an identity photo is a face reference, not
  // a composition input, and travels to the server exactly as picked.
  patch({
    identityImage: {
      kind: "upload",
      filename: file.name,
      base64,
      width: dimensions?.width ?? null,
      height: dimensions?.height ?? null,
      mime: file.type || null,
    },
  });
}

function onClear() {
  uploadError.value = null;
  patch({ identityImage: null });
}
</script>

<template>
  <section v-if="visible" class="idp" data-test="identity-panel">
    <IdentityPhotoWell
      :image="modelValue.identityImage?.base64 || null"
      :mime-type="modelValue.identityImage?.mime ?? null"
      :filename="modelValue.identityImage?.filename ?? null"
      :error="uploadError ?? conditioningError"
      :gallery="false"
      @file="onFile"
      @clear="onClear"
    />
    <p v-if="notice" class="idp__notice" data-test="identity-notice">
      {{ notice }}
    </p>
  </section>
</template>

<style scoped>
.idp {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 18px;
}
.idp__notice {
  margin: 8px 0 0;
  font-size: 10.5px;
  line-height: 1.45;
  color: var(--ink-3);
}
</style>
