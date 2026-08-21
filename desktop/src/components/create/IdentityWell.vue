<script setup lang="ts">
import { computed, ref } from "vue";
import IdentityPhotoWell from "@studio/components/IdentityPhotoWell.vue";
import {
  IDENTITY_PHOTO_LABEL,
  IDENTITY_PHOTO_UNAVAILABLE,
  identityImageError,
} from "@studio/lib/identityConditioning";
import type { GenerateForm } from "../../lib/generateForm";
import { fileToBase64, isStillImageFile } from "../../lib/image";
import { identityConditioningValidationError } from "../../lib/generateValidation";

/**
 * Desktop's acquire-and-validate wrapper around the shared identity photo
 * well. The shared component owns the wording, the accepted formats, and the
 * inline refusal slot; this owns what a dropped or picked `File` means here.
 *
 * Every reason is rendered INLINE beside the control — never a toast. A
 * refused face photo is a property of the composed print (the submit gate
 * reports the same message), not an event that scrolls away, and the whole
 * block is mounted only for a checkpoint that advertises identity support.
 *
 * Upload and drop only: there is deliberately no gallery picker yet, because a
 * gallery print is a render, not a reference photograph.
 */
const props = defineProps<{ form: GenerateForm }>();

/** A local read/format refusal, cleared by the next successful pick. */
const ingestError = ref<string | null>(null);

/**
 * Preview type from the provenance label, exactly as the shared H3 boundary
 * setter derives it — a JPEG previewed as `data:image/png` relies on browser
 * sniffing that is not guaranteed.
 */
const mimeType = computed(() => {
  const name = props.form.identityImage?.filename?.trim().toLowerCase() ?? "";
  return name.endsWith(".jpg") || name.endsWith(".jpeg") ? "image/jpeg" : "image/png";
});

/** Provenance with no bytes: Reuse settings restored a print whose photo is
 * not on this device. Saying so beats rendering someone else's face. */
const needsReattach = computed(
  () => Boolean(props.form.identityImage) && !props.form.identityImage?.base64,
);

const error = computed(() => {
  if (ingestError.value) return ingestError.value;
  if (needsReattach.value) return IDENTITY_PHOTO_UNAVAILABLE;
  return identityConditioningValidationError(props.form);
});

async function onFile(file: File) {
  // Dropped files bypass the input's accept filter, so gate by MIME with a
  // filename fallback — the engine takes PNG and JPEG only.
  if (
    file.type !== "image/png" &&
    file.type !== "image/jpeg" &&
    !(!file.type && isStillImageFile(file.name))
  ) {
    ingestError.value = `${IDENTITY_PHOTO_LABEL} must be a PNG or JPEG image.`;
    return;
  }
  let base64: string;
  try {
    base64 = await fileToBase64(file);
  } catch {
    ingestError.value = "Couldn't read the image.";
    return;
  }
  // The server's own header-only pre-checks, run before anything is staged so
  // a photo that cannot be admitted never becomes part of the draft.
  const refused = identityImageError(base64);
  if (refused) {
    ingestError.value = refused;
    return;
  }
  ingestError.value = null;
  props.form.identityImage = { filename: file.name || "identity photo", base64 };
}

function onClear() {
  ingestError.value = null;
  props.form.identityImage = null;
}
</script>

<template>
  <IdentityPhotoWell
    :image="form.identityImage?.base64 || null"
    :mime-type="mimeType"
    :filename="form.identityImage?.filename ?? null"
    :error="error"
    @file="onFile"
    @clear="onClear"
  />
</template>
