<script setup lang="ts">
import { computed, ref } from "vue";
import IdentityPhotoWell from "@studio/components/IdentityPhotoWell.vue";
import type { GenerateForm } from "../lib/generateForm";
import { identityConditioningValidationError } from "../lib/generateValidation";
import { fileToBase64 } from "../lib/image";
import {
  ingestMobileIdentityPhoto,
  mobileIdentityBudgetBytes,
  mobileIdentityFileRefusal,
  mobileIdentityMimeType,
} from "./identity";

/**
 * iPhone's acquire-and-validate wrapper around the shared identity photo well.
 *
 * The shared component owns the wording, the accepted formats, and the inline
 * refusal slot; this owns what a picked photo means on a phone. Picking uses
 * the well's own file input, which is the native iOS photo/camera picker —
 * deliberately NOT the gallery sheet the source wells offer, because a gallery
 * print is a render, not a reference photograph.
 *
 * Every reason renders INLINE beside the control (iPhone invariant: persistent
 * inline banners, never toasts), and the whole block is mounted only for a
 * checkpoint that advertises identity support — a parked photo is retained in
 * form state and simply not rendered.
 */
const props = defineProps<{ form: GenerateForm }>();

/** A local read/format refusal, cleared by the next successful pick. */
const ingestError = ref<string | null>(null);

const mimeType = computed(() => mobileIdentityMimeType(props.form.identityImage?.filename));

const error = computed(() => ingestError.value ?? identityConditioningValidationError(props.form));

async function onFile(file: File): Promise<void> {
  const refused = mobileIdentityFileRefusal(file);
  if (refused) {
    ingestError.value = refused;
    return;
  }
  let base64: string;
  try {
    base64 = await fileToBase64(file);
  } catch {
    ingestError.value = "Couldn’t read that photo.";
    return;
  }
  const result = ingestMobileIdentityPhoto(
    { filename: file.name, base64 },
    mobileIdentityBudgetBytes(props.form),
  );
  if (!result.ok) {
    ingestError.value = result.error;
    return;
  }
  ingestError.value = null;
  props.form.identityImage = result.image;
}

function onClear(): void {
  ingestError.value = null;
  props.form.identityImage = null;
}
</script>

<template>
  <div class="mobile-identity-well" data-test="mobile-identity-well">
    <IdentityPhotoWell
      touch-friendly
      :image="form.identityImage?.base64 || null"
      :mime-type="mimeType"
      :filename="form.identityImage?.filename ?? null"
      :error="error"
      @file="onFile"
      @clear="onClear"
    />
  </div>
</template>

<style scoped>
.mobile-identity-well {
  display: grid;
  gap: 8px;
}
</style>
