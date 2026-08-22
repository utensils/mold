<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import IdentityPhotoWell from "@studio/components/IdentityPhotoWell.vue";
import type { GenerateForm } from "../lib/generateForm";
import { identityConditioningValidationError } from "../lib/generateValidation";
import { fileToBase64 } from "../lib/image";
import {
  ingestMobileIdentityPhoto,
  mobileIdentityBudgetBytes,
  mobileIdentityFileRefusal,
  mobileIdentityFileSizeRefusal,
  mobileIdentityMimeType,
} from "./identity";
import MobileLibrarySheet from "./MobileLibrarySheet.vue";
import { isNativeAndroidRuntime } from "./platform";

/**
 * Phone acquire-and-validate wrapper around the shared identity photo well.
 *
 * The shared component owns the wording, the accepted formats, and the inline
 * refusal slot; this owns what a picked photo means on a phone. Picking uses
 * iOS uses the well's file input. Android delegates to the platform bridge's
 * Photo Picker or camera so no broad storage permission is needed. Neither
 * path offers the Mold gallery: a render is not a reference photograph.
 *
 * Every reason renders INLINE beside the control (mobile invariant: persistent
 * inline banners, never toasts), and the whole block is mounted only for a
 * checkpoint that advertises identity support — a parked photo is retained in
 * form state and simply not rendered.
 */
const props = defineProps<{ form: GenerateForm }>();
const androidNativeRuntime = isNativeAndroidRuntime();
const pickerOpen = ref(false);
const PICKER_HISTORY_KEY = "moldIdentityPicker";

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
  // Size is judged from the File itself, BEFORE it is read: a phone has no
  // room to spare, and reading a 40 MP photo into a base64 string only to
  // refuse it is exactly the allocation this check exists to avoid.
  const budget = mobileIdentityBudgetBytes(props.form);
  const tooLarge = mobileIdentityFileSizeRefusal(file.size, budget);
  if (tooLarge) {
    ingestError.value = tooLarge;
    return;
  }
  let base64: string;
  try {
    base64 = await fileToBase64(file);
  } catch {
    ingestError.value = "Couldn’t read that photo.";
    return;
  }
  const result = ingestMobileIdentityPhoto({ filename: file.name, base64 }, budget);
  if (!result.ok) {
    ingestError.value = result.error;
    return;
  }
  ingestError.value = null;
  props.form.identityImage = result.image;
}

interface NativeIdentityPhoto {
  cancelled: boolean;
  filename?: string | null;
  mimeType?: string | null;
  sizeBytes?: number | null;
  dataB64?: string | null;
}

function openNativePicker(): void {
  pickerOpen.value = true;
  window.history.pushState({ ...(window.history.state ?? {}), [PICKER_HISTORY_KEY]: true }, "");
}

function dismissNativePicker(): void {
  pickerOpen.value = false;
  if (window.history.state?.[PICKER_HISTORY_KEY]) window.history.back();
}

function onHistoryPop(): void {
  pickerOpen.value = false;
}

async function pickNativeIdentity(source: "library" | "camera"): Promise<void> {
  dismissNativePicker();
  try {
    const picked = await invoke<NativeIdentityPhoto>("pick_identity_photo", { source });
    if (picked.cancelled) return;
    if (!picked.filename || !picked.dataB64 || picked.sizeBytes == null) {
      ingestError.value = "Android returned an incomplete identity photo.";
      return;
    }
    const budget = mobileIdentityBudgetBytes(props.form);
    const tooLarge = mobileIdentityFileSizeRefusal(picked.sizeBytes, budget);
    if (tooLarge) {
      ingestError.value = tooLarge;
      return;
    }
    const refused = mobileIdentityFileRefusal({
      name: picked.filename,
      type: picked.mimeType ?? "",
    });
    if (refused) {
      ingestError.value = refused;
      return;
    }
    const result = ingestMobileIdentityPhoto(
      { filename: picked.filename, base64: picked.dataB64 },
      budget,
    );
    if (!result.ok) {
      ingestError.value = result.error;
      return;
    }
    ingestError.value = null;
    props.form.identityImage = result.image;
  } catch (error) {
    ingestError.value = error instanceof Error ? error.message : String(error);
  }
}

function onClear(): void {
  ingestError.value = null;
  props.form.identityImage = null;
}

onMounted(() => window.addEventListener("popstate", onHistoryPop));
onBeforeUnmount(() => window.removeEventListener("popstate", onHistoryPop));
</script>

<template>
  <div class="mobile-identity-well" data-test="mobile-identity-well">
    <IdentityPhotoWell
      touch-friendly
      :touch-target-size="androidNativeRuntime ? 48 : 44"
      :image="form.identityImage?.base64 || null"
      :mime-type="mimeType"
      :filename="form.identityImage?.filename ?? null"
      :error="error"
      :native-picker="androidNativeRuntime"
      @file="onFile"
      @clear="onClear"
      @pick="openNativePicker"
    />
    <MobileLibrarySheet
      :open="pickerOpen"
      title="Identity photo"
      test-id="mobile-identity-picker"
      :touch-target-size="androidNativeRuntime ? 48 : 46"
      @close="dismissNativePicker"
    >
      <div class="mobile-identity-picker-actions">
        <button
          type="button"
          data-test="mobile-identity-pick-library"
          @click="pickNativeIdentity('library')"
        >
          Choose photo
        </button>
        <button
          type="button"
          data-test="mobile-identity-pick-camera"
          @click="pickNativeIdentity('camera')"
        >
          Take photo
        </button>
      </div>
    </MobileLibrarySheet>
  </div>
</template>

<style scoped>
.mobile-identity-well {
  display: grid;
  gap: 8px;
}
.mobile-identity-picker-actions {
  display: grid;
  gap: 10px;
}
.mobile-identity-picker-actions button {
  min-height: 48px;
  border: 1px solid var(--edge);
  border-radius: 10px;
  background: var(--print-surface);
  color: var(--ink);
  font: inherit;
}
</style>
