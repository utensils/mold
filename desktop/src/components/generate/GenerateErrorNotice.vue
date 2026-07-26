<script setup lang="ts">
import { ref, watch } from "vue";

const props = defineProps<{ message: string }>();
const copied = ref(false);
const copyFailed = ref(false);

watch(
  () => props.message,
  () => {
    copied.value = false;
    copyFailed.value = false;
  },
);

async function copyError() {
  try {
    await navigator.clipboard.writeText(props.message);
    copied.value = true;
    copyFailed.value = false;
  } catch {
    copied.value = false;
    copyFailed.value = true;
  }
}
</script>

<template>
  <div
    role="alert"
    class="border-stop/45 flex flex-wrap items-start gap-3 rounded-control border bg-stop/10 px-3 py-2.5 text-stop"
  >
    <p
      data-test="generate-error-message"
      data-selectable
      class="min-w-0 flex-1 text-body leading-relaxed"
    >
      {{ message }}
    </p>
    <button
      type="button"
      data-test="copy-generate-error"
      class="border-stop/40 flex h-8 w-8 shrink-0 items-center justify-center rounded-control border text-stop transition-colors hover:bg-stop/10 hover:text-ink active:translate-y-px"
      :aria-label="
        copied ? 'Error copied' : copyFailed ? 'Copy error failed' : 'Copy error message'
      "
      :title="copied ? 'Copied' : copyFailed ? 'Could not copy' : 'Copy error message'"
      @click="copyError"
    >
      <svg
        v-if="!copied"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.7"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <rect x="8" y="8" width="11" height="11" rx="2" />
        <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
      </svg>
      <svg
        v-else
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.9"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="m5 12 4 4L19 6" />
      </svg>
    </button>
    <div v-if="$slots.actions" class="flex w-full flex-wrap items-center gap-2">
      <slot name="actions" />
    </div>
  </div>
</template>
