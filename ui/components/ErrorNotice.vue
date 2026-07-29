<script setup lang="ts">
import { ref, watch } from "vue";
import Icon from "./Icon.vue";

const props = withDefaults(
  defineProps<{
    message: string;
    /** Optional diagnostic payload; visible copy remains human-readable. */
    copyMessage?: string | null;
  }>(),
  { copyMessage: null },
);

const copied = ref(false);
const copyFailed = ref(false);

watch(
  () => [props.message, props.copyMessage],
  () => {
    copied.value = false;
    copyFailed.value = false;
  },
);

async function copyError() {
  try {
    await navigator.clipboard.writeText(props.copyMessage || props.message);
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
    data-test="error-notice"
    class="flex flex-wrap items-start gap-3 rounded-control border border-stop/45 bg-stop/10 px-3 py-2.5 text-stop"
  >
    <p
      data-test="error-notice-message"
      data-selectable
      class="min-w-0 flex-1 text-body leading-relaxed"
    >
      {{ message }}
    </p>
    <button
      type="button"
      data-test="copy-error-notice"
      class="flex h-9 w-9 shrink-0 items-center justify-center rounded-control border border-stop/40 text-stop transition-colors hover:bg-stop/10 hover:text-ink active:translate-y-px"
      :aria-label="
        copied
          ? 'Error copied'
          : copyFailed
            ? 'Copy error failed'
            : 'Copy error message'
      "
      :title="
        copied ? 'Copied' : copyFailed ? 'Could not copy' : 'Copy error message'
      "
      @click="copyError"
    >
      <Icon v-if="!copied" name="copy" :size="16" />
      <Icon v-else name="check" :size="16" />
    </button>
    <div v-if="$slots.actions" class="flex w-full flex-wrap items-center gap-2">
      <slot name="actions" />
    </div>
  </div>
</template>
