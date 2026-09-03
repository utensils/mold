<script setup lang="ts">
import { ref, watch } from "vue";
import Icon from "./Icon.vue";

const props = withDefaults(
  defineProps<{
    message: string;
    /** Optional diagnostic payload; visible copy remains human-readable. */
    copyMessage?: string | null;
    compact?: boolean;
    dismissible?: boolean;
  }>(),
  { copyMessage: null, compact: false, dismissible: false },
);

const emit = defineEmits<{ dismiss: [] }>();

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
    class="ms-error"
    :class="{ 'ms-error--compact': compact }"
  >
    <p
      data-test="error-notice-message"
      data-selectable
      class="ms-error__message"
    >
      {{ message }}
    </p>
    <button
      type="button"
      data-test="copy-error-notice"
      class="ms-error__button ms-error__button--bordered"
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
    <button
      v-if="dismissible"
      type="button"
      data-test="dismiss-error-notice"
      class="ms-error__button"
      aria-label="Dismiss error message"
      title="Dismiss"
      @click="emit('dismiss')"
    >
      <Icon name="close" :size="16" />
    </button>
    <div v-if="$slots.actions" class="ms-error__actions">
      <slot name="actions" />
    </div>
  </div>
</template>

<style scoped>
/* A notice is bordered in its state colour, like a toast (README §06). */
.ms-error {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 12px;
  padding: 10px 12px;
  border: var(--mold-bw) solid var(--mold-error);
  border-radius: var(--mold-radius-2);
  background: var(--mold-panel-raised, var(--mold-surface));
  color: var(--mold-error);
}

.ms-error--compact {
  align-items: center;
  gap: 8px;
  padding-top: 6px;
  padding-bottom: 6px;
}

.ms-error__message {
  min-width: 0;
  flex: 1;
  margin: 0;
  font-size: var(--mold-fs-sm);
  line-height: var(--mold-lh-body);
  overflow-wrap: anywhere;
}

.ms-error--compact .ms-error__message {
  line-height: var(--mold-lh-snug);
}

.ms-error__button {
  display: flex;
  width: 32px;
  height: 32px;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  border: var(--mold-bw) solid transparent;
  border-radius: var(--mold-radius-2);
  color: var(--mold-error);
  background: none;
  transition: color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-error__button--bordered {
  border-color: var(--mold-error);
}

.ms-error__button:hover {
  color: var(--mold-text);
}

.ms-error__actions {
  display: flex;
  width: 100%;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
</style>
