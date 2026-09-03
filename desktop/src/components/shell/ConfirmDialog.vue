<script setup lang="ts">
/*
 * Shared confirm dialog (§08 G9 spend, G12 destructive). Blunt copy, a single
 * primary action, and a danger tone when the action is irreversible or starts
 * billing. Body content is a slot so callers can itemise (e.g. GPU + disk cost)
 * without a bespoke modal each time.
 */
import { onBeforeUnmount, watch } from "vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    title: string;
    message?: string;
    confirmLabel?: string;
    cancelLabel?: string;
    /** Stop-toned confirm button for spend / irreversible actions. */
    danger?: boolean;
    /** Keep the dialog up and the buttons disabled while the action runs. */
    busy?: boolean;
  }>(),
  { confirmLabel: "Confirm", cancelLabel: "Cancel", danger: false, busy: false },
);

const emit = defineEmits<{ confirm: []; cancel: [] }>();

function cancel() {
  if (!props.busy) emit("cancel");
}

function onKeydown(e: KeyboardEvent) {
  if (!props.open) return;
  if (e.key === "Escape") {
    e.preventDefault();
    cancel();
  }
}

watch(
  () => props.open,
  (open) => {
    if (open) window.addEventListener("keydown", onKeydown);
    else window.removeEventListener("keydown", onKeydown);
  },
  { immediate: true },
);
onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      data-test="confirm-dialog"
      class="fixed inset-0 z-50 flex justify-center bg-scrim pt-24"
      @click.self="cancel"
    >
      <div
        class="border-border h-fit w-[22rem] max-w-[calc(100%-2rem)] rounded-window border bg-bg p-4 shadow-xl"
        role="alertdialog"
        aria-modal="true"
        :aria-label="title"
      >
        <p
          class="font-sans font-semibold text-base font-semibold text-fg"
          style="font-stretch: 92%"
        >
          {{ title }}
        </p>
        <p v-if="message" class="mt-1.5 text-sm text-fg-2">{{ message }}</p>
        <div v-if="$slots.default" class="mt-3">
          <slot />
        </div>
        <div class="mt-4 flex justify-end gap-2">
          <button
            type="button"
            data-test="confirm-cancel"
            class="min-h-8 rounded-control px-3 py-1.5 text-center text-sm leading-tight text-fg-2 hover:text-fg disabled:opacity-50"
            :disabled="busy"
            @click="cancel"
          >
            {{ cancelLabel }}
          </button>
          <button
            type="button"
            data-test="confirm-accept"
            class="min-h-8 rounded-control px-3.5 py-1.5 text-center text-sm leading-tight font-semibold text-on-accent hover:brightness-105 active:translate-y-px disabled:opacity-50"
            :class="danger ? 'bg-error' : 'bg-accent'"
            :disabled="busy"
            @click="emit('confirm')"
          >
            {{ confirmLabel }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
