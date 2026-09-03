<script setup lang="ts">
/*
 * Shared confirm dialog (§08 G9 spend, G12 destructive; README §04 dialog:
 * 480px, header / body / footer, radius-3, a 72% crust scrim). Blunt copy, a
 * single primary action, and a danger tone when the action is irreversible or
 * starts billing. Body content is a slot so callers can itemise (e.g. GPU +
 * disk cost) without a bespoke modal each time.
 */
import { onBeforeUnmount, watch } from "vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    title: string;
    message?: string;
    confirmLabel?: string;
    cancelLabel?: string;
    /** Error-toned confirm button for spend / irreversible actions. */
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
      class="fixed inset-0 z-50 flex items-center justify-center bg-scrim p-10"
      @click.self="cancel"
    >
      <div
        class="ms-fade-up flex w-[480px] max-w-full flex-col overflow-hidden rounded-window border border-border bg-surface shadow-md"
        role="alertdialog"
        aria-modal="true"
        :aria-label="title"
      >
        <div class="flex flex-col gap-1.5 border-b border-border p-4">
          <p class="text-md font-semibold text-fg">{{ title }}</p>
          <p v-if="message" class="text-xs text-fg-dim">{{ message }}</p>
        </div>
        <div v-if="$slots.default" class="p-4">
          <slot />
        </div>
        <div class="flex justify-end gap-2 border-t border-border px-4 py-3.5">
          <button
            type="button"
            data-test="confirm-cancel"
            class="min-h-8 rounded-control border border-border px-3.5 py-1.5 text-center text-xs leading-tight text-fg-2 transition-colors duration-100 hover:border-border-focus hover:text-fg disabled:opacity-50"
            :disabled="busy"
            @click="cancel"
          >
            {{ cancelLabel }}
          </button>
          <button
            type="button"
            data-test="confirm-accept"
            class="min-h-8 rounded-control px-3.5 py-1.5 text-center text-xs leading-tight font-semibold text-on-accent hover:brightness-105 active:translate-y-px disabled:opacity-50"
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
