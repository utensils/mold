<script setup lang="ts">
/*
 * Shared confirm dialog: blunt copy, a single primary action, and a danger tone
 * when the action is irreversible or starts billing. Body content is a slot so
 * callers can itemise (e.g. GPU + disk cost) without a bespoke modal each time.
 */
import ModalPanel from "@ui/components/ModalPanel.vue";

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
</script>

<template>
  <ModalPanel
    :open="open"
    :width="480"
    role="alertdialog"
    :title="title"
    :description="message"
    data-test="confirm-dialog"
    @close="cancel"
  >
    <template v-if="$slots.default" #default><slot /></template>
    <template #footer>
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
    </template>
  </ModalPanel>
</template>
