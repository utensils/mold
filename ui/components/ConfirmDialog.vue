<script setup lang="ts">
/*
 * The confirm dialog, for every surface: blunt copy, one primary action, a
 * danger tone when the action is irreversible or starts billing.
 *
 * A destructive question is a PLAIN question. The browser's copy could arm its
 * confirm behind a typed phrase; nothing asks for one any more and this does
 * not carry it across — a danger-toned button on a sentence that says what is
 * lost is the whole contract.
 *
 * Three shapes, one dialog, because the browser's app-level queue asks all
 * three of them: a yes/no, a one-line answer (`modelValue` present), and a
 * short list of alternatives (`choices` present, which replaces the confirm
 * button since every answer is one of them). Body content stays a slot so a
 * caller can itemise — a GPU and its hourly rate — without a bespoke modal.
 */
import ModalPanel from "./ModalPanel.vue";

export interface ConfirmChoice {
  id: string;
  label: string;
  danger?: boolean;
}

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
    /** Present = the question wants one line of text back. */
    modelValue?: string;
    /** Caption above that field. */
    inputLabel?: string;
    /** Present = the question has more than two answers. */
    choices?: readonly ConfirmChoice[];
  }>(),
  {
    confirmLabel: "Confirm",
    cancelLabel: "Cancel",
    danger: false,
    busy: false,
  },
);

const emit = defineEmits<{
  confirm: [];
  cancel: [];
  choose: [id: string];
  "update:modelValue": [value: string];
}>();

function cancel() {
  if (!props.busy) emit("cancel");
}

function onInput(event: Event) {
  emit("update:modelValue", (event.target as HTMLInputElement).value);
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
    <template v-if="$slots.default || modelValue !== undefined || choices">
      <slot />
      <label v-if="modelValue !== undefined" class="confirm-field">
        <span v-if="inputLabel">{{ inputLabel }}</span>
        <input
          type="text"
          autocomplete="off"
          spellcheck="false"
          data-test="dialog-text"
          :value="modelValue"
          @input="onInput"
          @keydown.enter.prevent="emit('confirm')"
        />
      </label>
      <div v-if="choices" class="confirm-choices">
        <button
          v-for="choice in choices"
          :key="choice.id"
          type="button"
          class="confirm-choice"
          :data-danger="choice.danger ? 'true' : undefined"
          :data-test="`dialog-choice-${choice.id}`"
          @click="emit('choose', choice.id)"
        >
          {{ choice.label }}
        </button>
      </div>
    </template>
    <template #footer>
      <button
        type="button"
        class="confirm-button"
        data-test="confirm-cancel"
        :disabled="busy"
        @click="cancel"
      >
        {{ cancelLabel }}
      </button>
      <button
        v-if="!choices"
        type="button"
        class="confirm-button confirm-button--go"
        data-test="confirm-accept"
        :data-danger="danger ? 'true' : undefined"
        :disabled="busy"
        @click="emit('confirm')"
      >
        {{ confirmLabel }}
      </button>
    </template>
  </ModalPanel>
</template>

<style scoped>
/* A long label wraps: a minimum height and vertical padding, never a fixed
   height — "Start it — billing begins now" used to be clipped. */
.confirm-button {
  min-height: 32px;
  box-sizing: border-box;
  padding: 6px 14px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-1);
  background: transparent;
  color: var(--mold-text-2);
  font-family: inherit;
  font-size: var(--mold-fs-xs);
  line-height: var(--mold-lh-snug);
  text-align: center;
  cursor: pointer;
  transition: border-color var(--mold-dur-quick) var(--mold-ease-out);
}
.confirm-button:hover:not(:disabled) {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}
.confirm-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.confirm-button--go {
  border-color: transparent;
  background: var(--mold-blue);
  color: var(--mold-on-accent);
  font-weight: 600;
}
.confirm-button--go[data-danger] {
  background: var(--mold-error);
}
.confirm-button--go:hover:not(:disabled) {
  border-color: transparent;
  color: var(--mold-on-accent);
  filter: brightness(1.05);
}
.confirm-button--go:active:not(:disabled) {
  transform: translateY(1px);
}

.confirm-field {
  display: flex;
  flex-direction: column;
  gap: 7px;
  margin-top: 14px;
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
}
.confirm-field input {
  padding: 9px 12px;
  border: var(--mold-bw) solid var(--mold-border-control);
  border-radius: var(--mold-radius-1);
  background: var(--mold-surface);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-sm);
  outline: none;
}
.confirm-field input:focus-visible {
  outline: 2px solid var(--mold-border-focus);
  outline-offset: 2px;
}

.confirm-choices {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
}
.confirm-choice {
  padding: 11px 14px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-text);
  font-family: inherit;
  font-size: var(--mold-fs-sm);
  font-weight: 600;
  text-align: left;
  cursor: pointer;
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}
.confirm-choice:hover {
  background: var(--mold-surface);
}
.confirm-choice[data-danger] {
  color: var(--mold-error);
  border-color: color-mix(in srgb, var(--mold-error) 50%, transparent);
}
</style>
