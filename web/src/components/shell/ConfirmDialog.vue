<script setup lang="ts">
import { computed, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import { settleConfirm, useNotifications } from "../../lib/toasts";

/*
 * App-level dialog for destructive actions and small decisions (spec §08
 * G12) — renders whatever requestConfirm / requestText / requestChoice
 * queued. Irreversible bulk actions may require a typed phrase before the
 * confirm button arms.
 */

const notifications = useNotifications();
const typed = ref("");
const textValue = ref("");

const confirm = computed(() => notifications.confirm);

watch(confirm, (request) => {
  typed.value = "";
  textValue.value = request?.inputInitial ?? "";
});

const armed = computed(() => {
  const request = confirm.value;
  if (!request) return false;
  if (!request.typedPhrase) return true;
  return typed.value.trim() === request.typedPhrase;
});

function cancel() {
  const request = confirm.value;
  if (!request) return;
  settleConfirm(request.kind === "confirm" ? false : null);
}

function accept() {
  const request = confirm.value;
  if (!request) return;
  settleConfirm(request.kind === "text" ? textValue.value : true);
}
</script>

<template>
  <ModalPanel
    v-if="confirm"
    :open="true"
    :width="420"
    :label="confirm.title"
    @close="cancel"
  >
    <div class="confirm">
      <div class="confirm__title">{{ confirm.title }}</div>
      <p v-if="confirm.body" class="confirm__body">{{ confirm.body }}</p>
      <label v-if="confirm.typedPhrase" class="confirm__typed">
        <span>
          Type <code>{{ confirm.typedPhrase }}</code> to continue
        </span>
        <input
          v-model="typed"
          type="text"
          autocomplete="off"
          spellcheck="false"
          data-test="confirm-typed"
        />
      </label>
      <label v-if="confirm.kind === 'text'" class="confirm__typed">
        <span v-if="confirm.inputLabel">{{ confirm.inputLabel }}</span>
        <input
          v-model="textValue"
          type="text"
          autocomplete="off"
          spellcheck="false"
          data-test="dialog-text"
          @keydown.enter.prevent="accept"
        />
      </label>
      <div v-if="confirm.kind === 'choice'" class="confirm__choices">
        <button
          v-for="choice in confirm.choices"
          :key="choice.id"
          type="button"
          class="confirm__choice"
          :class="{ 'confirm__choice--danger': choice.danger }"
          :data-test="`dialog-choice-${choice.id}`"
          @click="settleConfirm(choice.id)"
        >
          {{ choice.label }}
        </button>
      </div>
    </div>
    <template #footer>
      <button
        type="button"
        class="confirm__cancel"
        data-test="confirm-cancel"
        @click="cancel"
      >
        Cancel
      </button>
      <span class="confirm__spacer" />
      <button
        v-if="confirm.kind !== 'choice'"
        type="button"
        class="confirm__go"
        :class="{ 'confirm__go--danger': confirm.danger }"
        :disabled="!armed"
        data-test="confirm-accept"
        @click="accept"
      >
        {{ confirm.confirmLabel }}
      </button>
    </template>
  </ModalPanel>
</template>

<style scoped>
.confirm__title {
  font-family: var(--f-display);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.confirm__body {
  margin: 6px 0 0;
  font-size: 13px;
  color: var(--ink-2);
  line-height: 1.5;
}

.confirm__typed {
  display: flex;
  flex-direction: column;
  gap: 7px;
  margin-top: 14px;
  font-size: 12px;
  color: var(--ink-3);
}

.confirm__typed code {
  font-family: var(--f-mono);
  color: var(--stop);
}

.confirm__typed input {
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 13px;
  padding: 9px 12px;
  outline: none;
}

.confirm__typed input:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.confirm__choices {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
}

.confirm__choice {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 11px 14px;
  border-radius: var(--radius-control-lg);
  font-size: 13px;
  font-weight: 600;
  text-align: left;
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}

.confirm__choice:hover {
  background: var(--surface);
}

.confirm__choice--danger {
  color: var(--stop);
  border-color: color-mix(in srgb, var(--stop) 50%, transparent);
}

.confirm__cancel {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 10px 16px;
  border-radius: var(--radius-control-lg);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.confirm__spacer {
  flex: 1;
}

.confirm__go {
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 10px 20px;
  border-radius: var(--radius-control-lg);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}

.confirm__go--danger {
  background: var(--stop);
}

.confirm__go:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
