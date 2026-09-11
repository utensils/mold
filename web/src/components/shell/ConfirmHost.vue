<script setup lang="ts">
/*
 * The app-level home for whatever `requestConfirm` / `requestText` /
 * `requestChoice` queued. The DIALOG is the shared `@ui/ConfirmDialog`; this
 * is the part that is only the browser's — binding one queued request to it
 * and settling the promise that is waiting on the answer.
 *
 * Tab trapping, background-scroll locking and returning focus to the opener
 * are the browser's own overlay rules, so `useOverlayFocus` stays here too.
 */
import { computed, ref, watch } from "vue";
import ConfirmDialog from "@ui/components/ConfirmDialog.vue";
import { settleConfirm, useNotifications } from "../../lib/toasts";
import { useOverlayFocus } from "../../composables/useOverlayFocus";

const notifications = useNotifications();
const textValue = ref("");

const confirm = computed(() => notifications.confirm);
const host = ref<HTMLElement | null>(null);
const open = computed(() => confirm.value !== null);
const { onKeydown } = useOverlayFocus(open, host, () => cancel());

watch(confirm, (request) => {
  textValue.value = request?.inputInitial ?? "";
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
  <div v-if="confirm" ref="host" @keydown="onKeydown">
    <ConfirmDialog
      :open="true"
      :title="confirm.title"
      :message="confirm.body || undefined"
      :confirm-label="confirm.confirmLabel"
      :danger="confirm.danger"
      :input-label="confirm.inputLabel"
      :model-value="confirm.kind === 'text' ? textValue : undefined"
      :choices="confirm.kind === 'choice' ? confirm.choices : undefined"
      @update:model-value="textValue = $event"
      @confirm="accept"
      @cancel="cancel"
      @choose="settleConfirm($event)"
    />
  </div>
</template>
