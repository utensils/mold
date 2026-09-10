<script setup lang="ts">
import { ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import type { HeldQueueTransferController } from "../composables/useHeldQueueTransfer";
import { sendHeldQueueJob } from "../api/queueTransfer";
const props = defineProps<{ controller: HeldQueueTransferController }>();
const emit = defineEmits<{ (event: "sent"): void }>();
const selected = ref("");
const busy = props.controller.busy;
const message = ref("");
const error = ref("");
const done = ref(false);
watch(
  () => props.controller.selection.value,
  () => {
    selected.value = props.controller.destinations.value[0]?.id ?? "";
    message.value = "";
    error.value = "";
    done.value = false;
  },
);
function close() {
  props.controller.close();
}
async function send() {
  const selection = props.controller.selection.value;
  const destination = props.controller.destinations.value.find(
    (host) => host.id === selected.value,
  );
  if (!selection || !destination || busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    const result = await sendHeldQueueJob({
      source: selection.source,
      jobId: selection.jobId,
      destination: { ...destination, target: { ...destination.target } },
      onProgress: (text) => {
        message.value = text;
      },
    });
    message.value = result.message;
    done.value = true;
    emit("sent");
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
    message.value =
      "The original stays on its machine until the destination confirms acceptance.";
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <Teleport to="body"
    ><div v-if="controller.selection.value" class="held-transfer-layer">
      <ModalPanel
        :open="controller.selection.value !== null"
        label="Send to another machine"
        :width="480"
        @close="close"
      >
        <div class="held-transfer">
          <h2>Send to another machine</h2>
          <p v-if="!done">
            Keep the same prompt, seed, settings, and reference media. The
            destination queues the job; the held original is removed after
            acceptance.
          </p>
          <label v-if="!done"
            >Machine
            <select
              v-model="selected"
              :disabled="busy"
              data-test="transfer-host"
            >
              <option
                v-for="host in controller.destinations.value"
                :key="host.id"
                :value="host.id"
              >
                {{ host.label
                }}{{
                  host.gpuCount
                    ? ` · ${host.gpuCount} GPU${host.gpuCount === 1 ? "" : "s"}`
                    : ""
                }}{{
                  host.queueDepth != null ? ` · ${host.queueDepth} queued` : ""
                }}
              </option>
            </select>
          </label>
          <p v-if="message" role="status">{{ message }}</p>
          <p v-if="error" role="alert" class="held-transfer__error">
            {{ error }}
          </p>
          <div class="held-transfer__actions">
            <button type="button" :disabled="busy" @click="close">
              {{ done ? "Done" : "Cancel" }}
            </button>
            <button
              v-if="!done"
              type="button"
              :disabled="busy || !selected"
              data-test="transfer-send"
              @click="send"
            >
              {{ busy ? "Sending…" : error ? "Try again" : "Send job" }}
            </button>
          </div>
        </div>
      </ModalPanel>
    </div></Teleport
  >
</template>
<style scoped>
.held-transfer-layer {
  position: fixed;
  inset: 0;
  z-index: 500;
}
.held-transfer-layer :deep(.ms-modal) {
  padding: 12px;
}
.held-transfer-layer :deep(.ms-modal__panel) {
  max-width: 100%;
  max-height: 100%;
  overflow: auto;
}

.held-transfer {
  display: grid;
  gap: 16px;
  padding: 20px;
  color: var(--mold-text);
}
.held-transfer h2 {
  font-size: var(--mold-fs-lg);
  font-weight: 600;
}
.held-transfer p {
  line-height: 1.5;
}
.held-transfer label {
  display: grid;
  gap: 8px;
}
.held-transfer select,
.held-transfer button {
  min-height: 48px;
  padding: 8px 12px;
  font: inherit;
  font-size: max(
    16px,
    var(--mold-fs-md)
  ); /* literal: prevent iOS input focus zoom */
  color: var(--mold-text);
  background: var(--mold-bg-deep);
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
}
.held-transfer__actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
}
.held-transfer__error {
  color: var(--mold-error);
}
</style>
