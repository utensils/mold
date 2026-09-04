<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { ApiTarget } from "../api/client";
import {
  listPairedClients,
  revokePairedClient,
  type PairedClient,
  type PairedClientsResponse,
} from "../api/pairing";
import MobilePairingCard from "./MobilePairingCard.vue";

const props = defineProps<{
  target: ApiTarget | null;
  suggestedBaseUrl: string;
  hostLabel?: string | undefined;
}>();

const access = ref<PairedClientsResponse | null>(null);
const loading = ref(false);
const error = ref("");
const confirmingId = ref<string | null>(null);
const revokingId = ref<string | null>(null);
let claimPoll: ReturnType<typeof setInterval> | null = null;
let claimPollEndsAt = 0;
let requestEpoch = 0;

function formatTime(value: number | null): string {
  if (value === null) return "Not used yet";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function clientActivity(client: PairedClient): string {
  const paired = `Paired ${formatTime(client.created_at_ms)}`;
  const used =
    client.last_used_at_ms === null
      ? "Not used yet"
      : `Last used ${formatTime(client.last_used_at_ms)}`;
  return `${paired} · ${used}`;
}

async function load(): Promise<void> {
  const target = props.target;
  if (!target || revokingId.value) return;
  const epoch = ++requestEpoch;
  loading.value = true;
  error.value = "";
  try {
    const next = await listPairedClients(target);
    if (epoch === requestEpoch) access.value = next;
  } catch (reason) {
    if (epoch === requestEpoch) {
      error.value = reason instanceof Error ? reason.message : String(reason);
    }
  } finally {
    if (epoch === requestEpoch) loading.value = false;
  }
}

function watchForClaim(): void {
  claimPollEndsAt = Date.now() + 125_000;
  if (claimPoll) return;
  claimPoll = setInterval(() => {
    if (Date.now() >= claimPollEndsAt) {
      clearInterval(claimPoll!);
      claimPoll = null;
      return;
    }
    void load();
  }, 3_000);
}

async function revoke(client: PairedClient): Promise<void> {
  if (confirmingId.value !== client.id) {
    confirmingId.value = client.id;
    return;
  }
  const target = props.target;
  if (!target || revokingId.value) return;
  const epoch = ++requestEpoch;
  revokingId.value = client.id;
  error.value = "";
  try {
    await revokePairedClient(target, client.id);
    if (epoch === requestEpoch) {
      access.value = access.value
        ? {
            ...access.value,
            clients: access.value.clients.filter((row) => row.id !== client.id),
          }
        : null;
      confirmingId.value = null;
    }
  } catch (reason) {
    if (epoch === requestEpoch) {
      error.value = reason instanceof Error ? reason.message : String(reason);
    }
  } finally {
    if (epoch === requestEpoch) revokingId.value = null;
  }
}

watch(
  () => props.target,
  () => {
    requestEpoch += 1;
    access.value = null;
    loading.value = false;
    confirmingId.value = null;
    revokingId.value = null;
    void load();
  },
);
onMounted(() => void load());
onBeforeUnmount(() => {
  if (claimPoll) clearInterval(claimPoll);
});
</script>

<template>
  <div class="access-panel">
    <MobilePairingCard
      :target="target"
      :suggested-base-url="suggestedBaseUrl"
      :host-label="hostLabel"
      @session-created="watchForClaim"
    />

    <div class="access-card" data-test="paired-access-card">
      <div class="access-header">
        <div>
          <h3>Paired access</h3>
          <p>
            Each device has its own credential. Revoking it blocks new requests;
            open media links expire within 15 minutes.
          </p>
        </div>
        <button
          type="button"
          class="refresh-button"
          :disabled="!target || loading"
          data-test="paired-access-refresh"
          @click="load"
        >
          {{ loading ? "Refreshing…" : "Refresh" }}
        </button>
      </div>

      <p v-if="error" class="access-error" role="alert">{{ error }}</p>
      <p v-else-if="access?.auth_required === false" class="access-empty">
        Access control is off on this host, so devices do not need credentials.
      </p>
      <p v-else-if="access?.pairing_available === false" class="access-empty">
        Paired access needs the metadata database enabled on this host.
      </p>
      <p v-else-if="access && access.clients.length === 0" class="access-empty">
        No devices have paired yet.
      </p>
      <div v-else-if="access" class="access-list">
        <div
          v-for="client in access.clients"
          :key="client.id"
          class="access-row"
        >
          <div class="client-icon" aria-hidden="true">
            {{ client.client_kind === "ipad" ? "▭" : "▯" }}
          </div>
          <div class="client-copy">
            <strong>{{ client.name }}</strong>
            <span>{{ clientActivity(client) }}</span>
          </div>
          <div class="access-actions">
            <button
              v-if="confirmingId === client.id"
              type="button"
              class="cancel-button"
              @click="confirmingId = null"
            >
              Cancel
            </button>
            <button
              type="button"
              class="revoke-button"
              :class="{ 'revoke-button--confirm': confirmingId === client.id }"
              :disabled="revokingId !== null"
              :data-test="`revoke-paired-${client.id}`"
              @click="revoke(client)"
            >
              {{
                revokingId === client.id
                  ? "Revoking…"
                  : confirmingId === client.id
                    ? `Revoke ${client.name}?`
                    : "Revoke"
              }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.access-panel {
  display: grid;
  gap: 12px;
}
.access-card {
  padding: 20px;
  border: 1px solid var(--mold-border);
  border-radius: 14px;
  background: var(--mold-bg-deep);
}
.access-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}
h3 {
  margin: 0;
  color: var(--mold-text);
  font-size: 15px;
  font-weight: 650;
}
p {
  margin: 4px 0 0;
  color: var(--mold-text-dim);
  font-size: 12px;
  line-height: 1.45;
}
button {
  min-height: 36px;
  border-radius: 8px;
  font: inherit;
  font-weight: 650;
  cursor: pointer;
}
button:disabled {
  opacity: 0.5;
  cursor: default;
}
.refresh-button,
.cancel-button {
  border: 1px solid var(--mold-border);
  padding: 0 12px;
  background: var(--mold-surface);
  color: var(--mold-text-2);
}
.access-error {
  color: var(--mold-error);
}
.access-empty {
  margin-top: 18px;
}
.access-list {
  display: grid;
  margin-top: 14px;
}
.access-row {
  display: flex;
  min-height: 58px;
  align-items: center;
  gap: 12px;
  border-top: 1px solid var(--mold-border);
}
.client-icon {
  display: grid;
  width: 32px;
  height: 32px;
  place-items: center;
  border-radius: 9px;
  background: color-mix(in srgb, var(--mold-blue) 14%, transparent);
  color: var(--mold-blue);
  font-size: 20px;
}
.client-copy {
  display: grid;
  min-width: 0;
  flex: 1;
  gap: 2px;
}
.client-copy strong {
  overflow: hidden;
  color: var(--mold-text);
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.client-copy span {
  color: var(--mold-text-dim);
  font-size: 11px;
}
.access-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}
.revoke-button {
  border: 1px solid color-mix(in srgb, var(--mold-error) 45%, transparent);
  padding: 0 12px;
  background: transparent;
  color: var(--mold-error);
}
.revoke-button--confirm {
  border-color: var(--mold-error);
  background: var(--mold-error);
  color: var(--mold-on-accent);
}
@media (max-width: 560px) {
  .access-row {
    align-items: flex-start;
    flex-wrap: wrap;
    padding: 12px 0;
  }
  .access-actions {
    width: 100%;
    justify-content: flex-end;
  }
}
</style>
