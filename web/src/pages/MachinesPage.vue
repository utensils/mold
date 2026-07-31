<script setup lang="ts">
/*
 * Machines workspace overview (spec §04 / §08 G1). One card per host — the
 * primary "this server" origin first, then remembered remotes — each polling
 * its own `/api/status` for the status dot, GPU line, memory, and queue. A
 * dashed "add machine" card and the header button open the connect wizard.
 * With no remotes the origin card plus the add card stand in for an empty
 * state (G4).
 */
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import CardSurface from "@ui/components/CardSurface.vue";
import Icon from "@ui/components/Icon.vue";
import HostCard from "../components/machines/HostCard.vue";
import ConnectModal from "../components/machines/ConnectModal.vue";
import {
  HOSTS_CHANGED_EVENT,
  HOSTS_STORAGE_KEY,
  ORIGIN_HOST_ID,
  listKnownHosts,
  setHostConnected,
  type HostEntry,
} from "../lib/hostRegistry";
import { toast } from "../lib/toasts";

const router = useRouter();
const route = useRoute();
const hosts = ref<HostEntry[]>(listKnownHosts());
const connectOpen = ref(false);
watch(
  () => route.query.add,
  (add) => {
    if (add === "1") connectOpen.value = true;
  },
  { immediate: true },
);

function refreshHosts() {
  hosts.value = listKnownHosts();
}

function onStorage(event: StorageEvent) {
  if (event.key === HOSTS_STORAGE_KEY) refreshHosts();
}

onMounted(() => {
  window.addEventListener(HOSTS_CHANGED_EVENT, refreshHosts);
  window.addEventListener("storage", onStorage);
});

onBeforeUnmount(() => {
  window.removeEventListener(HOSTS_CHANGED_EVENT, refreshHosts);
  window.removeEventListener("storage", onStorage);
});

function openDetail(id: string) {
  void router.push(`/machines/${id}`);
}

function onAdded(host: HostEntry) {
  refreshHosts();
  toast("success", `${host.name} connected.`);
}

function reconnect(id: string) {
  const host = setHostConnected(id, true);
  if (host) toast("success", `${host.name} reconnected.`);
}
</script>

<template>
  <div class="mx-auto max-w-[1800px] px-4 pb-40 pt-6 sm:px-6 lg:px-10">
    <div class="mb-5 flex items-center gap-4">
      <h1
        class="font-display text-2xl font-bold tracking-tight text-ink"
        data-test="machines-title"
      >
        Machines
      </h1>
      <div class="flex-1" />
      <button
        type="button"
        class="ms-addbtn"
        data-test="add-machine"
        @click="connectOpen = true"
      >
        + Add machine
      </button>
    </div>

    <div class="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
      <HostCard
        v-for="host in hosts"
        :key="host.id"
        :host="host"
        :primary="host.id === ORIGIN_HOST_ID"
        @open="openDetail"
        @reconnect="reconnect"
      />

      <CardSurface dashed>
        <button
          type="button"
          class="ms-addcard"
          data-test="add-machine-card"
          @click="connectOpen = true"
        >
          <span class="ms-addcard__icon"><Icon name="plus" :size="20" /></span>
          <span class="ms-addcard__title">Add a machine</span>
          <span class="ms-addcard__sub">
            Generate on another GPU you control
          </span>
        </button>
      </CardSurface>
    </div>

    <ConnectModal
      :open="connectOpen"
      @close="connectOpen = false"
      @added="onAdded"
    />
  </div>
</template>

<style scoped>
.ms-addbtn {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 8px 15px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.ms-addbtn:hover {
  border-color: var(--ink-3);
}

.ms-addcard {
  width: 100%;
  height: 100%;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  background: transparent;
  border: 0;
  color: var(--ink-2);
  cursor: pointer;
}

.ms-addcard__icon {
  color: var(--ink-3);
  margin-bottom: 2px;
}

.ms-addcard__title {
  font-size: 13.5px;
  font-weight: 600;
}

.ms-addcard__sub {
  font-size: 11.5px;
  color: var(--ink-3);
}
</style>
