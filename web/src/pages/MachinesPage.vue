<script setup lang="ts">
/*
 * Machines workspace overview (spec §04 / §08 G1). One card per host — the
 * primary "this server" origin first, then remembered remotes — each polling
 * its own `/api/status` for the status dot, GPU line, memory, and queue. A
 * dashed "add machine" card and the header button open the connect wizard.
 * With no remotes the origin card plus the add card stand in for an empty
 * state (G4).
 */
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import CardSurface from "@ui/components/CardSurface.vue";
import Icon from "@ui/components/Icon.vue";
import HostCard from "../components/machines/HostCard.vue";
import ConnectModal from "../components/machines/ConnectModal.vue";
import {
  HOSTS_CHANGED_EVENT,
  HOSTS_STORAGE_KEY,
  ORIGIN_HOST_ID,
  getGenerateTargetId,
  listKnownHosts,
  originHost,
  reconcileOriginInstanceId,
  removeHost,
  setHostConnected,
  setGenerateTargetId,
  type HostEntry,
} from "../lib/hostRegistry";
import { hostStatus } from "../components/machines/hostClient";
import { requestConfirm, toast } from "../lib/toasts";

const router = useRouter();
const route = useRoute();
const hosts = ref<HostEntry[]>(listKnownHosts());
const connectOpen = ref(false);
type MachineMenu = {
  host: HostEntry;
  x: number;
  y: number;
  opener: HTMLElement;
};
const contextMenu = ref<MachineMenu | null>(null);
const contextMenuElement = ref<HTMLElement | null>(null);
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

function closeContextMenu(restoreFocus = true) {
  const opener = contextMenu.value?.opener;
  contextMenu.value = null;
  if (restoreFocus && opener?.isConnected) opener.focus();
}

function menuItems() {
  return Array.from(
    contextMenuElement.value?.querySelectorAll<HTMLButtonElement>(
      "button:not(:disabled)",
    ) ?? [],
  );
}

function onMenuKey(event: KeyboardEvent) {
  if (event.key === "Tab") {
    closeContextMenu();
    return;
  }
  const items = menuItems();
  const index = items.indexOf(document.activeElement as HTMLButtonElement);
  let next: number;
  if (event.key === "ArrowDown") next = (index + 1) % items.length;
  else if (event.key === "ArrowUp")
    next = (index - 1 + items.length) % items.length;
  else if (event.key === "Home") next = 0;
  else if (event.key === "End") next = items.length - 1;
  else return;
  event.preventDefault();
  items[next]?.focus();
}

function onWindowResize() {
  closeContextMenu();
}

function onDocumentPointer(event: PointerEvent) {
  const target = event.target as HTMLElement | null;
  if (!target?.closest("[data-test='machine-context-menu']"))
    closeContextMenu(false);
}

function onWindowKey(event: KeyboardEvent) {
  if (event.key === "Escape" && contextMenu.value && !event.defaultPrevented) {
    event.preventDefault();
    closeContextMenu();
  }
}

onMounted(() => {
  window.addEventListener(HOSTS_CHANGED_EVENT, refreshHosts);
  window.addEventListener("storage", onStorage);
  document.addEventListener("pointerdown", onDocumentPointer);
  window.addEventListener("keydown", onWindowKey);
  window.addEventListener("resize", onWindowResize);
  void hostStatus(originHost())
    .then((status) => {
      reconcileOriginInstanceId(status.instance_id ?? "");
      refreshHosts();
    })
    .catch(() => undefined);
});

onBeforeUnmount(() => {
  window.removeEventListener(HOSTS_CHANGED_EVENT, refreshHosts);
  window.removeEventListener("storage", onStorage);
  document.removeEventListener("pointerdown", onDocumentPointer);
  window.removeEventListener("keydown", onWindowKey);
  window.removeEventListener("resize", onWindowResize);
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

async function openHostContext(payload: MachineMenu) {
  contextMenu.value = payload;
  await nextTick();
  if (contextMenu.value?.opener !== payload.opener) return;
  const rect = contextMenuElement.value?.getBoundingClientRect();
  if (!rect) return;
  contextMenu.value = {
    ...payload,
    x: Math.max(8, Math.min(payload.x, window.innerWidth - rect.width - 8)),
    y: Math.max(8, Math.min(payload.y, window.innerHeight - rect.height - 8)),
  };
  menuItems()[0]?.focus();
}

function contextOpen() {
  const host = contextMenu.value?.host;
  closeContextMenu();
  if (host && host.connected !== false) openDetail(host.id);
}

function contextTarget() {
  const host = contextMenu.value?.host;
  closeContextMenu();
  if (!host || host.connected === false) return;
  setGenerateTargetId(host.id);
  toast("success", `${host.name} is now the generation target.`);
}

async function contextCopyAddress() {
  const host = contextMenu.value?.host;
  closeContextMenu();
  if (!host) return;
  try {
    await navigator.clipboard.writeText(host.url);
    toast("success", "Address copied.");
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  }
}

function contextConnect() {
  const host = contextMenu.value?.host;
  closeContextMenu();
  if (!host) return;
  const connected = setHostConnected(host.id, true);
  if (connected) toast("success", `${connected.name} connected.`);
}

function contextDisconnect() {
  const host = contextMenu.value?.host;
  closeContextMenu();
  if (!host || host.id === ORIGIN_HOST_ID) return;
  setHostConnected(host.id, false);
  if (getGenerateTargetId() === host.id) setGenerateTargetId(ORIGIN_HOST_ID);
  refreshHosts();
  toast("success", `${host.name} disconnected.`);
}

async function contextForget() {
  const host = contextMenu.value?.host;
  closeContextMenu();
  if (!host || host.id === ORIGIN_HOST_ID) return;
  const accepted = await requestConfirm({
    title: "Forget this machine?",
    body: `${host.name} and its saved API key will be removed from this browser.`,
    confirmLabel: "Forget machine",
    danger: true,
  });
  if (!accepted) return;
  removeHost(host.id);
  if (getGenerateTargetId() === host.id) setGenerateTargetId(ORIGIN_HOST_ID);
  refreshHosts();
  toast("success", `${host.name} forgotten.`);
}
</script>

<template>
  <!-- w-full is load-bearing: the app frame is a column flexbox and an
       mx-auto child otherwise shrinks to its content width, making the page
       grow sideways as async content lands. -->
  <div class="mx-auto w-full max-w-[1800px] px-4 pb-40 pt-6 sm:px-6 lg:px-10">
    <div class="mb-5 flex flex-wrap items-center gap-4">
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
        :actions-open="contextMenu?.host.id === host.id"
        @open="openDetail"
        @reconnect="reconnect"
        @context-menu="openHostContext"
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

    <div
      v-if="contextMenu"
      ref="contextMenuElement"
      class="machine-context"
      :aria-label="`Actions for ${contextMenu.host.name}`"
      @keydown="onMenuKey"
      data-test="machine-context-menu"
      role="menu"
      :style="{ left: `${contextMenu.x}px`, top: `${contextMenu.y}px` }"
    >
      <button
        type="button"
        role="menuitem"
        :disabled="contextMenu.host.connected === false"
        @click="contextOpen"
      >
        Open details
      </button>
      <button
        type="button"
        role="menuitem"
        :disabled="
          contextMenu.host.connected === false ||
          getGenerateTargetId() === contextMenu.host.id
        "
        @click="contextTarget"
      >
        {{
          getGenerateTargetId() === contextMenu.host.id
            ? "Generation target"
            : "Set as generation target"
        }}
      </button>
      <button type="button" role="menuitem" @click="contextCopyAddress">
        Copy address
      </button>
      <div
        v-if="contextMenu.host.id !== ORIGIN_HOST_ID"
        class="machine-context__separator"
        role="separator"
      />
      <button
        v-if="
          contextMenu.host.id !== ORIGIN_HOST_ID &&
          contextMenu.host.connected === false
        "
        type="button"
        role="menuitem"
        @click="contextConnect"
      >
        Connect
      </button>
      <button
        v-if="
          contextMenu.host.id !== ORIGIN_HOST_ID &&
          contextMenu.host.connected !== false
        "
        type="button"
        role="menuitem"
        @click="contextDisconnect"
      >
        Disconnect
      </button>
      <button
        v-if="contextMenu.host.id !== ORIGIN_HOST_ID"
        type="button"
        role="menuitem"
        class="machine-context__danger"
        @click="contextForget"
      >
        Forget…
      </button>
    </div>
  </div>
</template>

<style scoped>
.ms-addbtn {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 8px 15px;
  min-height: 44px;
  border-radius: 8px;
  font-size: 0.875rem;
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
  font-size: 0.875rem;
  font-weight: 600;
}

.ms-addcard__sub {
  font-size: 0.875rem;
  color: var(--ink-3);
}

.machine-context {
  position: fixed;
  z-index: 80;
  width: max-content;
  max-width: calc(100vw - 16px);
  max-height: calc(100svh - 16px);
  overflow: auto;
  box-sizing: border-box;
  padding: 5px;
  border: 1px solid var(--ce);
  border-radius: 9px;
  background: var(--bench);
  box-shadow: 0 14px 36px color-mix(in srgb, var(--bath) 50%, transparent);
}

.machine-context button {
  display: block;
  width: 100%;
  min-height: 44px;
  padding: 8px 10px;
  overflow-wrap: anywhere;
  border: 0;
  border-radius: 5px;
  background: transparent;
  color: var(--rebate);
  text-align: left;
  font-size: 0.875rem;
  cursor: pointer;
}

.machine-context button:hover:not(:disabled) {
  background: color-mix(in srgb, var(--safelight) 14%, transparent);
}

.machine-context button:disabled {
  color: var(--ink-3);
  cursor: default;
}

.machine-context__separator {
  height: 1px;
  margin: 4px 6px;
  background: var(--ce);
}

.machine-context .machine-context__danger {
  color: var(--stop);
}
</style>
