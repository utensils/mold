<script setup lang="ts">
/*
 * Connect-a-machine wizard (spec §08 G1, prototype CONNECT MODAL). Three
 * stepped panes: pick a type, enter connection details or choose a discovered
 * peer, then confirm. LAN discovery is server-assisted and only appears when
 * the primary server advertises the capability.
 *
 * Every path probes the peer's `/api/status` directly with its own key before
 * saving. On success the host is deduped by instance id in the registry.
 */
import { computed, onBeforeUnmount, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import Icon from "@ui/components/Icon.vue";
import {
  hostCapabilities,
  hostDiscoveryPeers,
  hostStatus,
  type DiscoveryPeer,
} from "./hostClient";
import {
  addHost,
  dedupeByInstanceId,
  hostIdFromUrl,
  listStoredHosts,
  normalizeHostAddress,
  originHost,
  type HostEntry,
} from "../../lib/hostRegistry";
import { useOverlayFocus } from "../../composables/useOverlayFocus";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: []; added: [host: HostEntry] }>();

const step = ref<1 | 2 | 3>(1);
const mode = ref<"remote" | "lan">("remote");
const address = ref("");
const name = ref("");
const apiKey = ref("");
const probing = ref(false);
const error = ref<string | null>(null);
const discoveryError = ref<string | null>(null);
const connected = ref<HostEntry | null>(null);
const discoveryAvailable = ref(false);
const checkingDiscovery = ref(false);
const scanning = ref(false);
const discovered = ref<DiscoveryPeer[]>([]);
const selectedPeer = ref<DiscoveryPeer | null>(null);
let openSequence = 0;
let capabilitySequence = 0;
let discoveryRefreshInFlight = false;
let discoverySequence = 0;
let discoveryTimer: ReturnType<typeof setInterval> | null = null;
const host = ref<HTMLElement | { $el?: unknown } | null>(null);
const isOpen = computed(() => props.open);
const { onKeydown } = useOverlayFocus(isOpen, host, () => close());

const visiblePeers = computed(() => {
  const storedIds = new Set(listStoredHosts().map((host) => host.id));
  const seenInstanceIds = new Set<string>();
  return discovered.value.filter((peer) => {
    const instanceId = peer.instance_id?.trim() || null;
    const visible =
      !peer.is_this_machine &&
      !(instanceId && dedupeByInstanceId(instanceId)) &&
      !storedIds.has(hostIdFromUrl(peer.url)) &&
      !(instanceId && seenInstanceIds.has(instanceId));
    if (visible && instanceId) seenInstanceIds.add(instanceId);
    return visible;
  });
});

function stopDiscoveryRefresh() {
  if (discoveryTimer !== null) clearInterval(discoveryTimer);
  discoveryTimer = null;
}

function invalidateStep() {
  ++openSequence;
  ++discoverySequence;
  discoveryRefreshInFlight = false;
  probing.value = false;
  scanning.value = false;
  stopDiscoveryRefresh();
}

function reset() {
  stopDiscoveryRefresh();
  step.value = 1;
  mode.value = "remote";
  address.value = "";
  name.value = "";
  apiKey.value = "";
  probing.value = false;
  error.value = null;
  connected.value = null;
  discoveryAvailable.value = false;
  checkingDiscovery.value = true;
  scanning.value = false;
  discovered.value = [];
  discoveryError.value = null;
  selectedPeer.value = null;
}

watch(
  () => props.open,
  async (open) => {
    invalidateStep();
    const sequence = ++capabilitySequence;
    if (!open) return;
    reset();
    const capabilities = await hostCapabilities(originHost());
    if (sequence !== capabilitySequence || !props.open) return;
    discoveryAvailable.value = capabilities.discovery?.can_browse === true;
    checkingDiscovery.value = false;
  },
  { immediate: true },
);

function close() {
  ++capabilitySequence;
  invalidateStep();
  emit("close");
}

function toStep2() {
  invalidateStep();
  mode.value = "remote";
  error.value = null;
  step.value = 2;
}

async function refreshDiscovery(initial = false) {
  if (
    discoveryRefreshInFlight ||
    !props.open ||
    mode.value !== "lan" ||
    step.value !== 2
  )
    return;
  const sequence = ++discoverySequence;
  const owner = openSequence;
  const current = () =>
    sequence === discoverySequence &&
    owner === openSequence &&
    props.open &&
    mode.value === "lan" &&
    step.value === 2;
  discoveryRefreshInFlight = true;
  if (initial) scanning.value = true;
  try {
    const peers = await hostDiscoveryPeers(originHost());
    if (!current()) return;
    discovered.value = peers;
    discoveryError.value = null;
  } catch {
    if (current())
      discoveryError.value =
        "Couldn't scan this server's local network. Try again.";
  } finally {
    if (current()) {
      discoveryRefreshInFlight = false;
      scanning.value = false;
    }
  }
}

async function browseLan() {
  if (!discoveryAvailable.value) return;
  invalidateStep();
  const sequence = openSequence;
  mode.value = "lan";
  selectedPeer.value = null;
  apiKey.value = "";
  error.value = null;
  step.value = 2;
  await refreshDiscovery(true);
  if (
    sequence !== openSequence ||
    !props.open ||
    step.value !== 2 ||
    mode.value !== "lan"
  )
    return;
  stopDiscoveryRefresh();
  discoveryTimer = setInterval(() => {
    if (props.open && mode.value === "lan" && step.value === 2) {
      void refreshDiscovery();
    }
  }, 2000);
}

function back() {
  invalidateStep();
  error.value = null;
  selectedPeer.value = null;
  step.value = 1;
}

function describeError(message: string, url: string): string {
  if (message.includes(" 401") || message.includes(" 403")) {
    return "Authentication failed — check the API key.";
  }
  if (/failed: \d/.test(message)) {
    return "That server rejected the connection. Check the address.";
  }
  return `Couldn't reach ${url}. Is it running mold serve?`;
}

async function connectTo(
  url: string,
  displayName: string,
  advertisedInstanceId?: string | null,
) {
  if (probing.value || !props.open || step.value !== 2) return;
  const sequence = openSequence;
  const key = apiKey.value.trim();
  const probe: HostEntry = {
    id: hostIdFromUrl(url),
    name: displayName.trim() || url,
    url,
  };
  if (key) probe.apiKey = key;

  probing.value = true;
  error.value = null;
  try {
    const status = await hostStatus(probe);
    if (sequence !== openSequence || !props.open || step.value !== 2) return;
    const instanceId = status.instance_id || advertisedInstanceId;
    // Desktop parity: with no typed display name, the server's own hostname
    // labels the machine — never the raw URL, which reads as debris in every
    // card, chip, and activity row.
    const resolvedName =
      displayName.trim() || status.hostname?.trim() || probe.name;
    const entry = addHost({
      url,
      name: resolvedName,
      ...(key ? { apiKey: key } : {}),
      ...(instanceId ? { instanceId } : {}),
    });
    connected.value = entry;
    stopDiscoveryRefresh();
    step.value = 3;
  } catch (e) {
    if (sequence !== openSequence || !props.open) return;
    const message = e instanceof Error ? e.message : String(e);
    error.value = describeError(message, url);
  } finally {
    if (sequence === openSequence) probing.value = false;
  }
}

async function connect() {
  if (probing.value) return;
  if (mode.value === "lan" && selectedPeer.value) {
    await connectTo(
      selectedPeer.value.url,
      selectedPeer.value.name,
      selectedPeer.value.instance_id,
    );
    return;
  }
  const url = normalizeHostAddress(address.value);
  if (!url) {
    error.value = "Enter an address like 192.168.1.20:7680.";
    return;
  }
  await connectTo(url, name.value);
}

async function pickDiscovered(peer: DiscoveryPeer) {
  if (probing.value) return;
  apiKey.value = "";
  error.value = null;
  if (peer.auth_required) {
    selectedPeer.value = peer;
    return;
  }
  await connectTo(peer.url, peer.name, peer.instance_id);
}

function done() {
  if (connected.value) emit("added", connected.value);
  emit("close");
}

onBeforeUnmount(() => {
  ++capabilitySequence;
  invalidateStep();
});
</script>

<template>
  <ModalPanel
    ref="host"
    class="connect-modal"
    :open="open"
    :width="480"
    :steps="3"
    :step="step"
    label="Add a machine"
    @close="close"
    @keydown="onKeydown"
  >
    <!-- Step 1 — type -->
    <template v-if="step === 1">
      <div class="cm__title">Add a machine</div>
      <p class="cm__sub">
        Generate on another GPU. Everything still runs on hardware you control.
      </p>
      <div class="cm__types">
        <button
          type="button"
          class="cm__type"
          data-on="true"
          data-test="type-remote"
          @click="toStep2"
        >
          <span class="cm__type-icon"><Icon name="machines" :size="18" /></span>
          <span class="cm__type-body">
            <span class="cm__type-name">Remote server</span>
            <span class="cm__type-desc">
              A machine running <span class="cm__mono">mold serve</span>
            </span>
          </span>
        </button>
        <button
          type="button"
          class="cm__type"
          :disabled="checkingDiscovery || !discoveryAvailable"
          data-test="type-lan"
          @click="browseLan"
        >
          <span class="cm__type-icon"><Icon name="wifi" :size="18" /></span>
          <span class="cm__type-body">
            <span class="cm__type-name">Local network</span>
            <span v-if="checkingDiscovery" class="cm__type-desc">
              Checking this server for network discovery…
            </span>
            <span v-else-if="discoveryAvailable" class="cm__type-desc">
              Find mold machines visible from this server
            </span>
            <span v-else class="cm__type-desc">
              This server does not advertise network discovery. Enter an address
              to connect.
            </span>
          </span>
        </button>
      </div>
    </template>

    <!-- Step 2 — manual details or server-assisted LAN results -->
    <template v-else-if="step === 2">
      <template v-if="mode === 'remote'">
        <div class="cm__title">Connection details</div>
        <p class="cm__sub">Point at a machine running mold serve.</p>

        <label class="cm__label" for="cm-address">Address</label>
        <div class="cm__field">
          <Icon name="lock" :size="14" />
          <input
            :disabled="probing"
            id="cm-address"
            v-model="address"
            class="cm__input"
            placeholder="192.168.1.20:7680"
            autocomplete="off"
            spellcheck="false"
            data-test="connect-address"
            @keydown.enter="connect"
          />
        </div>

        <label class="cm__label" for="cm-name">Display name</label>
        <div class="cm__field">
          <input
            :disabled="probing"
            id="cm-name"
            v-model="name"
            class="cm__input"
            placeholder="Studio Tower"
            autocomplete="off"
            data-test="connect-name"
            @keydown.enter="connect"
          />
        </div>

        <label class="cm__label" for="cm-key"
          >API key <span class="cm__opt">(optional)</span></label
        >
        <div class="cm__field">
          <input
            :disabled="probing"
            id="cm-key"
            v-model="apiKey"
            type="password"
            class="cm__input"
            placeholder="x-api-key"
            autocomplete="off"
            data-test="connect-key"
            @keydown.enter="connect"
          />
        </div>
      </template>

      <template v-else>
        <div class="cm__title">Local network</div>
        <p class="cm__sub">
          Machines visible on this server's local network. Your browser connects
          to the selected address directly.
        </p>

        <div
          v-if="scanning"
          class="cm__discovery-state"
          data-test="discovery-scanning"
        >
          Looking for mold machines…
        </div>
        <div
          v-else-if="
            !discoveryError && visiblePeers.length === 0 && !selectedPeer
          "
          class="cm__discovery-state"
          data-test="discovery-empty"
        >
          No other mold machines found. Check that they are running and mDNS is
          enabled.
        </div>
        <div v-else-if="!selectedPeer" class="cm__peers">
          <div
            v-for="peer in visiblePeers"
            :key="peer.instance_id || peer.url"
            class="cm__peer"
            data-test="discovery-peer"
          >
            <span class="cm__peer-body">
              <strong>{{ peer.name }}</strong>
              <span class="cm__mono">{{ peer.host }}:{{ peer.port }}</span>
              <small>{{
                peer.version ? `mold ${peer.version}` : "mold"
              }}</small>
            </span>
            <button
              type="button"
              class="cm__peer-connect"
              :disabled="probing"
              data-test="discovery-peer-connect"
              @click="pickDiscovered(peer)"
            >
              {{ probing ? "Testing…" : "Connect" }}
            </button>
          </div>
        </div>

        <template v-if="selectedPeer">
          <div class="cm__peer cm__peer--selected">
            <span class="cm__peer-body">
              <strong>{{ selectedPeer.name }}</strong>
              <span class="cm__mono"
                >{{ selectedPeer.host }}:{{ selectedPeer.port }}</span
              >
            </span>
          </div>
          <label class="cm__label" for="cm-discovery-key">API key</label>
          <div class="cm__field">
            <input
              :disabled="probing"
              id="cm-discovery-key"
              v-model="apiKey"
              type="password"
              class="cm__input"
              placeholder="x-api-key"
              autocomplete="off"
              data-test="discovery-key"
              @keydown.enter="connect"
            />
          </div>
        </template>
      </template>

      <div
        v-if="mode === 'lan' && discoveryError"
        class="cm__error"
        role="alert"
        data-test="discovery-error"
      >
        <p>{{ discoveryError }}</p>
        <button
          type="button"
          class="cm__btn cm__btn--ghost"
          :disabled="scanning"
          data-test="discovery-retry"
          @click="refreshDiscovery(true)"
        >
          Retry discovery
        </button>
      </div>
      <p v-if="error" class="cm__error" role="alert" data-test="connect-error">
        {{ error }}
      </p>
      <div v-else-if="mode === 'remote'" class="cm__note">
        <span class="cm__note-dot" />
        Your browser connects directly to this address.
      </div>
    </template>

    <!-- Step 3 — connected -->
    <template v-else>
      <div class="cm__confirm" data-test="connect-confirm">
        <span class="cm__confirm-icon"><Icon name="check" :size="30" /></span>
        <div class="cm__title">Machine connected</div>
        <p class="cm__sub cm__sub--center">
          {{ connected?.name }} is online and ready. Pick it as a generation
          target from Machines.
        </p>
      </div>
    </template>

    <template #footer>
      <button
        v-if="step === 1"
        type="button"
        class="cm__btn cm__btn--ghost"
        @click="close"
      >
        Cancel
      </button>
      <button
        v-if="step === 2"
        type="button"
        class="cm__btn cm__btn--ghost"
        data-test="connect-back"
        @click="back"
      >
        Back
      </button>
      <div class="cm__spacer" />
      <button
        v-if="step === 1"
        type="button"
        class="cm__btn cm__btn--primary"
        data-test="connect-continue"
        @click="toStep2"
      >
        Continue
      </button>
      <button
        v-else-if="step === 2 && (mode === 'remote' || selectedPeer)"
        type="button"
        class="cm__btn cm__btn--primary"
        :disabled="probing"
        data-test="connect-submit"
        @click="connect"
      >
        {{ probing ? "Connecting…" : "Connect" }}
      </button>
      <button
        v-else-if="step === 3"
        type="button"
        class="cm__btn cm__btn--primary"
        data-test="connect-done"
        @click="done"
      >
        Done
      </button>
    </template>
  </ModalPanel>
</template>

<style scoped>
.connect-modal {
  position: fixed;
  padding: 8px;
}
.connect-modal :deep(.ms-modal__panel) {
  max-width: 100%;
  max-height: calc(100svh - 16px);
  overflow-y: auto;
}
.connect-modal :deep(.ms-modal__footer) {
  flex-wrap: wrap;
}

.cm__title {
  font-family: var(--f-display);
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}

.cm__sub {
  font-size: 0.875rem;
  color: var(--ink-3);
  margin: 5px 0 18px;
  line-height: 1.5;
}

.cm__sub--center {
  text-align: center;
  margin: 6px auto 0;
  max-width: 300px;
}

.cm__types {
  display: flex;
  flex-direction: column;
  gap: 9px;
}

.cm__type {
  display: flex;
  align-items: center;
  gap: 13px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 14px;
  border-radius: 12px;
  text-align: left;
  cursor: pointer;
}

.cm__type[data-on="true"] {
  border-color: var(--sel-border);
  background: var(--sel-bg);
  box-shadow: var(--sel-ring);
}

.cm__type:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.cm__type-icon {
  width: 36px;
  height: 36px;
  flex: 0 0 36px;
  border-radius: 9px;
  background: color-mix(in srgb, var(--halide) 16%, transparent);
  color: var(--halide);
  display: flex;
  align-items: center;
  justify-content: center;
}

.cm__type-body {
  min-width: 0;
  overflow-wrap: anywhere;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.cm__type-name {
  font-size: 0.875rem;
  font-weight: 600;
}

.cm__type-desc {
  font-size: 0.875rem;
  color: var(--ink-3);
  line-height: 1.4;
}

.cm__mono {
  font-family: var(--f-mono);
}

.cm__label {
  display: block;
  font-size: 0.875rem;
  color: var(--ink-2);
  font-weight: 600;
  margin: 16px 0 7px;
}

.cm__label:first-of-type {
  margin-top: 0;
}

.cm__opt {
  font-weight: 400;
  color: var(--ink-3);
}

.cm__field {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: 10px;
  padding: 11px 13px;
  color: var(--ink-3);
}

.cm__field:focus-within {
  border-color: var(--safelight);
}

.cm__input {
  min-height: 44px;
  flex: 1;
  min-width: 0;
  background: transparent;
  border: 0;
  outline: none;
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 1rem;
}

.cm__input::placeholder {
  color: var(--ink-3);
}

.cm__note {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.875rem;
  color: var(--ink-3);
}

.cm__note-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--safelight);
}

.cm__discovery-state {
  min-height: 96px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  border: 1px dashed var(--ce);
  border-radius: 12px;
  color: var(--ink-3);
  font-size: 0.875rem;
  line-height: 1.5;
  text-align: center;
}

.cm__peers {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.cm__peer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border: 1px solid var(--ce);
  border-radius: 11px;
  background: var(--bath);
}

.cm__peer--selected {
  border-color: var(--sel-border);
  background: var(--sel-bg);
}

.cm__peer-body {
  overflow-wrap: anywhere;
  flex-basis: 10rem;
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  color: var(--ink-2);
  font-size: 0.875rem;
}

.cm__peer-body strong {
  color: var(--rebate);
  font-size: 1rem;
}

.cm__peer-body small {
  color: var(--ink-3);
}

.cm__peer-connect {
  min-height: 44px;
  border: 1px solid var(--ce);
  border-radius: 8px;
  background: transparent;
  color: var(--safelight);
  padding: 8px 11px;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
}

.cm__peer-connect:disabled {
  opacity: 0.65;
  cursor: progress;
}

.cm__error {
  overflow-wrap: anywhere;
  margin-top: 16px;
  font-size: 0.875rem;
  color: var(--stop);
  line-height: 1.45;
}

.cm__confirm {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 12px 0 6px;
}

.cm__confirm-icon {
  width: 64px;
  height: 64px;
  margin-bottom: 16px;
  border-radius: 50%;
  background: color-mix(in srgb, var(--safelight) 18%, transparent);
  color: var(--safelight);
  display: flex;
  align-items: center;
  justify-content: center;
}

.cm__btn {
  min-height: 44px;
  border-radius: 10px;
  padding: 11px 16px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
}

.cm__btn--ghost {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
}

.cm__btn--primary {
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 11px 24px;
  font-weight: 700;
}

.cm__btn--primary:disabled {
  opacity: 0.7;
  cursor: progress;
}

.cm__spacer {
  flex: 1;
}
</style>
