<script setup lang="ts">
/*
 * Connect-a-machine wizard (spec §08 G1, prototype CONNECT MODAL). Three
 * stepped panes: pick a type, enter connection details, confirm. Only the
 * "Remote server" type is selectable in the browser — LAN auto-discovery
 * needs the desktop app, so that card is dimmed with an explanatory caption.
 *
 * Step 2 probes `/api/status` with the entered key; a failure keeps the typed
 * values and shows a blunt inline error (G4). On success the host is deduped
 * by its instance id and written to the registry.
 */
import { ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import Icon from "@ui/components/Icon.vue";
import { hostStatus } from "./hostClient";
import {
  addHost,
  hostIdFromUrl,
  normalizeHostAddress,
  type HostEntry,
} from "../../lib/hostRegistry";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: []; added: [host: HostEntry] }>();

const step = ref<1 | 2 | 3>(1);
const address = ref("");
const name = ref("");
const apiKey = ref("");
const probing = ref(false);
const error = ref<string | null>(null);
const connected = ref<HostEntry | null>(null);

function reset() {
  step.value = 1;
  address.value = "";
  name.value = "";
  apiKey.value = "";
  probing.value = false;
  error.value = null;
  connected.value = null;
}

watch(
  () => props.open,
  (open) => {
    if (open) reset();
  },
);

function close() {
  emit("close");
}

function toStep2() {
  error.value = null;
  step.value = 2;
}

function back() {
  error.value = null;
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

async function connect() {
  const url = normalizeHostAddress(address.value);
  if (!url) {
    error.value = "Enter an address like 192.168.1.20:7680.";
    return;
  }
  const probe: HostEntry = {
    id: hostIdFromUrl(url),
    name: name.value.trim() || url,
    url,
  };
  if (apiKey.value.trim()) probe.apiKey = apiKey.value.trim();

  probing.value = true;
  error.value = null;
  try {
    const status = await hostStatus(probe);
    const entry = addHost({
      url,
      name: name.value,
      ...(apiKey.value.trim() ? { apiKey: apiKey.value.trim() } : {}),
      ...(status.instance_id ? { instanceId: status.instance_id } : {}),
    });
    connected.value = entry;
    step.value = 3;
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    error.value = describeError(message, url);
  } finally {
    probing.value = false;
  }
}

function done() {
  if (connected.value) emit("added", connected.value);
  emit("close");
}
</script>

<template>
  <ModalPanel
    :open="open"
    :width="480"
    :steps="3"
    :step="step"
    label="Add a machine"
    @close="close"
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
        >
          <span class="cm__type-icon"><Icon name="machines" :size="18" /></span>
          <span class="cm__type-body">
            <span class="cm__type-name">Remote server</span>
            <span class="cm__type-desc">
              A machine running <span class="cm__mono">mold serve</span>
            </span>
          </span>
        </button>
        <button type="button" class="cm__type" disabled data-test="type-lan">
          <span class="cm__type-icon"><Icon name="wifi" :size="18" /></span>
          <span class="cm__type-body">
            <span class="cm__type-name">Local network</span>
            <span class="cm__type-desc">
              Browsers can't discover LAN hosts — use the desktop app or enter
              an address.
            </span>
          </span>
        </button>
      </div>
    </template>

    <!-- Step 2 — details -->
    <template v-else-if="step === 2">
      <div class="cm__title">Connection details</div>
      <p class="cm__sub">Point at a machine running mold serve.</p>

      <label class="cm__label" for="cm-address">Address</label>
      <div class="cm__field">
        <Icon name="lock" :size="14" />
        <input
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

      <p v-if="error" class="cm__error" data-test="connect-error">
        {{ error }}
      </p>
      <div v-else class="cm__note">
        <span class="cm__note-dot" />
        Connection is direct — no data leaves your network.
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
        v-else-if="step === 2"
        type="button"
        class="cm__btn cm__btn--primary"
        :disabled="probing"
        data-test="connect-submit"
        @click="connect"
      >
        {{ probing ? "Connecting…" : "Connect" }}
      </button>
      <button
        v-else
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
.cm__title {
  font-family: var(--f-display);
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}

.cm__sub {
  font-size: 12.5px;
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
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.cm__type-name {
  font-size: 13.5px;
  font-weight: 600;
}

.cm__type-desc {
  font-size: 11.5px;
  color: var(--ink-3);
  line-height: 1.4;
}

.cm__mono {
  font-family: var(--f-mono);
}

.cm__label {
  display: block;
  font-size: 12px;
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
  flex: 1;
  min-width: 0;
  background: transparent;
  border: 0;
  outline: none;
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 13px;
}

.cm__input::placeholder {
  color: var(--ink-3);
}

.cm__note {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11.5px;
  color: var(--ink-3);
}

.cm__note-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--safelight);
}

.cm__error {
  margin-top: 16px;
  font-size: 12.5px;
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
  border-radius: 10px;
  padding: 11px 16px;
  font-size: 13px;
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
