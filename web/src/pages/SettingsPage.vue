<script setup lang="ts">
/*
 * Settings workspace (spec §04/§06, prototype desktop Settings). Appearance
 * (theme family + dark/light on the shared lib/theme refs), account tokens
 * (stored by the connected server and returned only as masked status), and an
 * About card sourced from GET /api/status. Set-once prefs only; per-generation
 * knobs stay in Create's Advanced.
 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import CardSurface from "@ui/components/CardSurface.vue";
import PairingAccessPanel from "@studio/components/PairingAccessPanel.vue";
import DevicePanel from "@studio/components/DevicePanel.vue";
import LicenseSettingsPanel from "@studio/components/LicenseSettingsPanel.vue";
import type { DeviceInfo } from "@studio/api/devices";
import { setQueueDevicePin, type QueuePlan } from "@studio/api/queuePlan";
import ConfigSettingsPanel from "../components/ConfigSettingsPanel.vue";
import SegmentedControl, {
  type SegmentOption,
} from "@ui/components/SegmentedControl.vue";
import { theme, themeFamily, type Theme, type ThemeFamily } from "../lib/theme";
import { toast } from "../lib/toasts";
import { useStatusPoll } from "../composables/useStatusPoll";
import {
  canMutateDevice,
  deviceActionLabel,
  deviceLifecycleMessage,
  deviceStateLabel,
} from "@studio/lib/deviceLifecycle";
import type { ServerCapabilities } from "../types";
import { originHost } from "../lib/hostRegistry";
import { subscribeToDeviceSnapshots } from "../lib/deviceEvents";
import {
  hostCapabilities,
  hostDevices,
  hostQueue,
  setHostDeviceEnabled,
} from "../components/machines/hostClient";
import {
  deleteCatalogCredential,
  getCatalogCredentialStatus,
  putCatalogCredential,
  type CatalogCredentialStatus,
} from "../api";

const familyOptions: SegmentOption<ThemeFamily>[] = [
  { value: "mold", label: "Mold" },
  { value: "safelight", label: "Safelight" },
];
const appearanceOptions: SegmentOption<Theme>[] = [
  { value: "system", label: "System" },
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
];
const pairingHost = computed(() => originHost());
const pairingTarget = computed(() => ({
  baseUrl: pairingHost.value.url,
  apiKey: pairingHost.value.apiKey ?? null,
}));

function setFamily(value: ThemeFamily) {
  themeFamily.value = value;
}
function setAppearance(value: Theme) {
  theme.value = value;
}

// ── Account tokens (owned by the connected server) ─────────────────────
const emptyCredentialStatus = (): CatalogCredentialStatus => ({
  hf: { configured: false, source: null, masked: null },
  civitai: { configured: false, source: null, masked: null },
});
const saved = ref<CatalogCredentialStatus>(emptyCredentialStatus());
const hfMasked = computed(() => saved.value.hf.masked);
const civitaiMasked = computed(() => saved.value.civitai.masked);
const hfHasSavedOverride = computed(() => saved.value.hf.source === "server");
const civitaiHasSavedOverride = computed(
  () => saved.value.civitai.source === "server",
);
const credentialSourceLabel = (source: "environment" | "server" | null) =>
  source === "server"
    ? "Saved override"
    : source === "environment"
      ? "Environment"
      : "";

// Draft inputs, only shown while entering/replacing a token.
const hfDraft = ref("");
const civitaiDraft = ref("");
const editingHf = ref(false);
const editingCivitai = ref(false);

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function loadCredentials() {
  try {
    saved.value = await getCatalogCredentialStatus();
  } catch (error) {
    toast("error", `Could not load server credentials: ${errorMessage(error)}`);
  }
}

onMounted(() => {
  void loadCredentials();
});

async function saveHf() {
  const value = hfDraft.value.trim();
  if (!value) return;
  try {
    saved.value = await putCatalogCredential("hf", value);
    hfDraft.value = "";
    editingHf.value = false;
    toast("success", "Hugging Face token saved on this server");
  } catch (error) {
    toast("error", `Hugging Face token was not saved: ${errorMessage(error)}`);
  }
}
async function removeHf() {
  try {
    saved.value = await deleteCatalogCredential("hf");
    hfDraft.value = "";
    editingHf.value = false;
  } catch (error) {
    toast(
      "error",
      `Hugging Face token was not removed: ${errorMessage(error)}`,
    );
  }
}
async function saveCivitai() {
  const value = civitaiDraft.value.trim();
  if (!value) return;
  try {
    saved.value = await putCatalogCredential("civitai", value);
    civitaiDraft.value = "";
    editingCivitai.value = false;
    toast("success", "Civitai token saved on this server");
  } catch (error) {
    toast("error", `Civitai token was not saved: ${errorMessage(error)}`);
  }
}
async function removeCivitai() {
  try {
    saved.value = await deleteCatalogCredential("civitai");
    civitaiDraft.value = "";
    editingCivitai.value = false;
  } catch (error) {
    toast("error", `Civitai token was not removed: ${errorMessage(error)}`);
  }
}

// About — server version + processing summary.
const { status } = useStatusPoll();
const version = computed(() => status.value?.version ?? "—");
const devices = ref<DeviceInfo[] | null>(null);
const deviceCapabilities = ref<ServerCapabilities | null>(null);
const deviceMutations = ref(new Set<string>());
const queuePlan = ref<QueuePlan | null>(null);
let devicePanelRequestGeneration = 0;

async function loadDevicePanel() {
  const host = originHost();
  const generation = ++devicePanelRequestGeneration;
  const [deviceResult, queueResult, capabilityResult] =
    await Promise.allSettled([
      hostDevices(host),
      hostQueue(host),
      hostCapabilities(host),
    ]);
  if (generation !== devicePanelRequestGeneration) return;
  if (deviceResult.status === "fulfilled")
    devices.value = deviceResult.value.devices;
  if (queueResult.status === "fulfilled")
    queuePlan.value = queueResult.value.plan ?? null;
  else queuePlan.value = null;
  deviceCapabilities.value =
    capabilityResult.status === "fulfilled" ? capabilityResult.value : null;
  if (deviceResult.status !== "fulfilled") devices.value = null;
}

async function toggleDeviceById(deviceId: string, enabled: boolean) {
  const device = devices.value?.find((candidate) => candidate.id === deviceId);
  if (!device || enabled === device.desired_enabled) return;
  if (!canMutateDevice(device, deviceCapabilities.value)) return;
  deviceMutations.value = new Set(deviceMutations.value).add(device.id);
  try {
    await setHostDeviceEnabled(originHost(), device.id, enabled);
    await loadDevicePanel();
  } catch (error) {
    toast("error", `Could not update ${device.name}: ${errorMessage(error)}`);
  } finally {
    const next = new Set(deviceMutations.value);
    next.delete(device.id);
    deviceMutations.value = next;
  }
}

function toggleDevice(device: DeviceInfo) {
  return toggleDeviceById(device.id, !device.desired_enabled);
}

async function unpinWork(workId: string) {
  const host = originHost();
  try {
    await setQueueDevicePin(
      { baseUrl: host.url, apiKey: host.apiKey ?? null },
      workId,
      null,
    );
    await loadDevicePanel();
  } catch (error) {
    toast("error", `Queue pin was not changed: ${errorMessage(error)}`);
  }
}

let deviceEventsAbort: AbortController | null = null;

onMounted(() => {
  const host = originHost();
  // Bootstrap even when the server predates `/api/events`; the subscription
  // is an invalidation accelerator, not the source of initial truth.
  void loadDevicePanel();
  deviceEventsAbort = new AbortController();
  subscribeToDeviceSnapshots(
    { baseUrl: host.url, apiKey: host.apiKey ?? null },
    deviceEventsAbort.signal,
    () => void loadDevicePanel(),
  );
});

onBeforeUnmount(() => {
  devicePanelRequestGeneration += 1;
  deviceEventsAbort?.abort();
});
</script>

<template>
  <div class="settings">
    <h1 class="settings__title">Settings</h1>

    <p class="kicker">Appearance</p>
    <CardSurface class="settings__card">
      <div class="row">
        <span class="row__label">Theme</span>
        <SegmentedControl
          data-test="seg-theme"
          :model-value="themeFamily"
          :options="familyOptions"
          label="Theme family"
          @update:model-value="setFamily"
        />
      </div>
      <div class="row">
        <span class="row__label">Appearance</span>
        <SegmentedControl
          data-test="seg-appearance"
          :model-value="theme"
          :options="appearanceOptions"
          label="Appearance"
          @update:model-value="setAppearance"
        />
      </div>
    </CardSurface>

    <p class="kicker">Mobile pairing</p>
    <PairingAccessPanel
      :target="pairingTarget"
      :suggested-base-url="pairingHost.url"
      host-label="This server"
    />

    <p v-if="devices !== null || queuePlan !== null" class="kicker">
      Scheduler plan
    </p>
    <CardSurface
      v-if="devices !== null || queuePlan !== null"
      class="settings__card"
    >
      <DevicePanel
        :devices="devices ?? []"
        :plan="queuePlan"
        :mutable="
          deviceCapabilities?.devices?.lifecycle === true &&
          deviceCapabilities?.dispatch?.v2_authoritative === true
        "
        :busy-device-ids="[...deviceMutations]"
        @unpin="unpinWork"
        @toggle="toggleDeviceById"
      />
    </CardSurface>

    <p class="kicker">Accounts</p>
    <CardSurface class="settings__card">
      <!-- Hugging Face -->
      <div class="row">
        <label class="row__label" for="hf_token">Hugging Face token</label>
        <div v-if="hfMasked && !editingHf" class="field" data-test="hf-saved">
          <code class="token-mask" data-test="hf-mask">{{ hfMasked }}</code>
          <button
            type="button"
            class="btn"
            data-test="replace-hf"
            @click="editingHf = true"
          >
            {{ hfHasSavedOverride ? "Replace" : "Override" }}
          </button>
          <button
            v-if="hfHasSavedOverride"
            type="button"
            class="btn btn--ghost"
            data-test="clear-hf"
            @click="removeHf"
          >
            Clear
          </button>
          <span class="env-badge" data-test="hf-source">{{
            credentialSourceLabel(saved.hf.source)
          }}</span>
        </div>
        <div v-else class="field">
          <input
            id="hf_token"
            v-model="hfDraft"
            name="hf_token"
            type="password"
            placeholder="hf_…"
            autocomplete="off"
            class="input"
          />
          <button
            type="button"
            class="btn"
            data-test="save-hf"
            :disabled="!hfDraft.trim()"
            @click="saveHf"
          >
            Save
          </button>
          <button
            v-if="editingHf"
            type="button"
            class="btn btn--ghost"
            @click="editingHf = false"
          >
            Cancel
          </button>
        </div>
      </div>

      <!-- Civitai -->
      <div class="row">
        <label class="row__label" for="civitai_token">Civitai token</label>
        <div
          v-if="civitaiMasked && !editingCivitai"
          class="field"
          data-test="civitai-saved"
        >
          <code class="token-mask" data-test="civitai-mask">{{
            civitaiMasked
          }}</code>
          <button
            type="button"
            class="btn"
            data-test="replace-civitai"
            @click="editingCivitai = true"
          >
            {{ civitaiHasSavedOverride ? "Replace" : "Override" }}
          </button>
          <button
            v-if="civitaiHasSavedOverride"
            type="button"
            class="btn btn--ghost"
            data-test="clear-civitai"
            @click="removeCivitai"
          >
            Clear
          </button>
          <span class="env-badge" data-test="civitai-source">{{
            credentialSourceLabel(saved.civitai.source)
          }}</span>
        </div>
        <div v-else class="field">
          <input
            id="civitai_token"
            v-model="civitaiDraft"
            name="civitai_token"
            type="password"
            placeholder="cv_…"
            autocomplete="off"
            class="input"
          />
          <button
            type="button"
            class="btn"
            data-test="save-civitai"
            :disabled="!civitaiDraft.trim()"
            @click="saveCivitai"
          >
            Save
          </button>
          <button
            v-if="editingCivitai"
            type="button"
            class="btn btn--ghost"
            @click="editingCivitai = false"
          >
            Cancel
          </button>
        </div>
      </div>

      <p class="settings__note">
        Tokens are stored privately on this server and used only for catalog
        discovery and downloads. Saved tokens override environment defaults;
        clear an override to use the environment value again.
      </p>
    </CardSurface>

    <ConfigSettingsPanel />

    <template v-if="devices !== null">
      <p class="kicker">Advanced · GPU devices</p>
      <CardSurface class="settings__card" data-test="settings-device-controls">
        <p class="device-state" data-test="settings-device-lifecycle-note">
          {{ deviceLifecycleMessage(deviceCapabilities) }}
        </p>
        <div
          v-for="device in devices"
          :key="device.id"
          class="row"
          data-test="device-row"
        >
          <span class="row__label">
            {{ device.name }}
            <small class="device-state">
              {{
                device.ordinal == null
                  ? device.backend
                  : `GPU ${device.ordinal}`
              }}
              · {{ deviceStateLabel(device) }}
            </small>
          </span>
          <button
            type="button"
            class="btn"
            :data-test="`settings-device-toggle-${device.ordinal ?? device.id}`"
            :disabled="
              !canMutateDevice(device, deviceCapabilities) ||
              deviceMutations.has(device.id)
            "
            @click="toggleDevice(device)"
          >
            {{ deviceActionLabel(device, deviceCapabilities) }}
          </button>
        </div>
      </CardSurface>
    </template>

    <p class="kicker">Model licenses</p>
    <CardSurface class="settings__card">
      <LicenseSettingsPanel :target="pairingTarget" host-label="This machine" />
    </CardSurface>

    <p class="kicker">About</p>
    <CardSurface class="settings__card settings__card--list" :padded="false">
      <div class="about-row">
        <span class="about-row__key">Version</span>
        <span class="about-row__val" data-test="about-version">{{
          version
        }}</span>
      </div>
      <div class="about-row">
        <span class="about-row__key">Processing</span>
        <span class="about-row__val about-row__val--muted">
          local + your hosts
        </span>
      </div>
      <div class="about-row about-row--last">
        <span class="about-row__key">Core contributors</span>
        <span class="about-row__val">James Brink · Jeffrey Dilley</span>
      </div>
    </CardSurface>
  </div>
</template>

<style scoped>
.settings {
  width: 100%;
  max-width: 900px;
  box-sizing: border-box;
  margin: 0 auto;
  padding: 24px 20px 96px;
}

.settings__title {
  margin: 0 0 22px;
  font-family: var(--f-display);
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}

.kicker {
  margin: 0 0 12px;
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.settings__card {
  margin-bottom: 24px;
}

.row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.row + .row {
  margin-top: 14px;
}

.row__label {
  font-size: 13.5px;
  color: var(--rebate);
}

.device-state {
  display: block;
  margin-top: 3px;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  text-transform: uppercase;
}

.field {
  display: flex;
  align-items: center;
  gap: 8px;
}

.env-badge {
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.input {
  width: 240px;
  max-width: 52vw;
  height: 34px;
  padding: 0 11px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 12px;
  outline: none;
}

.input:focus-visible,
.select:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 1px;
}

.select {
  height: 34px;
  padding: 0 11px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 12px;
  outline: none;
}

.btn {
  height: 34px;
  padding: 0 14px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}

.btn:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
}

.btn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn--ghost {
  color: var(--ink-3);
}
.btn--ghost:hover {
  color: var(--stop);
  background: transparent;
}

.token-mask {
  flex: 1;
  min-width: 0;
  height: 34px;
  display: inline-flex;
  align-items: center;
  padding: 0 12px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--ink-2);
  font-family: var(--f-mono);
  font-size: 12.5px;
  letter-spacing: 0.04em;
}

.settings__note {
  margin: 12px 0 0;
  font-size: 11.5px;
  color: var(--ink-3);
  line-height: 1.5;
}

/* About list — rows inside an unpadded card. */
.about-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 11px 16px;
  border-bottom: 1px solid var(--edge);
  font-size: 13px;
}

.about-row--last {
  border-bottom: 0;
}

.about-row__key {
  color: var(--ink-2);
}

.about-row__val {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--rebate);
}

.about-row__val--muted {
  color: var(--ink-3);
}
</style>
