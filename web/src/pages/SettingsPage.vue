<script setup lang="ts">
/*
 * Settings, on the shared studio kit.
 *
 * One searchable page behind a jump nav: `sectionsForSurface("web")` names the
 * sections, `studio/lib/settingsSchema.ts` names every curated engine key, and
 * `studio/components/settings/` draws every row — so web and desktop say the
 * same words about the same machine, and a key this build has never heard of
 * is the ONLY thing that reaches Advanced.
 *
 * The page owns what is web's alone: the browser's theme refs, the server's
 * catalog credentials, this origin's devices, and the `?section=` deep link.
 */
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { RouterLink, useRoute, useRouter } from "vue-router";
import PairingAccessPanel from "@studio/components/PairingAccessPanel.vue";
import DevicePanel from "@studio/components/DevicePanel.vue";
import LicenseSettingsPanel from "@studio/components/LicenseSettingsPanel.vue";
import SettingsShell from "@studio/components/settings/SettingsShell.vue";
import SettingRow from "@studio/components/settings/SettingRow.vue";
import ConfigSettingRow from "@studio/components/settings/ConfigSettingRow.vue";
import ConfigRowItem from "@studio/components/settings/ConfigRowItem.vue";
import PerStyleDefaultsRow from "@studio/components/settings/PerStyleDefaultsRow.vue";
import ThemePicker from "@studio/components/settings/ThemePicker.vue";
import ToggleControl from "@studio/components/settings/ToggleControl.vue";
import {
  groupPerStyleRows,
  schemaFor,
  schemasForSection,
  sectionForConfigKey,
  sectionsForSurface,
  type SectionId,
} from "@studio/lib/settingsSchema";
import {
  listConfig,
  listProfiles,
  resetConfig,
  setConfig,
  switchProfile,
  type ConfigRow,
  type ConfigValue,
} from "@studio/api/config";
import { styleDisplayName, styleLabel } from "@studio/lib/styleLabel";
import type { DeviceInfo } from "@studio/api/devices";
import { matchSystem, theme } from "../lib/theme";
import type { ThemeId } from "@ui/theme";
import { toast } from "../lib/toasts";
import { useStatusPoll } from "../composables/useStatusPoll";
import { canMutateDevice } from "@studio/lib/deviceLifecycle";
import type { ModelInfoExtended, ServerCapabilities } from "../types";
import {
  HOSTS_CHANGED_EVENT,
  listKnownHosts,
  originHost,
} from "../lib/hostRegistry";
import { autoTagTitle } from "../lib/fileUnder";
import { subscribeToDeviceSnapshots } from "../lib/deviceEvents";
import {
  hostCapabilities,
  hostDevices,
  setHostDeviceEnabled,
} from "../components/machines/hostClient";
import {
  deleteCatalogCredential,
  fetchModels,
  getCatalogCredentialStatus,
  putCatalogCredential,
  type CatalogCredentialStatus,
} from "../api";

const WEB_SECTIONS = sectionsForSurface("web");

const pairingHost = computed(() => originHost());
const originTarget = computed(() => ({
  baseUrl: pairingHost.value.url,
  apiKey: pairingHost.value.apiKey ?? null,
}));

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// ── The jump nav and the ?section= deep link ───────────────────────────────
const shell = ref<{ jump: (id: SectionId) => void } | null>(null);

/**
 * A browser page scrolls the window, so the shell runs as PANES here: the nav
 * is a navigation and one section is on screen at a time. That makes
 * `?section=` the address of what is showing, in both directions — a deep
 * link opens its pane, and picking a pane rewrites the address (replace, not
 * push: the panes are one page to the Back button).
 */
const route = useRoute();
const router = useRouter();

function sectionFromQuery(section: unknown): SectionId | null {
  if (typeof section !== "string") return null;
  // The retired `about` section folded into Updates & about.
  const id = section === "about" ? "updates" : section;
  return WEB_SECTIONS.some((candidate) => candidate.id === id)
    ? (id as SectionId)
    : null;
}

watch(
  () => route?.query.section,
  async (section) => {
    const id = sectionFromQuery(section);
    if (!id) return;
    await nextTick();
    shell.value?.jump(id);
  },
  { immediate: true },
);

function onActiveSection(id: SectionId) {
  if (!route || !router) return;
  if (route.query.section === id) return;
  void router.replace({ query: { ...route.query, section: id } });
}

// ── Engine configuration (this origin) ────────────────────────────────────
const configRows = ref<ConfigRow[]>([]);
const configError = ref("");
const configLoaded = ref(false);
let configSequence = 0;

async function loadConfig() {
  const sequence = ++configSequence;
  try {
    const rows = await listConfig(originTarget.value);
    if (sequence !== configSequence) return;
    // `tui.*` belongs to the terminal app, not to a graphical surface.
    configRows.value = rows.filter((row) => !row.key.startsWith("tui."));
    configError.value = "";
    configLoaded.value = true;
  } catch (error) {
    if (sequence !== configSequence) return;
    configError.value = errorMessage(error);
  }
}

const rowsByKey = computed(
  () => new Map(configRows.value.map((row) => [row.key, row])),
);
function rowFor(key: string): ConfigRow | null {
  return rowsByKey.value.get(key) ?? null;
}
function labelFor(key: string): string {
  return schemaFor(key)?.label ?? key;
}
function keysOf(section: SectionId): string[] {
  return schemasForSection(section).map((schema) => schema.key);
}

async function saveConfig(key: string, value: ConfigValue) {
  try {
    await setConfig(originTarget.value, key, value);
    await loadConfig();
  } catch (error) {
    toast("error", `${labelFor(key)} was not saved: ${errorMessage(error)}`);
  }
}

async function resetConfigKey(key: string) {
  try {
    await resetConfig(originTarget.value, key);
    await loadConfig();
  } catch (error) {
    toast("error", `${labelFor(key)} was not reset: ${errorMessage(error)}`);
  }
}

/** Keys with no curated schema — a server newer than this client, and nothing
 *  else. `settingsSchema.contract.test.ts` is what makes that true. */
const advancedRows = computed(() =>
  configRows.value.filter(
    (row) => !schemaFor(row.key) && sectionForConfigKey(row.key) === "advanced",
  ),
);
const perStyleRows = computed(() =>
  configRows.value.filter(
    (row) => sectionForConfigKey(row.key) === "styleDefaults",
  ),
);
const perStyleGroups = computed(() => groupPerStyleRows(perStyleRows.value));
const styleFilter = ref("");
/** A type-to-filter field earns its place only past a screenful of styles. */
const showStyleFilter = computed(() => perStyleGroups.value.length > 8);
const visiblePerStyleGroups = computed(() => {
  const query = styleFilter.value.trim().toLowerCase();
  if (!query) return perStyleGroups.value;
  return perStyleGroups.value.filter(
    (group) =>
      group.style.toLowerCase().includes(query) ||
      (styleNames.value.get(group.style) ?? "").toLowerCase().includes(query),
  );
});

/** Raw rows each section renders, so the page search matches them too. */
const rawKeysBySection = computed(() => ({
  advanced: advancedRows.value.map((row) => row.key),
  styleDefaults: perStyleRows.value.map((row) => row.key),
}));

// ── Profiles ──────────────────────────────────────────────────────────────
const profiles = ref<string[]>([]);
const activeProfile = ref("default");
const profileName = ref("");
const profilesError = ref("");
const switchingProfile = ref(false);

async function loadProfiles() {
  try {
    const listing = await listProfiles(originTarget.value);
    profiles.value = listing.profiles;
    activeProfile.value = listing.active;
    profilesError.value = "";
  } catch {
    profilesError.value =
      "Could not refresh profiles. Existing settings remain available.";
  }
}

async function changeProfile(name: string) {
  if (switchingProfile.value || !name) return;
  switchingProfile.value = true;
  try {
    await switchProfile(originTarget.value, name);
    activeProfile.value = name;
    await loadConfig();
    await loadProfiles();
    toast("success", `Profile switched to ${name}`);
  } catch (error) {
    toast("error", `Profile was not switched: ${errorMessage(error)}`);
  } finally {
    switchingProfile.value = false;
  }
}

function selectProfile(name: string) {
  if (name === activeProfile.value) return;
  void changeProfile(name);
}

async function createProfile() {
  const name = profileName.value.trim();
  if (!name) return;
  await changeProfile(name);
  profileName.value = "";
}

// ── Installed styles (the Style-to-start-with options) ────────────────────
const models = ref<ModelInfoExtended[]>([]);

async function loadModels() {
  try {
    const listing = await fetchModels();
    models.value = Array.isArray(listing) ? listing : [];
  } catch {
    // The picker degrades to the stored value alone; nothing here is fatal.
    models.value = [];
  }
}

const styleNames = computed(
  () =>
    new Map(models.value.map((model) => [model.name, styleDisplayName(model)])),
);

const defaultModelOptions = computed(() => {
  const options = models.value
    .filter((model) => model.downloaded !== false)
    .map((model) => ({ value: model.name, label: styleLabel(model) }));
  const current = rowFor("default_model")?.value;
  // A style that is no longer installed is still what this machine starts
  // with; dropping it from the list would silently offer to change it.
  if (
    typeof current === "string" &&
    current !== "" &&
    !options.some((option) => option.value === current)
  )
    options.unshift({ value: current, label: current });
  if (current === "" || current == null)
    options.unshift({ value: "", label: "Not set" });
  return options;
});

// ── Account tokens (owned by the connected server) ─────────────────────
const emptyCredentialStatus = (): CatalogCredentialStatus => ({
  hf: { configured: false, source: null, masked: null },
  civitai: { configured: false, source: null, masked: null },
});
const saved = ref<CatalogCredentialStatus>(emptyCredentialStatus());
const credentialsLoaded = ref(false);
const credentialsLoading = ref(false);
const credentialsError = ref("");
const credentialMutation = ref(false);
const canChangeCredentials = computed(
  () =>
    credentialsLoaded.value &&
    !credentialsLoading.value &&
    !credentialsError.value &&
    !credentialMutation.value,
);
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

async function loadCredentials() {
  if (credentialsLoading.value || credentialMutation.value) return;
  credentialsLoading.value = true;
  credentialsError.value = "";
  try {
    saved.value = await getCatalogCredentialStatus();
    credentialsLoaded.value = true;
  } catch (error) {
    credentialsError.value = `Could not load server credentials: ${errorMessage(error)}`;
  } finally {
    credentialsLoading.value = false;
  }
}

async function saveHf() {
  if (!canChangeCredentials.value) return;
  const value = hfDraft.value.trim();
  if (!value) return;
  credentialMutation.value = true;
  try {
    saved.value = await putCatalogCredential("hf", value);
    hfDraft.value = "";
    editingHf.value = false;
    toast("success", "Hugging Face token saved on this server");
  } catch (error) {
    toast("error", `Hugging Face token was not saved: ${errorMessage(error)}`);
  } finally {
    credentialMutation.value = false;
  }
}
async function removeHf() {
  if (!canChangeCredentials.value) return;
  credentialMutation.value = true;
  try {
    saved.value = await deleteCatalogCredential("hf");
    hfDraft.value = "";
    editingHf.value = false;
  } catch (error) {
    toast(
      "error",
      `Hugging Face token was not removed: ${errorMessage(error)}`,
    );
  } finally {
    credentialMutation.value = false;
  }
}
async function saveCivitai() {
  if (!canChangeCredentials.value) return;
  const value = civitaiDraft.value.trim();
  if (!value) return;
  credentialMutation.value = true;
  try {
    saved.value = await putCatalogCredential("civitai", value);
    civitaiDraft.value = "";
    editingCivitai.value = false;
    toast("success", "Civitai token saved on this server");
  } catch (error) {
    toast("error", `Civitai token was not saved: ${errorMessage(error)}`);
  } finally {
    credentialMutation.value = false;
  }
}
async function removeCivitai() {
  if (!canChangeCredentials.value) return;
  credentialMutation.value = true;
  try {
    saved.value = await deleteCatalogCredential("civitai");
    civitaiDraft.value = "";
    editingCivitai.value = false;
  } catch (error) {
    toast("error", `Civitai token was not removed: ${errorMessage(error)}`);
  } finally {
    credentialMutation.value = false;
  }
}

// ── This server ───────────────────────────────────────────────────────────
const { status } = useStatusPoll();
const version = computed(() => status.value?.version ?? "—");
const gitSha = computed(() => status.value?.git_sha ?? null);
/** The short sha is what a person reads and types; the whole 40 characters
 *  ride the row's tooltip, because at phone width they are wider than the
 *  screen. */
const gitShaShort = computed(() => gitSha.value?.slice(0, 7) ?? "—");
const buildDate = computed(() => status.value?.build_date ?? "—");
const devices = ref<DeviceInfo[] | null>(null);
const deviceCapabilities = ref<ServerCapabilities | null>(null);
const deviceMutations = ref(new Set<string>());
let devicePanelRequestGeneration = 0;

/** What this machine computes on, as the machine itself reports it. */
const backend = computed(() => {
  const list = devices.value;
  if (!list || list.length === 0) return "—";
  return [...new Set(list.map((device) => device.backend))].join(" · ");
});

const serverSummary = computed(() =>
  [`mold ${version.value}`, backend.value].filter((part) => part).join(" · "),
);

/* The registry lives in localStorage and announces its own edits; a machine
 * added from another tab must appear in this picker without a reload. */
const knownHosts = ref(listKnownHosts());
function refreshKnownHosts() {
  knownHosts.value = listKnownHosts();
}
const licenceHosts = computed(() => knownHosts.value);
const licenceHostId = ref(originHost().id);
const licenceTarget = computed(() => {
  const host =
    licenceHosts.value.find((entry) => entry.id === licenceHostId.value) ??
    originHost();
  return { baseUrl: host.url, apiKey: host.apiKey ?? null };
});
const licenceHostLabel = computed(
  () =>
    licenceHosts.value.find((entry) => entry.id === licenceHostId.value)
      ?.name ?? "This machine",
);

async function loadDevicePanel() {
  const host = originHost();
  const generation = ++devicePanelRequestGeneration;
  const [deviceResult, capabilityResult] = await Promise.allSettled([
    hostDevices(host),
    hostCapabilities(host),
  ]);
  if (generation !== devicePanelRequestGeneration) return;
  if (deviceResult.status === "fulfilled")
    devices.value = deviceResult.value.devices;
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

let deviceEventsAbort: AbortController | null = null;

onMounted(() => {
  const host = originHost();
  void loadCredentials();
  void loadConfig();
  void loadProfiles();
  void loadModels();
  // Bootstrap even when the server predates `/api/events`; the subscription
  // is an invalidation accelerator, not the source of initial truth.
  void loadDevicePanel();
  window.addEventListener(HOSTS_CHANGED_EVENT, refreshKnownHosts);
  deviceEventsAbort = new AbortController();
  subscribeToDeviceSnapshots(
    { baseUrl: host.url, apiKey: host.apiKey ?? null },
    deviceEventsAbort.signal,
    () => void loadDevicePanel(),
  );
});

onBeforeUnmount(() => {
  window.removeEventListener(HOSTS_CHANGED_EVENT, refreshKnownHosts);
  devicePanelRequestGeneration += 1;
  configSequence += 1;
  deviceEventsAbort?.abort();
});
</script>

<template>
  <div class="workspace-page settings-page">
    <h1 class="workspace-heading">Settings</h1>

    <p v-if="configError" class="settings-alert" role="alert">
      Could not read this machine's configuration. {{ configError }}
      <button type="button" class="settings-link" @click="loadConfig">
        Retry
      </button>
    </p>

    <SettingsShell
      ref="shell"
      :sections="WEB_SECTIONS"
      :raw-keys-by-section="rawKeysBySection"
      layout="pane"
      @update:active="onActiveSection"
    >
      <template #section="{ section, mounted }">
        <template v-if="!mounted" />

        <!-- Look -->
        <div v-else-if="section.id === 'app'" class="settings-body">
          <ThemePicker
            :theme="theme"
            :match-system="matchSystem"
            @update:theme="(value: ThemeId) => (theme = value)"
            @update:match-system="(value: boolean) => (matchSystem = value)"
          />
        </div>

        <!-- Defaults for new images · Write more for me · Styles & disk ·
             Speed & memory · Cloud GPUs: one curated key per row. -->
        <template v-else-if="section.id === 'generation'">
          <ConfigSettingRow
            v-for="key in keysOf('generation')"
            :key="key"
            :schema-key="key"
            :row="rowFor(key)"
            :options="key === 'default_model' ? defaultModelOptions : undefined"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
        </template>

        <template v-else-if="section.id === 'expansion'">
          <ConfigSettingRow
            v-for="key in keysOf('expansion')"
            :key="key"
            :schema-key="key"
            :row="rowFor(key)"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
        </template>

        <!-- Machines -->
        <template v-else-if="section.id === 'hosts'">
          <SettingRow label="This server" :help="pairingHost.url">
            <span class="settings-value">{{ serverSummary }}</span>
          </SettingRow>
          <div class="settings-panel-row">
            <DevicePanel
              :devices="devices ?? []"
              :plan="null"
              :mutable="
                deviceCapabilities?.devices?.lifecycle === true &&
                deviceCapabilities?.dispatch?.v2_authoritative === true
              "
              :restart-enable="
                deviceCapabilities?.devices?.restart_enable === true
              "
              :capabilities-known="deviceCapabilities !== null"
              :busy-device-ids="[...deviceMutations]"
              show-controls
              @toggle="toggleDeviceById"
            />
          </div>
          <SettingRow
            label="All machines"
            help="Add a machine on your network, or look at what each one is doing."
          >
            <RouterLink class="settings-link" to="/machines"
              >Open Machines</RouterLink
            >
          </SettingRow>
        </template>

        <template v-else-if="section.id === 'styles'">
          <ConfigSettingRow
            v-for="key in keysOf('styles')"
            :key="key"
            :schema-key="key"
            :row="rowFor(key)"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
        </template>

        <!-- Style licences -->
        <div v-else-if="section.id === 'licenses'" class="settings-body">
          <LicenseSettingsPanel
            :target="licenceTarget"
            :host-label="licenceHostLabel"
          >
            <template v-if="licenceHosts.length > 1" #machine>
              <label class="settings-inline-field">
                <span>Machine</span>
                <select
                  v-model="licenceHostId"
                  class="settings-select"
                  data-test="licence-machine"
                  aria-label="Machine"
                >
                  <option
                    v-for="host in licenceHosts"
                    :key="host.id"
                    :value="host.id"
                  >
                    {{ host.name }}
                  </option>
                </select>
              </label>
            </template>
          </LicenseSettingsPanel>
        </div>

        <!-- My images & trash -->
        <template v-else-if="section.id === 'library'">
          <ConfigSettingRow
            schema-key="gallery.trash_retention_days"
            :row="rowFor('gallery.trash_retention_days')"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
          <!-- This browser's own preference: the ghost chip is offered by this
               client, so there is nothing to save on the machine. It sits
               immediately before the engine's own switch, which is the one
               `mold run` reads. -->
          <SettingRow
            label="Tag new prints with their title"
            help="New image offers each titled print its own tag as a removable chip. Off turns the chip off for future prints — it never rewrites existing ones."
          >
            <span class="settings-note-inline" data-test="source-auto-tag-title"
              >this browser</span
            >
            <ToggleControl
              :model-value="autoTagTitle"
              data-test="config-auto-tag-title"
              aria-label="Tag new prints with their title"
              @commit="(value: boolean) => (autoTagTitle = value)"
            />
          </SettingRow>
          <ConfigSettingRow
            schema-key="generate.auto_tag_title"
            :row="rowFor('generate.auto_tag_title')"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
        </template>

        <!-- Phone pairing -->
        <div v-else-if="section.id === 'pairing'" class="settings-body">
          <PairingAccessPanel
            :target="originTarget"
            :suggested-base-url="pairingHost.url"
            host-label="This server"
          />
        </div>

        <template v-else-if="section.id === 'performance'">
          <ConfigSettingRow
            v-for="key in keysOf('performance')"
            :key="key"
            :schema-key="key"
            :row="rowFor(key)"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
        </template>

        <!-- Accounts & tokens -->
        <template v-else-if="section.id === 'accounts'">
          <p v-if="credentialsLoading" class="settings-body" role="status">
            Loading server credentials…
          </p>
          <div
            v-if="credentialsError"
            class="settings-body settings-alert"
            role="alert"
            data-test="credentials-error"
          >
            <p>{{ credentialsError }}</p>
            <button
              type="button"
              class="settings-button"
              data-test="retry-credentials"
              :disabled="credentialsLoading"
              @click="loadCredentials"
            >
              Retry
            </button>
          </div>
          <fieldset
            v-if="credentialsLoaded"
            class="settings-fieldset"
            data-test="credential-fields"
            :disabled="!canChangeCredentials"
            :aria-busy="credentialMutation"
          >
            <legend class="sr-only">Server account tokens</legend>
            <SettingRow
              label="Hugging Face token"
              help="Used to discover and download styles from Hugging Face. Stored on this server."
            >
              <template v-if="hfMasked && !editingHf">
                <code class="settings-mask" data-test="hf-mask">{{
                  hfMasked
                }}</code>
                <span class="settings-note-inline" data-test="hf-source">{{
                  credentialSourceLabel(saved.hf.source)
                }}</span>
                <button
                  type="button"
                  class="settings-button"
                  data-test="replace-hf"
                  @click="editingHf = true"
                >
                  {{ hfHasSavedOverride ? "Replace" : "Override" }}
                </button>
                <button
                  v-if="hfHasSavedOverride"
                  type="button"
                  class="settings-button settings-button--ghost"
                  data-test="clear-hf"
                  @click="removeHf"
                >
                  Clear
                </button>
              </template>
              <template v-else>
                <input
                  id="hf_token"
                  v-model="hfDraft"
                  name="hf_token"
                  type="password"
                  placeholder="hf_…"
                  autocomplete="off"
                  class="settings-input"
                  aria-label="Hugging Face token"
                />
                <button
                  type="button"
                  class="settings-button"
                  data-test="save-hf"
                  :disabled="!hfDraft.trim()"
                  @click="saveHf"
                >
                  Save
                </button>
                <button
                  v-if="editingHf"
                  type="button"
                  class="settings-button settings-button--ghost"
                  @click="editingHf = false"
                >
                  Cancel
                </button>
              </template>
            </SettingRow>

            <SettingRow
              label="Civitai token"
              help="Used to discover and download styles from Civitai. Stored on this server."
            >
              <template v-if="civitaiMasked && !editingCivitai">
                <code class="settings-mask" data-test="civitai-mask">{{
                  civitaiMasked
                }}</code>
                <span class="settings-note-inline" data-test="civitai-source">{{
                  credentialSourceLabel(saved.civitai.source)
                }}</span>
                <button
                  type="button"
                  class="settings-button"
                  data-test="replace-civitai"
                  @click="editingCivitai = true"
                >
                  {{ civitaiHasSavedOverride ? "Replace" : "Override" }}
                </button>
                <button
                  v-if="civitaiHasSavedOverride"
                  type="button"
                  class="settings-button settings-button--ghost"
                  data-test="clear-civitai"
                  @click="removeCivitai"
                >
                  Clear
                </button>
              </template>
              <template v-else>
                <input
                  id="civitai_token"
                  v-model="civitaiDraft"
                  name="civitai_token"
                  type="password"
                  placeholder="cv_…"
                  autocomplete="off"
                  class="settings-input"
                  aria-label="Civitai token"
                />
                <button
                  type="button"
                  class="settings-button"
                  data-test="save-civitai"
                  :disabled="!civitaiDraft.trim()"
                  @click="saveCivitai"
                >
                  Save
                </button>
                <button
                  v-if="editingCivitai"
                  type="button"
                  class="settings-button settings-button--ghost"
                  @click="editingCivitai = false"
                >
                  Cancel
                </button>
              </template>
            </SettingRow>
            <p v-if="credentialMutation" class="settings-body" role="status">
              Updating server credentials…
            </p>
            <p class="settings-body settings-note">
              Tokens are stored privately on this server and used only for
              catalog discovery and downloads. A saved token overrides the
              environment value; clear it to use the environment again.
            </p>
          </fieldset>
        </template>

        <template v-else-if="section.id === 'cloud'">
          <ConfigSettingRow
            v-for="key in keysOf('cloud')"
            :key="key"
            :schema-key="key"
            :row="rowFor(key)"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
        </template>

        <!-- Per-style defaults -->
        <template v-else-if="section.id === 'styleDefaults'">
          <div v-if="showStyleFilter" class="settings-filter">
            <input
              v-model="styleFilter"
              type="search"
              class="settings-input settings-input--wide"
              data-test="style-filter"
              aria-label="Filter styles"
              placeholder="Filter styles…"
            />
          </div>
          <PerStyleDefaultsRow
            v-for="group in visiblePerStyleGroups"
            :key="group.style"
            :style="group.style"
            :rows="group.rows"
            :display-name="styleNames.get(group.style) ?? null"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
          <p v-if="perStyleGroups.length === 0" class="settings-body">
            No style on this machine has its own defaults yet. Change a style's
            settings with
            <code>mold config set models.&lt;style&gt;.&lt;field&gt;</code>.
          </p>
        </template>

        <!-- Profiles -->
        <template v-else-if="section.id === 'profiles'">
          <SettingRow
            label="Active profile"
            help="Each profile keeps its own generation and expansion preferences."
          >
            <select
              class="settings-select"
              data-test="profile-select"
              aria-label="Active profile"
              :value="activeProfile"
              :disabled="switchingProfile"
              @change="
                selectProfile(($event.target as HTMLSelectElement).value)
              "
            >
              <option
                v-for="profile in profiles"
                :key="profile"
                :value="profile"
              >
                {{ profile }}
              </option>
            </select>
          </SettingRow>
          <SettingRow
            label="New profile"
            help="Switching to a name this machine has never seen creates it."
          >
            <input
              v-model="profileName"
              class="settings-input"
              data-test="profile-name"
              aria-label="New profile name"
              :disabled="switchingProfile"
              placeholder="New profile"
            />
            <button
              type="button"
              class="settings-button"
              data-test="profile-create"
              :disabled="switchingProfile || !profileName.trim()"
              @click="createProfile"
            >
              Create &amp; switch
            </button>
          </SettingRow>
          <p v-if="profilesError" class="settings-body" role="status">
            {{ profilesError }}
          </p>
        </template>

        <!-- Advanced -->
        <template v-else-if="section.id === 'advanced'">
          <p class="settings-body settings-note">
            Keys this machine reports that this app has never heard of. On a
            current machine there are none — every setting it knows about has a
            place of its own above.
          </p>
          <ConfigRowItem
            v-for="row in advancedRows"
            :key="row.key"
            :row="row"
            @save="(value: ConfigValue) => saveConfig(row.key, value)"
            @reset="resetConfigKey(row.key)"
          />
        </template>

        <!-- Updates & about -->
        <template v-else-if="section.id === 'updates'">
          <SettingRow label="Version" help="What this machine is running.">
            <span class="settings-value" data-test="about-version">{{
              version
            }}</span>
          </SettingRow>
          <SettingRow label="Build">
            <span class="settings-value" :title="gitSha ?? undefined"
              >{{ gitShaShort }} · {{ buildDate }}</span
            >
          </SettingRow>
          <SettingRow label="Backend" help="What it computes on.">
            <span class="settings-value">{{ backend }}</span>
          </SettingRow>
          <ConfigSettingRow
            v-for="key in keysOf('updates')"
            :key="key"
            :schema-key="key"
            :row="rowFor(key)"
            @save="saveConfig"
            @reset="resetConfigKey"
          />
          <SettingRow label="Processing" help="Where pictures are made.">
            <span class="settings-value">local + your hosts</span>
          </SettingRow>
          <SettingRow label="Core contributors">
            <span class="settings-people">James Brink · Jeffrey Dilley</span>
          </SettingRow>
        </template>
      </template>
    </SettingsShell>
  </div>
</template>

<style scoped>
.settings-page {
  width: 100%;
  box-sizing: border-box;
}

.settings-body {
  margin: 0;
  padding: var(--mold-sp-3);
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
}

.settings-note {
  color: var(--mold-text-dim);
}

.settings-note-inline {
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  white-space: nowrap;
}

.settings-alert {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--mold-sp-2);
  color: var(--mold-error);
  overflow-wrap: anywhere;
}

.settings-panel-row {
  padding: var(--mold-sp-3);
  border-bottom: var(--mold-bw) solid var(--mold-border);
}

.settings-fieldset {
  min-width: 0;
  margin: 0;
  padding: 0;
  border: 0;
}

.settings-value {
  color: var(--mold-text-2);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
  overflow-wrap: anywhere;
}

.settings-people {
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
}

.settings-mask {
  color: var(--mold-text-2);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.04em;
}

.settings-filter {
  padding: var(--mold-sp-2) var(--mold-sp-3);
  border-bottom: var(--mold-bw) solid var(--mold-border);
}

.settings-input,
.settings-select {
  min-width: 0;
  box-sizing: border-box;
  height: var(--mold-ctl-lg, 32px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
}

.settings-input {
  width: 176px;
}

.settings-input--wide {
  width: 100%;
}

.settings-input:focus,
.settings-select:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}

.settings-button {
  height: var(--mold-ctl-md, 26px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
}

.settings-button:disabled {
  opacity: 0.5;
  cursor: default;
}

.settings-button--ghost {
  border-color: transparent;
  background: none;
  color: var(--mold-text-dim);
}

.settings-button--ghost:hover:not(:disabled) {
  color: var(--mold-error);
}

.settings-link {
  border: none;
  background: none;
  color: var(--mold-blue);
  font-size: var(--mold-fs-xs);
  text-decoration: none;
  cursor: pointer;
}

.settings-inline-field {
  display: flex;
  align-items: center;
  gap: var(--mold-sp-2);
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
}
</style>
