<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { parseDeviceListResponse, setDeviceEnabled, type DeviceInfo } from "@studio/api/devices";
import {
  listQueue,
  queuePageRequestForCapacity,
  setQueueDevicePin,
  type QueuePlan,
} from "@studio/api/queuePlan";
import DevicePanel from "@studio/components/DevicePanel.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import LicenseSettingsPanel from "@studio/components/LicenseSettingsPanel.vue";
import MobilePairScanCard from "./MobilePairScanCard.vue";
import { canMutateDevice } from "@studio/lib/deviceLifecycle";
import { apiJsonTo } from "../lib/api/client";
import type { ServerCapabilities, ServerStatus } from "../lib/api/types";
import { describeTransportError } from "../lib/api/errors";
import { openExternal } from "../lib/openExternal";
import {
  THEME_FAMILY_META,
  applyFamilyChoice,
  applyToneChoice,
  familyOf,
  themeId,
  toneChoice,
  toneOf,
  type ThemeFamilyId,
  type ToneChoice,
} from "../lib/theme";
import { mobileHostTarget, type MobileHost } from "./hosts";
import { MOBILE_AUTO_ROUTING_HINT, MOBILE_CAPABLE_ROUTING_HINT } from "./generateTarget";
import type { MobileSettings } from "./settings";
import { subscribeToDeviceSnapshots } from "../lib/api/deviceEvents";
import { fetchServerCapabilities } from "../lib/api/serverCapabilities";

const PRIVACY_POLICY_URL = "https://utensils.io/mold/privacy";

const props = defineProps<{
  settings: MobileSettings;
  hostCount: number;
  appVersion: string;
  host?: MobileHost | null;
  updateChannel?: string;
  /** True while the native camera is already open for a pairing scan. */
  pairingScanning?: boolean;
}>();

const emit = defineEmits<{
  update: [patch: Partial<MobileSettings>];
  "manage-hosts": [];
  "scan-pairing": [];
}>();

/**
 * The whole row toggles, not just the 44×26 switch: on a phone the words are
 * what a thumb lands on. A click that started INSIDE the switch is already
 * handled by it, so forwarding that one too would toggle twice. The switch
 * stays the only announced control and the only keyboard target.
 */
function toggleFromRow(event: MouseEvent, apply: () => void): void {
  if (event.target instanceof Element && event.target.closest(".ms-switch")) return;
  apply();
}

function openPrivacyPolicy(): void {
  void openExternal(PRIVACY_POLICY_URL);
}

const devices = ref<DeviceInfo[] | null>(null);
const deviceCapabilities = ref<ServerCapabilities | null>(null);
const deviceMutations = ref(new Set<string>());
const plan = ref<QueuePlan | null>(null);
const deviceError = ref("");
let deviceEventsAbort: AbortController | null = null;
let devicePollTimer: ReturnType<typeof setTimeout> | null = null;
let deviceRefreshPromise: Promise<void> | null = null;
let deviceRefreshQueued = false;
let deviceServicesEpoch = 0;
let deviceRequestGeneration = 0;

async function loadDevices(): Promise<void> {
  const host = props.host;
  const generation = ++deviceRequestGeneration;
  if (!host) {
    devices.value = null;
    plan.value = null;
    return;
  }
  const isCurrent = () =>
    generation === deviceRequestGeneration &&
    props.host?.id === host.id &&
    props.host.baseUrl === host.baseUrl &&
    props.host.apiKey === host.apiKey;
  const target = mobileHostTarget(host);
  const statusRequest = apiJsonTo<ServerStatus>(target, "/api/status");
  const queueRequest = statusRequest.then((status) =>
    listQueue(target, queuePageRequestForCapacity(status.queue_capacity) ?? null),
  );
  const [deviceResult, capabilityResult, queueResult] = await Promise.allSettled([
    apiJsonTo<unknown>(target, "/api/devices"),
    fetchServerCapabilities(target),
    queueRequest,
  ]);
  if (!isCurrent()) return;

  plan.value = queueResult.status === "fulfilled" ? queueResult.value.plan : null;
  const capabilityPayload =
    capabilityResult.status === "fulfilled" &&
    capabilityResult.value !== null &&
    typeof capabilityResult.value === "object"
      ? capabilityResult.value
      : null;
  deviceCapabilities.value = capabilityPayload;

  let nextDevices: DeviceInfo[] | null = null;
  let deviceFailure: unknown = null;
  if (deviceResult.status === "fulfilled") {
    try {
      nextDevices = parseDeviceListResponse(deviceResult.value).devices;
    } catch (error) {
      deviceFailure = error;
    }
  } else {
    deviceFailure = deviceResult.reason;
  }

  if (deviceFailure === null) {
    devices.value = nextDevices;
    deviceError.value = "";
    return;
  }

  // A fulfilled capabilities response is the compatibility authority. Hosts
  // that predate the additive device API omit this group (and newer servers
  // can explicitly report it unavailable), so their rejected endpoint remains
  // a quiet legacy state. A failed capability request is uncertainty, not
  // evidence that the host is old.
  const legacyDeviceApi =
    capabilityResult.status === "fulfilled" &&
    capabilityPayload !== null &&
    capabilityPayload.devices?.available !== true;
  if (legacyDeviceApi) {
    devices.value = null;
    deviceError.value = "";
    return;
  }

  const detail = describeTransportError(deviceFailure, host.name);
  deviceError.value = `Couldn’t refresh compute devices. ${detail} Mold will try again automatically.`;
}

function requestDeviceRefresh(epoch = deviceServicesEpoch): Promise<void> {
  if (epoch !== deviceServicesEpoch) return Promise.resolve();
  if (deviceRefreshPromise) {
    deviceRefreshQueued = true;
    // Supersede the in-flight result immediately. The queued pass will publish
    // the newest snapshot after the current request settles.
    deviceRequestGeneration += 1;
    return deviceRefreshPromise;
  }

  const refresh = (async () => {
    do {
      deviceRefreshQueued = false;
      await loadDevices();
    } while (deviceRefreshQueued && epoch === deviceServicesEpoch);
  })();
  deviceRefreshPromise = refresh;
  return refresh.finally(() => {
    if (deviceRefreshPromise === refresh) deviceRefreshPromise = null;
  });
}

async function refreshDevicesSafely(epoch: number): Promise<void> {
  try {
    await requestDeviceRefresh(epoch);
  } catch (error) {
    if (epoch !== deviceServicesEpoch || !props.host) return;
    const detail = describeTransportError(error, props.host.name);
    deviceError.value = `Couldn’t refresh compute devices. ${detail} Mold will try again automatically.`;
  }
}

function scheduleDevicePoll(epoch: number): void {
  if (epoch !== deviceServicesEpoch) return;
  devicePollTimer = setTimeout(async () => {
    devicePollTimer = null;
    await refreshDevicesSafely(epoch);
    scheduleDevicePoll(epoch);
  }, 5_000);
}

function stopDeviceServices(): void {
  deviceServicesEpoch += 1;
  deviceRequestGeneration += 1;
  deviceEventsAbort?.abort();
  deviceEventsAbort = null;
  if (devicePollTimer) clearTimeout(devicePollTimer);
  devicePollTimer = null;
  deviceRefreshPromise = null;
  deviceRefreshQueued = false;
}

async function toggleDevice(device: DeviceInfo): Promise<void> {
  if (!props.host || !canMutateDevice(device, deviceCapabilities.value)) return;
  const enabled = !device.desired_enabled;
  deviceMutations.value = new Set(deviceMutations.value).add(device.id);
  try {
    await setDeviceEnabled(mobileHostTarget(props.host), device.id, enabled);
    await requestDeviceRefresh();
  } catch (error) {
    deviceError.value = describeTransportError(error, props.host.name);
  } finally {
    const next = new Set(deviceMutations.value);
    next.delete(device.id);
    deviceMutations.value = next;
  }
}

async function toggleDeviceById(deviceId: string, enabled: boolean): Promise<void> {
  const device = devices.value?.find((candidate) => candidate.id === deviceId);
  if (!device || enabled === device.desired_enabled) return;
  await toggleDevice(device);
}

async function unpinWork(workId: string): Promise<void> {
  if (!props.host) return;
  try {
    await setQueueDevicePin(mobileHostTarget(props.host), workId, null);
    await requestDeviceRefresh();
  } catch (error) {
    deviceError.value = describeTransportError(error, props.host.name);
  }
}

watch(
  () => [props.host?.id, props.host?.baseUrl, props.host?.apiKey] as const,
  () => {
    stopDeviceServices();
    devices.value = null;
    plan.value = null;
    deviceCapabilities.value = null;
    deviceError.value = "";
    if (!props.host) {
      return;
    }
    const epoch = deviceServicesEpoch;
    // Do not make initial rendering depend on SSE support. The stream's
    // onOpen callback refetches again to close the subscribe/bootstrap gap.
    void refreshDevicesSafely(epoch);
    scheduleDevicePoll(epoch);
    deviceEventsAbort = new AbortController();
    subscribeToDeviceSnapshots(
      mobileHostTarget(props.host),
      deviceEventsAbort.signal,
      () => void refreshDevicesSafely(epoch),
    );
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  stopDeviceServices();
});

/*
 * Two independent controls: the list names a THEME, this names the TONE.
 * "Match phone" was a switch beside a list that already said "dark", so the
 * two disagreed; System is now simply one of the three tone positions.
 */
const TONE_OPTIONS = [
  { value: "system" as const, label: "Match phone", help: "Follow your phone’s appearance" },
  { value: "light" as const, label: "Light", help: "Always the light tone" },
  { value: "dark" as const, label: "Dark", help: "Always the dark tone" },
];

const activeFamily = computed(() => familyOf(props.settings.theme));
const tone = computed(() => toneOf(props.settings.theme));
const activeTone = computed<ToneChoice>(() =>
  toneChoice({ theme: props.settings.theme, matchSystem: props.settings.matchSystem }),
);

function pickFamily(family: ThemeFamilyId) {
  emit(
    "update",
    applyFamilyChoice(family, {
      theme: props.settings.theme,
      matchSystem: props.settings.matchSystem,
    }),
  );
}

function pickTone(choice: ToneChoice) {
  emit("update", applyToneChoice(choice, props.settings.theme));
}
</script>

<template>
  <div class="mobile-settings" data-test="mobile-settings">
    <section class="mobile-settings-section" aria-labelledby="mobile-settings-theme-title">
      <div class="mobile-settings-section-copy">
        <h1 id="mobile-settings-theme-title">Theme</h1>
        <p>Choose a look for Mold. Your images and videos keep their original colors.</p>
      </div>

      <fieldset class="mobile-settings-fieldset">
        <legend>Look</legend>
        <div class="mobile-theme-options">
          <label
            v-for="meta in THEME_FAMILY_META"
            :key="meta.id"
            class="mobile-theme-option"
            :data-selected="activeFamily === meta.id"
          >
            <input
              class="sr-only"
              type="radio"
              name="mobile-theme"
              :value="meta.id"
              :checked="activeFamily === meta.id"
              @change="pickFamily(meta.id)"
            />
            <span
              class="mobile-theme-preview"
              :data-theme="themeId(meta.id, tone)"
              aria-hidden="true"
            >
              <span />
              <span />
              <span />
            </span>
            <span class="mobile-theme-option-copy">
              <strong>{{ meta.label }}</strong>
              <small>{{ meta.type }}</small>
            </span>
            <span class="mobile-settings-check" aria-hidden="true">✓</span>
          </label>
        </div>
      </fieldset>

      <fieldset class="mobile-settings-fieldset">
        <legend>Light or dark</legend>
        <label
          v-for="option in TONE_OPTIONS"
          :key="option.value"
          class="mobile-appearance-option"
          :data-selected="activeTone === option.value"
        >
          <input
            class="sr-only"
            type="radio"
            name="mobile-theme-tone"
            :value="option.value"
            :checked="activeTone === option.value"
            @change="pickTone(option.value)"
          />
          <strong>{{ option.label }}</strong>
          <small>{{ option.help }}</small>
        </label>
      </fieldset>
    </section>

    <section class="mobile-settings-section" aria-labelledby="mobile-settings-photos-title">
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-photos-title">Photos</h2>
        <p>Keep newly generated images in your phone’s photo library as well as My images.</p>
      </div>
      <fieldset class="mobile-settings-fieldset">
        <legend>Generated images</legend>
        <div
          class="mobile-photo-setting"
          @click="
            toggleFromRow($event, () =>
              emit('update', { autoSavePhotos: !settings.autoSavePhotos }),
            )
          "
        >
          <span>
            <strong>Save to Photos automatically</strong>
            <small>Videos stay in My images. Open one to watch or save it.</small>
          </span>
          <SwitchToggle
            data-test="mobile-auto-save-photos"
            label="Save to Photos automatically"
            :model-value="settings.autoSavePhotos"
            @update:model-value="emit('update', { autoSavePhotos: $event })"
          />
        </div>
      </fieldset>
    </section>

    <section class="mobile-settings-section" aria-labelledby="mobile-settings-license-title">
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-license-title">Model licenses</h2>
      </div>
      <LicenseSettingsPanel
        :target="host ? mobileHostTarget(host) : null"
        :host-label="host?.name ?? 'Selected machine'"
        :open-external="openExternal"
      />
    </section>

    <section
      class="mobile-settings-section"
      aria-labelledby="mobile-settings-library-title"
      data-test="mobile-settings-library"
    >
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-library-title">My images</h2>
        <p>Choose how new results are organized.</p>
      </div>
      <fieldset class="mobile-settings-fieldset">
        <legend>File under</legend>
        <div
          class="mobile-settings-switch"
          @click="
            toggleFromRow($event, () => emit('update', { autoTagTitle: !settings.autoTagTitle }))
          "
        >
          <span>
            <!-- Never a silent write: the tag this files is always shown on
                 Create as the removable ghost chip, before Generate. -->
            <strong>Tag new prints with their title</strong>
            <small
              >Shown in Make under Name and organize. Remove the tag there to skip it for one
              result.</small
            >
          </span>
          <SwitchToggle
            data-test="mobile-auto-tag-title"
            label="Tag new prints with their title"
            :model-value="settings.autoTagTitle"
            @update:model-value="emit('update', { autoTagTitle: $event })"
          />
        </div>
      </fieldset>
    </section>

    <section class="mobile-settings-section" aria-labelledby="mobile-settings-hosts-title">
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-hosts-title">Machines</h2>
        <p>
          {{
            hostCount === 0
              ? "No machines saved."
              : `${hostCount} machine${hostCount === 1 ? "" : "s"} saved. API keys stay in secure device storage.`
          }}
        </p>
        <!-- Create only offers these while two or more machines are
             reachable; one line each, so the choice is never a mystery. -->
        <p v-if="hostCount > 1" data-test="mobile-settings-auto-hint">
          {{ MOBILE_AUTO_ROUTING_HINT }}
        </p>
        <p v-if="hostCount > 1" data-test="mobile-settings-capable-hint">
          {{ MOBILE_CAPABLE_ROUTING_HINT }}
        </p>
      </div>
      <button
        class="secondary-button mobile-settings-manage"
        type="button"
        @click="emit('manage-hosts')"
      >
        Manage machines
      </button>
      <!-- Pairing used to be reachable only from a disclosure inside the
           Machines tab, which is not where a fresh install looks. -->
      <MobilePairScanCard :scanning="pairingScanning" @scan="emit('scan-pairing')" />
    </section>

    <section
      v-if="devices !== null || plan !== null || deviceError"
      class="mobile-settings-section"
      aria-labelledby="mobile-settings-devices-title"
      data-test="mobile-settings-devices"
    >
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-devices-title">Compute devices</h2>
      </div>
      <DevicePanel
        :devices="devices ?? []"
        :plan="plan"
        :mutable="
          deviceCapabilities?.devices?.lifecycle === true &&
          deviceCapabilities?.dispatch?.v2_authoritative === true
        "
        :restart-enable="deviceCapabilities?.devices?.restart_enable === true"
        show-controls
        :busy-device-ids="[...deviceMutations]"
        @unpin="unpinWork"
        @toggle="toggleDeviceById"
      />
      <p v-if="deviceError" class="status-line error-text" role="alert">{{ deviceError }}</p>
    </section>

    <section class="mobile-settings-section" aria-labelledby="mobile-settings-about-title">
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-about-title">About</h2>
      </div>
      <dl class="mobile-settings-about">
        <div>
          <dt>Version</dt>
          <dd>{{ appVersion }}</dd>
        </div>
        <div>
          <dt>Processing</dt>
          <dd>Connected machines only</dd>
        </div>
        <div>
          <dt>Updates</dt>
          <dd data-test="mobile-update-channel">{{ updateChannel ?? "TestFlight" }}</dd>
        </div>
        <div>
          <dt>Privacy</dt>
          <dd>
            <button
              class="mobile-settings-link"
              data-test="mobile-privacy-policy"
              type="button"
              @click="openPrivacyPolicy"
            >
              Privacy policy
            </button>
          </dd>
        </div>
        <div>
          <dt>Core contributors</dt>
          <dd>James Brink · Jeffrey Dilley</dd>
        </div>
      </dl>
    </section>
  </div>
</template>
