<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from "vue";
import { parseDeviceListResponse, setDeviceEnabled, type DeviceInfo } from "@studio/api/devices";
import { listQueue, setQueueDevicePin, type QueuePlan } from "@studio/api/queuePlan";
import DevicePanel from "@studio/components/DevicePanel.vue";
import { canMutateDevice } from "@studio/lib/deviceLifecycle";
import { apiJsonTo } from "../lib/api/client";
import type { ServerCapabilities } from "../lib/api/types";
import { describeTransportError } from "../lib/api/errors";
import { openExternal } from "../lib/openExternal";
import type { Theme, ThemeFamily } from "../lib/theme";
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
}>();

const emit = defineEmits<{
  update: [patch: Partial<MobileSettings>];
  "manage-hosts": [];
}>();

const families: Array<{
  value: ThemeFamily;
  label: string;
  description: string;
}> = [
  { value: "mold", label: "Mold", description: "Cyan and magenta studio color." },
  { value: "safelight", label: "Safelight", description: "Warm, classic darkroom color." },
];

const appearances: Array<{ value: Theme; label: string; description: string }> = [
  { value: "system", label: "System", description: "Match iPhone" },
  { value: "dark", label: "Dark", description: "Lights off" },
  { value: "light", label: "Light", description: "Lights on" },
];

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
  const [deviceResult, capabilityResult, queueResult] = await Promise.allSettled([
    apiJsonTo<unknown>(target, "/api/devices"),
    fetchServerCapabilities(target),
    listQueue(target),
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
</script>

<template>
  <div class="mobile-settings" data-test="mobile-settings">
    <section class="mobile-settings-section" aria-labelledby="mobile-settings-theme-title">
      <div class="mobile-settings-section-copy">
        <h1 id="mobile-settings-theme-title">Theme</h1>
        <p>Change the chrome without changing the color of your prints or videos.</p>
      </div>

      <fieldset class="mobile-settings-fieldset">
        <legend>Color family</legend>
        <div class="mobile-theme-options">
          <label
            v-for="family in families"
            :key="family.value"
            class="mobile-theme-option"
            :data-selected="settings.themeFamily === family.value"
          >
            <input
              class="sr-only"
              type="radio"
              name="mobile-theme-family"
              :value="family.value"
              :checked="settings.themeFamily === family.value"
              @change="emit('update', { themeFamily: family.value })"
            />
            <span class="mobile-theme-preview" :data-theme-family="family.value" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
            <span class="mobile-theme-option-copy">
              <strong>{{ family.label }}</strong>
              <small>{{ family.description }}</small>
            </span>
            <span class="mobile-settings-check" aria-hidden="true">✓</span>
          </label>
        </div>
      </fieldset>

      <fieldset class="mobile-settings-fieldset">
        <legend>Appearance</legend>
        <div class="mobile-appearance-options">
          <label
            v-for="appearance in appearances"
            :key="appearance.value"
            class="mobile-appearance-option"
            :data-selected="settings.theme === appearance.value"
          >
            <input
              class="sr-only"
              type="radio"
              name="mobile-theme-appearance"
              :value="appearance.value"
              :checked="settings.theme === appearance.value"
              @change="emit('update', { theme: appearance.value })"
            />
            <strong>{{ appearance.label }}</strong>
            <small>{{ appearance.description }}</small>
          </label>
        </div>
      </fieldset>
    </section>

    <section class="mobile-settings-section" aria-labelledby="mobile-settings-photos-title">
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-photos-title">Photos</h2>
        <p>Keep newly generated images in your iPhone photo library as well as Mold Library.</p>
      </div>
      <fieldset class="mobile-settings-fieldset">
        <legend>Generated images</legend>
        <label class="mobile-photo-setting">
          <span>
            <strong>Save to Photos automatically</strong>
            <small>Videos remain in Mold Library and can be streamed from their host.</small>
          </span>
          <input
            name="mobile-auto-save-photos"
            type="checkbox"
            :checked="settings.autoSavePhotos"
            @change="
              emit('update', {
                autoSavePhotos: ($event.target as HTMLInputElement).checked,
              })
            "
          />
        </label>
      </fieldset>
    </section>

    <section
      class="mobile-settings-section"
      aria-labelledby="mobile-settings-library-title"
      data-test="mobile-settings-library"
    >
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-library-title">Library</h2>
        <p>How a new print is filed the moment it is developed.</p>
      </div>
      <fieldset class="mobile-settings-fieldset">
        <legend>File under</legend>
        <label class="mobile-photo-setting">
          <span>
            <!-- Never a silent write: the tag this files is always shown on
                 Create as the removable ghost chip, before Generate. -->
            <strong>Tag new prints with their title</strong>
            <small>Shown on Create as a removable chip — remove it to opt one print out.</small>
          </span>
          <input
            name="mobile-auto-tag-title"
            type="checkbox"
            :checked="settings.autoTagTitle"
            @change="
              emit('update', {
                autoTagTitle: ($event.target as HTMLInputElement).checked,
              })
            "
          />
        </label>
      </fieldset>
    </section>

    <section class="mobile-settings-section" aria-labelledby="mobile-settings-hosts-title">
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-hosts-title">Remote hosts</h2>
        <p>
          {{
            hostCount === 0
              ? "No hosts saved."
              : `${hostCount} host${hostCount === 1 ? "" : "s"} saved. API keys stay in your iPhone Keychain.`
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
        Manage hosts
      </button>
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
          <dd>Remote hosts only</dd>
        </div>
        <div>
          <dt>Updates</dt>
          <dd>TestFlight</dd>
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
