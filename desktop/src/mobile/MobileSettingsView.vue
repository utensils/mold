<script setup lang="ts">
import { ref, watch } from "vue";
import { parseDeviceListResponse, setDeviceEnabled, type DeviceInfo } from "@studio/api/devices";
import {
  canMutateDevice,
  deviceActionLabel,
  deviceLifecycleMessage,
  deviceStateLabel,
} from "@studio/lib/deviceLifecycle";
import { apiJsonTo } from "../lib/api/client";
import type { ServerCapabilities } from "../lib/api/types";
import { describeTransportError } from "../lib/api/errors";
import { openExternal } from "../lib/openExternal";
import type { Theme, ThemeFamily } from "../lib/theme";
import { mobileHostTarget, type MobileHost } from "./hosts";
import type { MobileSettings } from "./settings";

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
const deviceError = ref("");

async function loadDevices(): Promise<void> {
  if (!props.host) {
    devices.value = null;
    return;
  }
  try {
    const target = mobileHostTarget(props.host);
    const value = await apiJsonTo<unknown>(target, "/api/devices");
    const capabilities = await apiJsonTo<ServerCapabilities>(target, "/api/capabilities").catch(
      () => null,
    );
    devices.value = parseDeviceListResponse(value).devices;
    deviceCapabilities.value = capabilities;
    deviceError.value = "";
  } catch {
    // Older hosts do not expose runtime lifecycle controls.
    devices.value = null;
    deviceCapabilities.value = null;
  }
}

async function toggleDevice(device: DeviceInfo): Promise<void> {
  if (!props.host || !canMutateDevice(device, deviceCapabilities.value)) return;
  const enabled = !device.desired_enabled;
  deviceMutations.value = new Set(deviceMutations.value).add(device.id);
  try {
    await setDeviceEnabled(mobileHostTarget(props.host), device.id, enabled);
    await loadDevices();
  } catch (error) {
    deviceError.value = describeTransportError(error, props.host.name);
  } finally {
    const next = new Set(deviceMutations.value);
    next.delete(device.id);
    deviceMutations.value = next;
  }
}

watch(() => props.host?.id, loadDevices, { immediate: true });
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
      v-if="devices !== null"
      class="mobile-settings-section"
      aria-labelledby="mobile-settings-devices-title"
      data-test="mobile-settings-devices"
    >
      <div class="mobile-settings-section-copy">
        <h2 id="mobile-settings-devices-title">GPU devices</h2>
        <p>{{ deviceLifecycleMessage(deviceCapabilities) }}</p>
      </div>
      <p v-if="deviceError" class="status-line error-text" role="alert">{{ deviceError }}</p>
      <ul class="mobile-data-list">
        <li v-for="device in devices" :key="device.id" data-test="device-row">
          <div>
            <strong>{{ device.name }}</strong>
            <span>
              {{ device.ordinal == null ? device.backend.toUpperCase() : `GPU ${device.ordinal}` }}
              · {{ deviceStateLabel(device) }}
            </span>
          </div>
          <button
            type="button"
            class="secondary-button"
            :data-test="`mobile-settings-device-toggle-${device.ordinal ?? device.id}`"
            :disabled="
              !canMutateDevice(device, deviceCapabilities) || deviceMutations.has(device.id)
            "
            @click="toggleDevice(device)"
          >
            {{ deviceActionLabel(device, deviceCapabilities) }}
          </button>
        </li>
      </ul>
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
