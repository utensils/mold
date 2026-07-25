<script setup lang="ts">
import { openExternal } from "../lib/openExternal";
import type { Theme, ThemeFamily } from "../lib/theme";
import type { MobileSettings } from "./settings";

const PRIVACY_POLICY_URL = "https://utensils.io/mold/privacy";

defineProps<{
  settings: MobileSettings;
  hostCount: number;
  appVersion: string;
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
