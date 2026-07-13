<script setup lang="ts">
import SettingRow from "./SettingRow.vue";
import SelectControl from "./SelectControl.vue";
import ToggleControl from "./ToggleControl.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { Theme, ThemeFamily } from "../../lib/ipc";
import { shortcutLabel } from "../../lib/platform";

const prefs = useAppPrefsStore();

const THEMES = [
  { value: "system", label: "System" },
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
];

const THEME_FAMILIES = [
  { value: "safelight", label: "Safelight" },
  { value: "mold", label: "Mold" },
];

const UI_SCALES = [80, 90, 100, 110, 120, 130].map((value) => ({
  value: String(value),
  label: `${value}%`,
}));
</script>

<template>
  <div>
    <SettingRow
      label="Theme"
      help="Safelight is the original darkroom palette. Mold matches the cyan and magenta identity used by the website and logo."
    >
      <SelectControl
        :model-value="prefs.themeFamily"
        :options="THEME_FAMILIES"
        aria-label="Theme"
        @commit="(v) => prefs.update({ themeFamily: v as ThemeFamily })"
      />
    </SettingRow>
    <SettingRow
      label="Appearance"
      help="Follow the operating system or keep the selected theme light or dark. Prints and previews never invert."
    >
      <SelectControl
        :model-value="prefs.theme"
        :options="THEMES"
        aria-label="Appearance"
        @commit="(v) => prefs.update({ theme: v as Theme })"
      />
    </SettingRow>
    <SettingRow
      label="Interface size"
      :help="`Scale the complete interface, including menus and overlays. Use ${shortcutLabel('+')}, ${shortcutLabel('−')}, or ${shortcutLabel('0')} at any time.`"
    >
      <SelectControl
        :model-value="String(prefs.uiScalePercent)"
        :options="UI_SCALES"
        aria-label="Interface size"
        @commit="(v) => prefs.update({ uiScalePercent: Number(v) })"
      />
    </SettingRow>
    <SettingRow
      label="Notifications"
      help="Notify when a print or chain finishes while the app is in the background."
    >
      <ToggleControl
        :model-value="prefs.notifications"
        aria-label="Notifications"
        @commit="(v) => prefs.update({ notifications: v })"
      />
    </SettingRow>
    <SettingRow label="App badge" help="Show this app's active job count on its launcher icon.">
      <ToggleControl
        :model-value="prefs.dockBadge"
        aria-label="App badge"
        @commit="(v) => prefs.update({ dockBadge: v })"
      />
    </SettingRow>
    <SettingRow
      label="Save remote prints locally"
      help="Also save generations from remote hosts and RunPod into this device's gallery."
    >
      <ToggleControl
        :model-value="prefs.saveRemoteOutputs"
        aria-label="Save remote prints locally"
        @commit="(v) => prefs.update({ saveRemoteOutputs: v })"
      />
    </SettingRow>
    <SettingRow label="Reopen last view" help="Launch into the view you left instead of Generate.">
      <ToggleControl
        :model-value="prefs.restoreLastRoute"
        aria-label="Reopen last view"
        @commit="(v) => prefs.update({ restoreLastRoute: v })"
      />
    </SettingRow>
  </div>
</template>
