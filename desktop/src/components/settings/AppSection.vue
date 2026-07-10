<script setup lang="ts">
import SettingRow from "./SettingRow.vue";
import SelectControl from "./SelectControl.vue";
import ToggleControl from "./ToggleControl.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { Theme } from "../../lib/ipc";

const prefs = useAppPrefsStore();

const THEMES = [
  { value: "system", label: "System" },
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
];
</script>

<template>
  <div>
    <SettingRow
      label="Appearance"
      help="Lights on or off. Prints and previews never invert — media stays true either way."
    >
      <SelectControl
        :model-value="prefs.theme"
        :options="THEMES"
        @commit="(v) => prefs.update({ theme: v as Theme })"
      />
    </SettingRow>
    <SettingRow
      label="Notifications"
      help="Notify when a print or chain finishes while the app is in the background."
    >
      <ToggleControl
        :model-value="prefs.notifications"
        @commit="(v) => prefs.update({ notifications: v })"
      />
    </SettingRow>
    <SettingRow label="Dock badge" help="Show the queue depth on the Dock icon.">
      <ToggleControl
        :model-value="prefs.dockBadge"
        @commit="(v) => prefs.update({ dockBadge: v })"
      />
    </SettingRow>
    <SettingRow label="Reopen last view" help="Launch into the view you left instead of Generate.">
      <ToggleControl
        :model-value="prefs.restoreLastRoute"
        @commit="(v) => prefs.update({ restoreLastRoute: v })"
      />
    </SettingRow>
  </div>
</template>
