<script setup lang="ts">
/*
 * The top Appearance card (Mold Studio Settings, spec §04/G7): theme family
 * and light/dark as segmented rows, interface scale as a slider, and the
 * remaining app-behaviour toggles beneath a divider. All of it drives the
 * existing appPrefs plumbing — nothing here blocks first use.
 */
import { computed } from "vue";
import SegmentedControl, { type SegmentOption } from "@ui/components/SegmentedControl.vue";
import ToggleControl from "./ToggleControl.vue";
import CardSurface from "@ui/components/CardSurface.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { Theme, ThemeFamily } from "../../lib/theme";
import { shortcutLabel } from "../../lib/platform";

const prefs = useAppPrefsStore();

const FAMILY_OPTIONS: SegmentOption<ThemeFamily>[] = [
  { value: "mold", label: "Mold" },
  { value: "safelight", label: "Safelight" },
];

const APPEARANCE_OPTIONS: SegmentOption<Theme>[] = [
  { value: "system", label: "System" },
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
];

const scaleHelp = computed(
  () =>
    `Scale the complete interface, including menus and overlays. Use ${shortcutLabel(
      "+",
    )}, ${shortcutLabel("−")}, or ${shortcutLabel("0")} at any time.`,
);

const BEHAVIOUR_TOGGLES = [
  {
    key: "notifications",
    label: "Notifications",
    help: "Notify when a print or chain finishes while the app is in the background.",
  },
  {
    key: "dockBadge",
    label: "App badge",
    help: "Show this app's active job count on its launcher icon.",
  },
  {
    key: "saveRemoteOutputs",
    label: "Save remote prints locally",
    help: "Also save generations from remote hosts and RunPod into this device's gallery.",
  },
  {
    key: "restoreLastRoute",
    label: "Reopen last view",
    help: "Launch into the view you left instead of Generate.",
  },
] as const;

function toggleValue(key: (typeof BEHAVIOUR_TOGGLES)[number]["key"]): boolean {
  return prefs[key];
}
</script>

<template>
  <CardSurface>
    <!-- Theme family -->
    <div class="flex items-center justify-between gap-4 py-1.5">
      <span class="text-body text-ink">Theme</span>
      <SegmentedControl
        class="w-44 shrink-0"
        :model-value="prefs.themeFamily"
        :options="FAMILY_OPTIONS"
        label="Theme"
        @update:model-value="(v) => prefs.update({ themeFamily: v as ThemeFamily })"
      />
    </div>

    <!-- Light / dark -->
    <div class="mt-2 flex items-center justify-between gap-4 py-1.5">
      <span class="text-body text-ink">Appearance</span>
      <SegmentedControl
        class="w-52 shrink-0"
        :model-value="prefs.theme"
        :options="APPEARANCE_OPTIONS"
        label="Appearance"
        @update:model-value="(v) => prefs.update({ theme: v as Theme })"
      />
    </div>

    <!-- Interface scale -->
    <div class="mt-2 flex items-center justify-between gap-4 py-1.5">
      <span class="text-body text-ink" :title="scaleHelp">Interface size</span>
      <div class="flex shrink-0 items-center gap-3">
        <input
          type="range"
          min="80"
          max="130"
          step="10"
          :value="prefs.uiScalePercent"
          aria-label="Interface size"
          class="w-40 accent-[var(--safelight)]"
          @input="
            (e) => prefs.update({ uiScalePercent: Number((e.target as HTMLInputElement).value) })
          "
        />
        <span class="data-mono w-10 text-right text-caption text-ink-2">
          {{ prefs.uiScalePercent }}%
        </span>
      </div>
    </div>

    <!-- App behaviour -->
    <div class="border-edge mt-3 border-t pt-1">
      <div
        v-for="toggle in BEHAVIOUR_TOGGLES"
        :key="toggle.key"
        class="flex items-center justify-between gap-4 py-2"
      >
        <div class="min-w-0">
          <div class="text-body text-ink">{{ toggle.label }}</div>
          <p class="mt-0.5 text-caption text-ink-3">{{ toggle.help }}</p>
        </div>
        <ToggleControl
          :model-value="toggleValue(toggle.key)"
          :aria-label="toggle.label"
          @commit="(v) => prefs.update({ [toggle.key]: v })"
        />
      </div>
    </div>
  </CardSurface>
</template>
