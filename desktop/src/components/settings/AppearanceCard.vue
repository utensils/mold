<script setup lang="ts">
/*
 * Settings ▸ Look: the theme picker, interface scale, and the app-behaviour
 * toggles beneath a divider. All of it drives the existing appPrefs plumbing —
 * nothing here blocks first use.
 *
 * The theme cards and the System · Light · Dark control are the shared kit's
 * `ThemePicker`, so the app, the browser and the phone pick a theme the same
 * way. Interface scale and the four behaviour toggles stay here: they are
 * properties of THIS app, and a browser tab has neither.
 */
import { computed } from "vue";
import ThemePicker from "@studio/components/settings/ThemePicker.vue";
import ToggleControl from "@studio/components/settings/ToggleControl.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { ThemeId } from "../../lib/theme";
import type { AppSettings } from "../../lib/api/types";
import { shortcutLabel } from "../../lib/platform";

const prefs = useAppPrefsStore();

/**
 * One write per choice, even when a choice moves both fields.
 *
 * `ThemePicker` emits `update:theme` and `update:matchSystem` separately, and
 * picking System moves both. `prefs.update` re-reads settings.json before it
 * merges — deliberately, so other writers are not clobbered — so two
 * overlapping calls would both read the pre-change file and the second would
 * erase the first field. Coalescing on the microtask keeps one click to one
 * write.
 */
let pending: Partial<AppSettings> | null = null;
function queue(patch: Partial<AppSettings>) {
  if (pending) {
    Object.assign(pending, patch);
    return;
  }
  pending = { ...patch };
  void Promise.resolve().then(() => {
    const next = pending;
    pending = null;
    if (next) void prefs.update(next);
  });
}

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
    label: "Save pictures from other machines here",
    help: "Keep a copy of anything another machine or a rented GPU makes in My images on this device.",
  },
  {
    key: "restoreLastRoute",
    label: "Reopen last view",
    help: "Launch into the view you left instead of New image.",
  },
] as const;

function toggleValue(key: (typeof BEHAVIOUR_TOGGLES)[number]["key"]): boolean {
  return prefs[key];
}

</script>

<template>
  <div class="p-3.5">
    <ThemePicker
      :theme="prefs.theme"
      :match-system="prefs.matchSystem"
      @update:theme="(value: ThemeId) => queue({ theme: value })"
      @update:match-system="(value: boolean) => queue({ matchSystem: value })"
    />

    <!-- Interface scale -->
    <div class="mt-2 flex items-center justify-between gap-4 py-1.5">
      <span class="text-sm text-fg" :title="scaleHelp">Interface size</span>
      <div class="flex shrink-0 items-center gap-3">
        <input
          type="range"
          min="80"
          max="130"
          step="10"
          :value="prefs.uiScalePercent"
          aria-label="Interface size"
          class="w-40 accent-accent"
          @input="
            (e) => prefs.update({ uiScalePercent: Number((e.target as HTMLInputElement).value) })
          "
        />
        <span class="w-10 text-right font-mono text-micro text-fg-2">
          {{ prefs.uiScalePercent }}%
        </span>
      </div>
    </div>

    <!-- App behaviour -->
    <div class="mt-3 border-t border-border pt-1">
      <div
        v-for="toggle in BEHAVIOUR_TOGGLES"
        :key="toggle.key"
        class="flex items-center justify-between gap-4 py-2"
      >
        <div class="min-w-0">
          <div class="text-sm text-fg">{{ toggle.label }}</div>
          <p class="mt-0.5 text-micro text-fg-dim">{{ toggle.help }}</p>
        </div>
        <ToggleControl
          :model-value="toggleValue(toggle.key)"
          :aria-label="toggle.label"
          @commit="(v) => prefs.update({ [toggle.key]: v })"
        />
      </div>
    </div>
  </div>
</template>
