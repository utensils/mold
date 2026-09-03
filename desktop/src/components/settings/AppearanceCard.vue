<script setup lang="ts">
/*
 * Settings ▸ Look: the six themes as cards, the Match-system toggle, interface
 * scale, and the app-behaviour toggles beneath a divider. All of it drives
 * the existing appPrefs plumbing — nothing here blocks first use.
 */
import { computed } from "vue";
import ToggleControl from "./ToggleControl.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { THEME_META, THEME_PAIR, THEME_TONE, themeMeta, type ThemeId } from "../../lib/theme";
import { shortcutLabel } from "../../lib/platform";

const prefs = useAppPrefsStore();

/** "Switches to Blueprint in daylight." — names the paired theme. */
const matchSystemHelp = computed(() => {
  const partner = THEME_PAIR[prefs.theme][THEME_TONE[prefs.theme] === "dark" ? "light" : "dark"];
  const when = THEME_TONE[prefs.theme] === "dark" ? "in daylight" : "after dark";
  return `Switches to ${themeMeta(partner).label} ${when}.`;
});

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
    help: "Launch into the view you left instead of New image.",
  },
] as const;

function toggleValue(key: (typeof BEHAVIOUR_TOGGLES)[number]["key"]): boolean {
  return prefs[key];
}

function pick(theme: ThemeId) {
  void prefs.update({ theme });
}
</script>

<template>
  <div class="p-3.5">
    <!-- Theme -->
    <div class="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Theme">
      <button
        v-for="meta in THEME_META"
        :key="meta.id"
        type="button"
        role="radio"
        :aria-checked="prefs.theme === meta.id"
        :data-test="`theme-${meta.id}`"
        class="flex flex-col gap-1.5 rounded-control border p-2.5 text-left transition-colors duration-100"
        :class="
          prefs.theme === meta.id
            ? 'border-accent bg-accent-tint'
            : 'border-border hover:border-border-focus'
        "
        @click="pick(meta.id)"
      >
        <span class="flex items-center gap-2">
          <span
            class="h-3 w-3 shrink-0 rounded-full"
            :style="{ background: meta.accent }"
            aria-hidden="true"
          />
          <span class="text-sm font-semibold text-fg">{{ meta.label }}</span>
          <span class="ml-auto font-mono text-micro text-fg-dim">{{ meta.tone }}</span>
        </span>
        <span class="text-micro text-fg-dim">{{ meta.blurb }}</span>
        <span class="font-mono text-micro text-fg-faint">{{ meta.type }}</span>
      </button>
    </div>

    <!-- Match system -->
    <div class="mt-3 flex items-center justify-between gap-4 py-1.5">
      <div class="min-w-0">
        <div class="text-sm text-fg">Match system appearance</div>
        <p class="mt-0.5 text-micro text-fg-dim">{{ matchSystemHelp }}</p>
      </div>
      <ToggleControl
        :model-value="prefs.matchSystem"
        aria-label="Match system appearance"
        @commit="(v) => prefs.update({ matchSystem: v })"
      />
    </div>

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
