<script setup lang="ts">
/*
 * The top Appearance card (Mold Studio Settings, spec §04/G7): the six themes
 * as cards, the Match-system toggle, interface scale, and the remaining
 * app-behaviour toggles beneath a divider. All of it drives the existing
 * appPrefs plumbing — nothing here blocks first use.
 */
import { computed } from "vue";
import ToggleControl from "./ToggleControl.vue";
import CardSurface from "@ui/components/CardSurface.vue";
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
    help: "Launch into the view you left instead of Create.",
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
  <CardSurface>
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
            ? 'border-safelight bg-sel-bg'
            : 'border-edge hover:border-safelight'
        "
        @click="pick(meta.id)"
      >
        <span class="flex items-center gap-2">
          <span
            class="h-3 w-3 shrink-0 rounded-full"
            :style="{ background: meta.accent }"
            aria-hidden="true"
          />
          <span class="text-body font-semibold text-ink">{{ meta.label }}</span>
          <span class="ml-auto font-utility text-caption text-ink-3">{{ meta.tone }}</span>
        </span>
        <span class="text-caption text-ink-3">{{ meta.blurb }}</span>
        <span class="font-utility text-caption text-ink-3">{{ meta.type }}</span>
      </button>
    </div>

    <!-- Match system -->
    <div class="mt-3 flex items-center justify-between gap-4 py-1.5">
      <div class="min-w-0">
        <div class="text-body text-ink">Match system appearance</div>
        <p class="mt-0.5 text-caption text-ink-3">{{ matchSystemHelp }}</p>
      </div>
      <ToggleControl
        :model-value="prefs.matchSystem"
        aria-label="Match system appearance"
        @commit="(v) => prefs.update({ matchSystem: v })"
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
