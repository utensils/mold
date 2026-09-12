<script setup lang="ts">
/*
 * The theme picker, shared by every shell: the five families as cards, and one
 * System · Light · Dark control beside them.
 *
 * A card names a THEME and nothing else; the tone control names the TONE. The
 * two are independent, which is why there is no Match-system switch sitting
 * beside a card that already said "dark".
 *
 * Fully controlled: it holds nothing. The caller passes `{ theme, matchSystem }`
 * from wherever it persists them and applies what comes back.
 */
import { computed } from "vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import {
  THEME_FAMILY_META,
  applyFamilyChoice,
  applyToneChoice,
  familyOf,
  themeFamilyMeta,
  themeId,
  toneChoice,
  toneOf,
  type ThemeFamilyId,
  type ThemeId,
  type ToneChoice,
} from "@ui/theme";

const props = defineProps<{ theme: ThemeId; matchSystem: boolean }>();
const emit = defineEmits<{
  (e: "update:theme", value: ThemeId): void;
  (e: "update:matchSystem", value: boolean): void;
}>();

const TONE_OPTIONS = [
  { value: "system" as const, label: "System" },
  { value: "light" as const, label: "Light" },
  { value: "dark" as const, label: "Dark" },
];

/** The card that reads as chosen, and the map each card's band paints from. */
const activeFamily = computed(() => familyOf(props.theme));
const tone = computed(() => toneOf(props.theme));
const activeTone = computed<ToneChoice>(() =>
  toneChoice({ theme: props.theme, matchSystem: props.matchSystem }),
);

const toneHelp = computed(() =>
  props.matchSystem
    ? `Follows this machine: ${themeFamilyMeta(props.theme).label} switches between its light and dark tone.`
    : `${themeFamilyMeta(props.theme).label} stays ${tone.value} whatever this machine does.`,
);

function apply(next: { theme: ThemeId; matchSystem: boolean }) {
  if (next.theme !== props.theme) emit("update:theme", next.theme);
  if (next.matchSystem !== props.matchSystem)
    emit("update:matchSystem", next.matchSystem);
}

function pickFamily(family: ThemeFamilyId) {
  apply(
    applyFamilyChoice(family, {
      theme: props.theme,
      matchSystem: props.matchSystem,
    }),
  );
}

function pickTone(choice: ToneChoice) {
  apply(applyToneChoice(choice, props.theme));
}
</script>

<template>
  <div class="ms-theme-picker">
    <div
      class="ms-theme-grid"
      role="radiogroup"
      aria-label="Theme"
      data-test="theme-select"
    >
      <button
        v-for="meta in THEME_FAMILY_META"
        :key="meta.id"
        type="button"
        role="radio"
        class="ms-theme-card"
        :class="{ 'ms-theme-card--active': activeFamily === meta.id }"
        :aria-checked="activeFamily === meta.id"
        :data-test="`theme-${meta.id}`"
        @click="pickFamily(meta.id)"
      >
        <!-- The theme's own surfaces, painted by its own map: the band carries
             `data-theme`, so ui/tokens.css stays the only place a hex lives.
             The cells read `var(--mold-*)` DIRECTLY — a Tailwind `bg-*` alias
             is substituted where it is DEFINED, on the root, and the
             substituted value is what inherits, so the band's own
             `[data-theme]` block would redefine it far too late for anyone to
             read. Every card painted the theme the app was already wearing.
             See the note in ThemePicker.test.ts; do not "clean these up". -->
        <span
          :data-theme="themeId(meta.id, tone)"
          class="ms-band"
          aria-hidden="true"
        >
          <span class="ms-band__rail" />
          <span class="ms-band__field" />
          <span class="ms-band__card" />
          <span class="ms-band__accent" />
        </span>
        <span class="ms-theme-card__label">{{ meta.label }}</span>
        <span class="ms-theme-card__blurb">{{ meta.blurb }}</span>
        <span class="ms-theme-card__type">{{ meta.type }}</span>
      </button>
    </div>

    <div class="ms-theme-tone">
      <div class="ms-theme-tone__text">
        <div class="ms-theme-tone__label">Light or dark</div>
        <p class="ms-theme-tone__help">{{ toneHelp }}</p>
      </div>
      <div class="ms-theme-tone__control" data-test="theme-tone">
        <SegmentedControl
          :model-value="activeTone"
          :options="TONE_OPTIONS"
          label="Light or dark"
          variant="neutral"
          compact
          @update:model-value="pickTone"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.ms-theme-picker {
  display: flex;
  flex-direction: column;
  gap: var(--mold-sp-3);
  padding: var(--mold-sp-3);
}
.ms-theme-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: var(--mold-sp-2);
}
.ms-theme-card {
  display: flex;
  flex-direction: column;
  gap: var(--mold-sp-1);
  padding: var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: none;
  text-align: left;
  cursor: pointer;
  transition: border-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-theme-card:hover {
  border-color: var(--mold-border-focus);
}
.ms-theme-card--active {
  border-color: var(--mold-blue);
  background: var(--mold-accent-tint);
}
.ms-theme-card__label {
  color: var(--mold-text);
  font-size: var(--mold-fs-sm);
  font-weight: 600;
}
.ms-theme-card__blurb {
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
}
.ms-theme-card__type {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}

/* The swatch band. Each cell reads the theme map the band itself carries, so a
 * nested `[data-theme]` actually reaches it. */
.ms-band {
  display: flex;
  height: 44px;
  overflow: hidden;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-1);
}
.ms-band__rail {
  flex: 0 0 22%;
  background: var(--mold-bg-deep);
}
.ms-band__field {
  flex: 1 1 auto;
  background: var(--mold-bg);
}
.ms-band__card {
  flex: 0 0 22%;
  background: var(--mold-surface);
}
.ms-band__accent {
  flex: 0 0 16%;
  background: var(--mold-blue);
}

.ms-theme-tone {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--mold-sp-4);
}
.ms-theme-tone__text {
  min-width: 0;
}
.ms-theme-tone__label {
  color: var(--mold-text);
  font-size: var(--mold-fs-sm);
}
.ms-theme-tone__help {
  margin: 2px 0 0;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
}
.ms-theme-tone__control {
  flex: none;
}
</style>
