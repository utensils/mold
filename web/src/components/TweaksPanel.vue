<script setup lang="ts">
import { ref } from "vue";
import {
  LAB_SWATCHES,
  STUDIO_SWATCHES,
  useTweaks,
  type Density,
  type Direction,
  type LabAccent,
  type StudioAccent,
} from "../composables/useTweaks";

/*
 * Floating panel in the bottom-right. A pill-shaped trigger button is
 * always visible; clicking it opens the panel. We keep the trigger
 * rendered when open as well so the user has a clear way back to
 * closed-state without reaching for Escape.
 */

const open = ref(false);
const { tweaks, update } = useTweaks();

function setDirection(d: Direction) {
  update({ direction: d });
}
function setDensity(d: Density) {
  update({ density: d });
}
function setStudioAccent(k: StudioAccent) {
  update({ accentStudio: k });
}
function setLabAccent(k: LabAccent) {
  update({ accentLab: k });
}
function toggleChips() {
  update({ showChips: !tweaks.value.showChips });
}

function onKey(e: KeyboardEvent) {
  if (e.key === "Escape" && open.value) open.value = false;
}
</script>

<template>
  <button
    type="button"
    class="tweaks-toggle-btn"
    :aria-label="open ? 'Close tweaks' : 'Open tweaks'"
    @click="open = !open"
  >
    <svg
      class="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="3" />
      <path
        d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33 1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82c.2.49.66.84 1.17 1H21a2 2 0 1 1 0 4h-.09c-.51.16-.97.51-1.17 1z"
      />
    </svg>
  </button>

  <div
    v-if="open"
    class="tweaks"
    role="dialog"
    aria-label="Visual tweaks"
    @keydown="onKey"
  >
    <div class="tweaks-head">
      <span class="tweaks-title">Tweaks</span>
      <button class="tweaks-close" aria-label="Close" @click="open = false">
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M6 6l12 12" />
          <path d="M18 6 6 18" />
        </svg>
      </button>
    </div>

    <div class="tweaks-row">
      <span class="tweaks-label">Direction</span>
      <div class="tweaks-seg">
        <button
          :class="{ on: tweaks.direction === 'studio' }"
          @click="setDirection('studio')"
        >
          Studio
        </button>
        <button
          :class="{ on: tweaks.direction === 'lab' }"
          @click="setDirection('lab')"
        >
          Lab
        </button>
      </div>
    </div>

    <div class="tweaks-row">
      <span class="tweaks-label">Density</span>
      <div class="tweaks-seg">
        <button
          :class="{ on: tweaks.density === 'compact' }"
          @click="setDensity('compact')"
        >
          Compact
        </button>
        <button
          :class="{ on: tweaks.density === 'cozy' }"
          @click="setDensity('cozy')"
        >
          Cozy
        </button>
        <button
          :class="{ on: tweaks.density === 'roomy' }"
          @click="setDensity('roomy')"
        >
          Roomy
        </button>
      </div>
    </div>

    <div v-if="tweaks.direction === 'studio'" class="tweaks-row">
      <span class="tweaks-label">Studio accent</span>
      <div class="tweaks-swatches">
        <button
          v-for="s in STUDIO_SWATCHES"
          :key="s.key"
          class="tweaks-swatch"
          :class="{ on: tweaks.accentStudio === s.key }"
          :style="{ background: s.color }"
          :aria-label="s.key"
          @click="setStudioAccent(s.key)"
        />
      </div>
    </div>

    <div v-else class="tweaks-row">
      <span class="tweaks-label">Lab duotone</span>
      <div class="tweaks-swatches">
        <button
          v-for="s in LAB_SWATCHES"
          :key="s.key"
          class="tweaks-swatch"
          :class="{ on: tweaks.accentLab === s.key }"
          :style="{ background: s.bg }"
          :aria-label="s.key"
          @click="setLabAccent(s.key)"
        />
      </div>
    </div>

    <div class="tweaks-row">
      <button
        class="tweaks-toggle"
        :class="{ on: tweaks.showChips }"
        @click="toggleChips"
      >
        <span>Show card chips</span>
        <span class="tweaks-toggle-dot" />
      </button>
    </div>
  </div>
</template>
