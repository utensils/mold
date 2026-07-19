<script setup lang="ts">
import { computed, ref, useId, watch } from "vue";
import { seedMode } from "../lib/generateForm";
import { randomSeed } from "../stores/generation";

const props = withDefaults(
  defineProps<{
    lastSeed?: number | null;
    disabled?: boolean;
  }>(),
  { lastSeed: null, disabled: false },
);

const seed = defineModel<string>({ required: true });
const emit = defineEmits<{
  "validity-change": [valid: boolean];
}>();
const uiMode = ref<"random" | "fixed">(seedMode(seed.value));
const noteId = `mobile-seed-note-${useId()}`;

watch(seed, (next) => {
  // External reuse and history actions should reveal the fixed editor. Keep
  // the editor open when its value is cleared mid-entry so focus is stable.
  if (seedMode(next) === "fixed") uiMode.value = "fixed";
});

const seedNote = computed(() => {
  if (uiMode.value === "random") return "New seed for every print.";
  const raw = seed.value.trim();
  if (raw === "") return "Enter a whole number, or choose Random.";
  if (!/^\d+$/.test(raw)) return "Use a whole number, 0 or higher.";
  if (!Number.isSafeInteger(Number(raw))) return "Seed is too large to reproduce exactly.";
  return "Repeat this seed for matching results.";
});

const seedValid = computed(() => {
  if (uiMode.value === "random") return true;
  const raw = seed.value.trim();
  return /^\d+$/.test(raw) && Number.isSafeInteger(Number(raw));
});
const seedInvalid = computed(() => !seedValid.value);

watch(seedValid, (valid) => emit("validity-change", valid), { immediate: true });

function setMode(mode: "random" | "fixed"): void {
  uiMode.value = mode;
  if (mode === "random") {
    seed.value = "";
  } else if (seedMode(seed.value) === "random") {
    seed.value = String(props.lastSeed ?? randomSeed());
  }
}

function useLastSeed(): void {
  if (props.lastSeed === null) return;
  uiMode.value = "fixed";
  seed.value = String(props.lastSeed);
}

function newSeed(): void {
  uiMode.value = "fixed";
  seed.value = String(randomSeed());
}
</script>

<template>
  <fieldset class="mobile-seed-picker" :disabled="disabled">
    <legend class="mobile-seed-legend">Seed</legend>
    <div class="mobile-seed-modes" role="group" aria-label="Seed mode">
      <button
        type="button"
        class="mobile-seed-mode"
        :class="{ 'is-selected': uiMode === 'random' }"
        :aria-pressed="uiMode === 'random'"
        data-test="mobile-seed-mode-random"
        @click="setMode('random')"
      >
        Random
      </button>
      <button
        type="button"
        class="mobile-seed-mode"
        :class="{ 'is-selected': uiMode === 'fixed' }"
        :aria-pressed="uiMode === 'fixed'"
        data-test="mobile-seed-mode-fixed"
        @click="setMode('fixed')"
      >
        Fixed
      </button>
    </div>

    <div v-if="uiMode === 'fixed'" class="mobile-seed-fixed">
      <input
        v-model="seed"
        class="control mobile-seed-input"
        data-selectable
        data-test="mobile-seed-input"
        type="text"
        inputmode="numeric"
        enterkeyhint="done"
        pattern="[0-9]*"
        autocomplete="off"
        autocapitalize="none"
        :spellcheck="false"
        aria-label="Fixed seed value"
        :aria-invalid="seedInvalid || undefined"
        :aria-describedby="noteId"
      />
      <button
        type="button"
        class="secondary-button mobile-seed-new"
        data-test="mobile-new-seed"
        aria-label="Generate a new fixed seed"
        @click="newSeed"
      >
        New value
      </button>
    </div>

    <div class="mobile-seed-footer">
      <p
        class="mobile-seed-note"
        :class="{ 'is-error': seedInvalid }"
        :id="noteId"
        data-test="mobile-seed-note"
        :role="seedInvalid ? 'alert' : undefined"
      >
        {{ seedNote }}
      </p>
      <button
        v-if="uiMode === 'random' && lastSeed !== null"
        type="button"
        class="mobile-seed-use-last"
        data-test="mobile-use-last-seed"
        @click="useLastSeed"
      >
        Use last <span>{{ lastSeed }}</span>
      </button>
    </div>
  </fieldset>
</template>
