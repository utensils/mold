<script setup lang="ts">
import { computed } from "vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import { useSequenceDraftStore, type ClipMode } from "@studio/stores/sequenceDraft";
import type { GenerateForm } from "../../lib/generateForm";
import { outputKindFor } from "../../composables/useCreateOutputKind";

/**
 * The row under the New image toolbar, present only while the kind is Short clip:
 * Simple | Scenes — how the clip gets made — and one sentence saying what
 * the chosen way does.
 *
 * It is a row of its own so the toolbar's Still picture | Short clip | 3-D object
 * control never moves. As a second control on the toolbar it sat between
 * that control and the doors, and choosing Short clip pushed the whole right-hand
 * cluster left by its width — the control a person had just clicked jumped
 * away from the pointer. The same chrome as My images' chip row: the strip
 * belongs to the view, and the canvas absorbs its height.
 *
 * What is on screen answers the toggle: a sequence is the scene-by-scene
 * way, anything else is the plain render. `draft.clipMode` is the remembered
 * preference the Short clip door reads, not the live state, so the two can never
 * disagree on screen.
 */
const props = defineProps<{ form: GenerateForm }>();
const emit = defineEmits<{
  /** Simple ↔ Scenes. The view owns it: it seeds scene 1 and parks the
   *  draft, which needs the chain limits. */
  "set-clip-mode": [mode: ClipMode];
}>();

const draft = useSequenceDraftStore();

// The same decision the toolbar and the title bar read (`useCreateOutputKind`),
// from this form so the strip answers for the form the toolbar renders.
const visible = computed(() => outputKindFor(draft.output, props.form.family) === "clip");

const clipMode = computed<ClipMode>(() => (draft.output === "sequence" ? "scenes" : "simple"));
const clipModeOptions = [
  { value: "simple" as const, label: "Simple" },
  { value: "scenes" as const, label: "Scenes" },
];

/** One plain sentence per way, so the choice explains itself. */
const HINT: Readonly<Record<ClipMode, string>> = {
  simple: "One prompt, one clip.",
  scenes: "Scene by scene, joined into one clip.",
};
const hint = computed(() => HINT[clipMode.value]);

function setClipMode(mode: string | number) {
  const next: ClipMode = mode === "scenes" ? "scenes" : "simple";
  if (next === clipMode.value) return;
  emit("set-clip-mode", next);
}
</script>

<template>
  <div
    v-if="visible"
    data-test="clip-mode-strip"
    class="flex shrink-0 items-center gap-2.5 border-b border-border bg-chrome px-3.5 py-2"
  >
    <SegmentedControl
      data-test="clip-mode"
      class="ms-clipstrip__seg"
      :model-value="clipMode"
      :options="clipModeOptions"
      variant="neutral"
      compact
      label="How to make the clip"
      @update:model-value="setClipMode"
    />
    <span data-test="clip-mode-hint" class="truncate text-xs text-fg-dim">{{ hint }}</span>
  </div>
</template>

<style scoped>
/* Never shrink: its segments are nowrap, so shrinking it only overflows. */
.ms-clipstrip__seg {
  flex: 0 0 auto;
}
</style>
