<script setup lang="ts">
/**
 * The composer's Shape chip AND the picker it opens.
 *
 * It used to be a door: a caret that advertised a menu, wired to an event
 * whose only handler set `inspectorTab = "settings"` — already the value on
 * mount — so clicking it did nothing at all. Worse, its label came from
 * `outputFamilyLabel(width, height)` while the rail's came from
 * `resolveOutputShape`, so the two could state different things about one
 * canvas: the rail highlighting **Source** while the chip read "Square ·
 * 1024", or a size the rail marked approximate stated flatly here.
 *
 * Both now read `useOutputShape`, the one resolver, and both WRITE through it.
 * The Style chip next door has owned its own picker since the redesign; this
 * is the same shape, and the two controls are now genuinely interchangeable
 * rather than one being a shortcut to the other.
 */
import { computed, ref } from "vue";
import Popover from "@ui/components/Popover.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import { useOutputShape } from "../../composables/useOutputShape";
import type { GenerationCapabilities } from "../../lib/capabilities";
import type { CanvasIntent, OutputShapeModel } from "@studio/lib/outputShape";
import type { GenerateForm } from "../../lib/generateForm";

/**
 * `caps` and `contractModel` are handed down rather than recomputed here:
 * the view already resolves both from the same five inputs the form's own
 * validators use, and a control that answered for the checkpoint itself is
 * exactly how the chip came to disagree with the rail in the first place.
 */
const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    caps: GenerationCapabilities;
    contractModel?: OutputShapeModel | null;
    canvasIntent?: CanvasIntent;
  }>(),
  { contractModel: null, canvasIntent: "model-default" },
);
const emit = defineEmits<{ "canvas-intent": [intent: CanvasIntent] }>();

const open = ref(false);
const shape = useOutputShape({
  form: () => props.form,
  canvasIntent: () => props.canvasIntent,
  contractModel: () => props.contractModel,
  supportsSourceImage: () => props.caps.supportsSourceImage,
  onCanvasIntent: (intent) => emit("canvas-intent", intent),
});

/** A 3-D style has no canvas, so it has no chip. */
const canvasless = computed(() => props.caps.canvasless || shape.outputShape.value.canvasless);

/**
 * "Square · 1024" — the resolver's own answer, not a second reading of the
 * pixels. A canvas that follows its source says so instead of naming a size
 * the next render may not use, and an off-ladder size keeps the `≈` the rail
 * shows it with.
 */
const label = computed(() => {
  const resolved = shape.outputShape.value;
  const family =
    resolved.families.find((option) => option.id === resolved.selectedFamilyId)?.label ??
    resolved.selectedFamilyId;
  const { width, height } = props.form;
  const size = width === height ? `${width}` : `${width}×${height}`;
  // The ratio ladder is the resolver's, but the composer speaks plain words
  // where it has one — the chip has always said "Square", and the rail's
  // picker keeps the ladder's own "1:1" beside its swatch.
  const plain = family === "1:1" ? "Square" : family;
  return `${plain} · ${resolved.approximate ? "≈" : ""}${size}`;
});
</script>

<template>
  <Popover
    v-if="!canvasless"
    v-model:open="open"
    class="ms-shape"
    placement="top-start"
    label="Shape and size"
  >
    <template #trigger>
      <button
        type="button"
        data-test="shape-chip"
        class="ms-chip"
        :aria-expanded="open"
        aria-haspopup="dialog"
        title="Shape and size"
        @click="open = !open"
      >
        {{ label }} <span class="ms-chip__caret">▼</span>
      </button>
    </template>
    <div class="ms-shape__menu" data-test="shape-menu">
      <div class="ms-shape__field">
        <div class="ms-shape__label">Shape</div>
        <ShapePicker
          :model-value="shape.shapeId.value"
          :options="shape.shapeOptions.value"
          :approximate="shape.shapeApproximate.value"
          label="Aspect ratio"
          @update:model-value="shape.onShape"
        />
      </div>
      <div class="ms-shape__field">
        <div class="ms-shape__label">Resolution</div>
        <ResolutionSelector
          :model-value="shape.resolutionSizeId.value"
          :ratio="shape.resolutionRatio.value"
          :options="shape.resolutionOptions.value"
          :resolved-width="form.width"
          :resolved-height="form.height"
          :custom-label="shape.sourceResolution.value ? shape.outputShape.value.badge : undefined"
          :status="shape.outputShape.value.status"
          @update:model-value="shape.onResolution"
        />
        <button
          v-if="shape.sourceResolution.value && !shape.followsSource.value"
          type="button"
          class="ms-shape__match"
          data-test="shape-chip-match-source"
          @click="shape.matchSource"
        >
          Match source
        </button>
      </div>
    </div>
  </Popover>
</template>

<style scoped>
/* Slot content is compiled in the parent and never inherits a child's scoped
 * sheet, so the chip's look is defined here beside the button — the same
 * reason StylePicker carries its own copy. */
.ms-shape {
  flex-shrink: 0;
}
.ms-chip {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 10px;
  flex-shrink: 0;
  white-space: nowrap;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-2);
  cursor: pointer;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-chip:hover {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}
.ms-chip__caret {
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
}
.ms-shape__menu {
  display: flex;
  flex-direction: column;
  gap: 14px;
  width: 320px;
  padding: 14px;
}
.ms-shape__label {
  margin-bottom: 8px;
  font-size: var(--mold-fs-xs);
  font-weight: 600;
  color: var(--mold-text-2);
}
.ms-shape__match {
  margin-top: 8px;
  font-size: var(--mold-fs-xs);
  color: var(--mold-accent);
  cursor: pointer;
}
</style>
