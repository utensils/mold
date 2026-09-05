<script setup lang="ts">
/**
 * The three geometry knobs a host applies to an OBJ / STL / PLY export, as
 * plain form controls. Transport-free on purpose: the desktop and web
 * lightboxes wrap these in `MeshExportDialog`, and the phone renders them
 * inline in its swipe-up sheet, where a dialog stacked over the viewport
 * would hide the print it belongs to.
 *
 * Every bound, axis and origin offered here is the HOST's own
 * `capabilities.mesh.export_geometry`; nothing is a client constant. The one
 * presentation rule this component owns is that **As stored** is offered only
 * where the format's own default is already "no scaling" — the wire cannot
 * ask a size-defaulting format to skip it.
 */
import { computed } from "vue";
import {
  meshExportSizeLabel,
  meshGeometryDefaults,
  type MeshBounds,
  type MeshExportGeometryCapabilities,
  type MeshExportOrigin,
  type MeshGeometryOptions,
  type MeshUpAxis,
} from "@studio/lib/meshExport";

const props = withDefaults(
  defineProps<{
    modelValue: MeshGeometryOptions;
    /** The holding host's advertised bounds, axes, origins and defaults. */
    capabilities: MeshExportGeometryCapabilities;
    /** The container being exported; decides whether "As stored" is offered. */
    format: string;
    /** The viewer's own bounding box, so the size line can name real extents. */
    bounds?: MeshBounds | null;
    disabled?: boolean;
  }>(),
  { bounds: null, disabled: false },
);

const emit = defineEmits<{ "update:modelValue": [MeshGeometryOptions] }>();

/** Millimetre presets, clamped into the host's range and deduplicated with
 * its own default so the recommended size is always one tap away. */
const SIZE_PRESETS = [50, 100, 200];

const UP_AXIS_LABELS: Record<MeshUpAxis, string> = {
  y: "Y-up · as stored (glTF, Blender OBJ)",
  z: "Z-up · slicers, CAD, Blender STL/PLY",
};
const ORIGIN_LABELS: Record<MeshExportOrigin, string> = {
  center: "Centred",
  floor: "On the bed",
};

function clampSize(value: number): number {
  const { min, max } = props.capabilities.size_mm;
  return Math.min(max, Math.max(min, value));
}

/** "As stored" is a real choice only where the host's own default is null:
 * there is no key that asks a size-defaulting format to skip scaling. */
const offersAsStored = computed(
  () =>
    meshGeometryDefaults(props.capabilities, props.format)?.size_mm === null,
);

const sizeChoices = computed<number[]>(() => {
  const unique = new Set<number>();
  for (const preset of [...SIZE_PRESETS, props.capabilities.size_mm.default]) {
    unique.add(clampSize(preset));
  }
  return [...unique].sort((a, b) => a - b);
});

const sizeSelection = computed(() =>
  props.modelValue.size_mm == null
    ? "stored"
    : String(props.modelValue.size_mm),
);

const sizeLabel = computed(() =>
  meshExportSizeLabel(props.bounds, props.modelValue),
);

const upAxes = computed(() => props.capabilities.up_axes ?? []);
const origins = computed(() => props.capabilities.origins ?? []);

function setSize(size_mm: number | null): void {
  emit("update:modelValue", { ...props.modelValue, size_mm });
}

function setUpAxis(up_axis: MeshUpAxis): void {
  emit("update:modelValue", { ...props.modelValue, up_axis });
}

function setOrigin(origin: MeshExportOrigin): void {
  emit("update:modelValue", { ...props.modelValue, origin });
}

/** The typed size is clamped into the host's range on commit, and the field
 * is written back so a refused number never sits there looking accepted. */
function commitTypedSize(event: Event): void {
  const field = event.target as HTMLInputElement;
  const parsed = Number(field.value);
  if (field.value.trim() === "" || !Number.isFinite(parsed)) {
    field.value =
      props.modelValue.size_mm == null ? "" : String(props.modelValue.size_mm);
    return;
  }
  const clamped = clampSize(parsed);
  field.value = String(clamped);
  setSize(clamped);
}
</script>

<template>
  <div class="mesh-geometry" data-test="mesh-geometry-fields">
    <fieldset :disabled="disabled">
      <legend>Size</legend>
      <div class="mesh-geometry-options">
        <label v-if="offersAsStored">
          <input
            type="radio"
            name="mesh-geometry-size"
            value="stored"
            data-test="mesh-geometry-size-stored"
            :checked="sizeSelection === 'stored'"
            @change="setSize(null)"
          />
          <span>As stored</span>
        </label>
        <label v-for="choice in sizeChoices" :key="choice">
          <input
            type="radio"
            name="mesh-geometry-size"
            :value="String(choice)"
            :data-test="`mesh-geometry-size-${choice}`"
            :checked="sizeSelection === String(choice)"
            @change="setSize(choice)"
          />
          <span>{{ choice }} mm</span>
        </label>
      </div>
      <label class="mesh-geometry-custom">
        <span class="mesh-geometry-custom-label">Longest side</span>
        <input
          type="number"
          inputmode="decimal"
          data-test="mesh-geometry-size-input"
          :min="capabilities.size_mm.min"
          :max="capabilities.size_mm.max"
          :value="modelValue.size_mm ?? ''"
          :placeholder="`${capabilities.size_mm.min}–${capabilities.size_mm.max} mm`"
          @change="commitTypedSize"
        />
      </label>
      <p data-test="mesh-geometry-size-label">{{ sizeLabel }}</p>
    </fieldset>

    <fieldset :disabled="disabled">
      <legend>Up axis</legend>
      <div class="mesh-geometry-options mesh-geometry-options--stacked">
        <label v-for="axis in upAxes" :key="axis">
          <input
            type="radio"
            name="mesh-geometry-up-axis"
            :value="axis"
            :data-test="`mesh-geometry-up-${axis}`"
            :checked="modelValue.up_axis === axis"
            @change="setUpAxis(axis)"
          />
          <span>{{ UP_AXIS_LABELS[axis] ?? axis.toUpperCase() }}</span>
        </label>
      </div>
    </fieldset>

    <fieldset :disabled="disabled">
      <legend>Origin</legend>
      <div class="mesh-geometry-options">
        <label v-for="choice in origins" :key="choice">
          <input
            type="radio"
            name="mesh-geometry-origin"
            :value="choice"
            :data-test="`mesh-geometry-origin-${choice}`"
            :checked="modelValue.origin === choice"
            @change="setOrigin(choice)"
          />
          <span>{{ ORIGIN_LABELS[choice] ?? choice }}</span>
        </label>
      </div>
    </fieldset>
  </div>
</template>

<style scoped>
.mesh-geometry fieldset {
  margin: 0 0 17px;
  padding: 0;
  border: 0;
}
.mesh-geometry fieldset:last-child {
  margin-bottom: 0;
}
legend {
  margin-bottom: 8px;
  font-family: var(--mold-font-mono, monospace);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--mold-text-dim, #777078);
}
.mesh-geometry fieldset > p {
  margin: 8px 0 0;
  color: var(--mold-text-dim, #777078);
  font-family: var(--mold-font-mono, monospace);
  font-size: 12px;
}
.mesh-geometry-options {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}
.mesh-geometry-options--stacked {
  grid-template-columns: minmax(0, 1fr);
}
.mesh-geometry-options label {
  position: relative;
}
.mesh-geometry-options input {
  position: absolute;
  opacity: 0;
}
.mesh-geometry-options span {
  display: grid;
  min-height: 44px;
  padding: 0 8px;
  place-items: center;
  border: 1px solid var(--mold-border, rgb(128 120 140 / 35%));
  border-radius: 10px;
  font-size: 13px;
  font-weight: 650;
  text-align: center;
}
.mesh-geometry-options input:checked + span {
  border-color: var(--mold-blue, #ad5700);
  background: color-mix(in srgb, var(--mold-blue, #ad5700) 12%, transparent);
  color: var(--mold-blue, #ad5700);
}
.mesh-geometry-options input:focus-visible + span {
  outline: 2px solid var(--mold-blue, #ad5700);
  outline-offset: 2px;
}
.mesh-geometry-custom {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 8px;
}
.mesh-geometry-custom-label {
  color: var(--mold-text-dim, #777078);
  font-size: 12px;
}
.mesh-geometry-custom input {
  flex: 1;
  min-width: 0;
  min-height: 44px;
  padding: 0 10px;
  border: 1px solid var(--mold-border, rgb(128 120 140 / 35%));
  border-radius: 10px;
  background: transparent;
  color: inherit;
  font-family: var(--mold-font-mono, monospace);
  font-size: 13px;
}
fieldset:disabled {
  opacity: 0.55;
}
</style>
