<script setup lang="ts">
/**
 * The options sheet a geometry export opens, when the holding host advertises
 * `capabilities.mesh.export_geometry`. It is the video sheet's chrome around
 * `MeshGeometryFields`: the format is already chosen by the menu entry that
 * opened this, so the only decisions left are the three geometry knobs and,
 * where the caller offers one, the destination.
 *
 * Transport-free like the video sheet. The destination rides BESIDE the
 * options in the `export` event, never inside them: the options are the
 * request body posted to the host, and where the file goes afterwards is the
 * client's business alone.
 */
import { ref, watch } from "vue";
import MeshGeometryFields from "./MeshGeometryFields.vue";
import type { ExportDestination } from "./VideoExportDialog.vue";
import {
  meshGeometryDefaults,
  type MeshBounds,
  type MeshExportGeometryCapabilities,
  type MeshGeometryOptions,
} from "@studio/lib/meshExport";

const props = withDefaults(
  defineProps<{
    open: boolean;
    filename: string;
    /** The container this export writes: obj, stl, ply. */
    format: string;
    /** The holding host's own geometry contract. */
    capabilities: MeshExportGeometryCapabilities;
    /** The viewer's bounding box, so the size line names real extents. */
    bounds?: MeshBounds | null;
    busy?: boolean;
    error?: string;
    destinations?: ExportDestination[];
  }>(),
  {
    bounds: null,
    busy: false,
    error: "",
    destinations: () => [],
  },
);

const emit = defineEmits<{
  close: [];
  export: [geometry: MeshGeometryOptions, destination?: string];
}>();

/** Only ever read when the host advertised no default for this format, which
 * is also when the caller should not have opened this sheet at all. */
const FALLBACK: MeshGeometryOptions = {
  size_mm: null,
  up_axis: "y",
  origin: "floor",
};

const draft = ref<MeshGeometryOptions>({ ...FALLBACK });
const destination = ref<string>("");

// Each opening starts from the HOST's defaults for the format being exported,
// so a size typed for an STL never rides along into the next OBJ.
watch(
  () =>
    [props.open, props.format, props.capabilities, props.destinations] as const,
  ([open]) => {
    if (!open) return;
    draft.value = meshGeometryDefaults(props.capabilities, props.format) ?? {
      ...FALLBACK,
    };
    if (
      !props.destinations.some((choice) => choice.value === destination.value)
    )
      destination.value = props.destinations[0]?.value ?? "";
  },
  { immediate: true },
);

function submit(): void {
  if (props.destinations.length > 1)
    emit("export", { ...draft.value }, destination.value);
  else emit("export", { ...draft.value });
}
</script>

<template>
  <div
    v-if="open"
    class="mesh-export-scrim"
    role="dialog"
    aria-modal="true"
    aria-labelledby="mesh-export-title"
    data-test="mesh-export-dialog"
    @click.self="!busy && emit('close')"
    @keydown.esc.stop="!busy && emit('close')"
  >
    <form class="mesh-export-card" @submit.prevent="submit">
      <div class="mesh-export-heading">
        <div>
          <span class="mesh-export-kicker">Mesh export</span>
          <h2 id="mesh-export-title">Export as {{ format.toUpperCase() }}</h2>
        </div>
        <button
          type="button"
          aria-label="Close export options"
          :disabled="busy"
          @click="emit('close')"
        >
          ×
        </button>
      </div>
      <p class="mesh-export-file">{{ filename }}</p>

      <MeshGeometryFields
        v-model="draft"
        :capabilities="capabilities"
        :format="format"
        :bounds="bounds"
        :disabled="busy"
      />

      <fieldset v-if="destinations.length > 1" class="mesh-export-destination">
        <legend>Destination</legend>
        <div class="mesh-export-options">
          <label v-for="choice in destinations" :key="choice.value">
            <input
              v-model="destination"
              type="radio"
              name="mesh-export-destination"
              :value="choice.value"
            />
            <span>{{ choice.label }}</span>
          </label>
        </div>
      </fieldset>

      <p v-if="error" class="mesh-export-error" role="alert">{{ error }}</p>
      <div class="mesh-export-actions">
        <button type="button" :disabled="busy" @click="emit('close')">
          Cancel
        </button>
        <button
          type="submit"
          class="mesh-export-primary"
          data-test="mesh-export-submit"
          :disabled="busy"
        >
          {{ busy ? "Converting…" : "Export" }}
        </button>
      </div>
    </form>
  </div>
</template>

<style scoped>
.mesh-export-scrim {
  position: fixed;
  z-index: 1000;
  inset: 0;
  display: grid;
  place-items: center;
  padding: max(20px, env(safe-area-inset-top)) 20px
    max(20px, env(safe-area-inset-bottom));
  background: rgb(6 5 10 / 72%);
  backdrop-filter: blur(8px);
}
.mesh-export-card {
  width: min(420px, 100%);
  max-height: 100%;
  overflow: auto;
  padding: 22px;
  border: 1px solid var(--edge, rgb(128 120 140 / 35%));
  border-radius: 18px;
  background: var(--bench, #f7f4ef);
  color: var(--rebate, #171316);
  box-shadow: 0 24px 80px rgb(0 0 0 / 38%);
}
.mesh-export-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}
.mesh-export-heading h2 {
  margin: 4px 0 0;
  font-size: 20px;
  line-height: 1.2;
}
.mesh-export-heading button {
  width: 44px;
  height: 44px;
  border: 0;
  border-radius: 999px;
  background: rgb(128 120 140 / 12%);
  color: inherit;
  font-size: 25px;
}
.mesh-export-kicker,
legend {
  font-family: var(--f-mono, monospace);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3, #777078);
}
.mesh-export-file {
  margin: 14px 0 18px;
  overflow: hidden;
  color: var(--ink-2, #575057);
  font-family: var(--f-mono, monospace);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mesh-export-destination {
  margin: 17px 0 0;
  padding: 0;
  border: 0;
}
legend {
  margin-bottom: 8px;
}
.mesh-export-options {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}
.mesh-export-options label {
  position: relative;
}
.mesh-export-options input {
  position: absolute;
  opacity: 0;
}
.mesh-export-options span {
  display: grid;
  min-height: 44px;
  place-items: center;
  border: 1px solid var(--edge, rgb(128 120 140 / 35%));
  border-radius: 10px;
  font-weight: 650;
}
.mesh-export-options input:checked + span {
  border-color: var(--safelight, #ad5700);
  background: color-mix(in srgb, var(--safelight, #ad5700) 12%, transparent);
  color: var(--safelight, #ad5700);
}
.mesh-export-options input:focus-visible + span {
  outline: 2px solid var(--safelight, #ad5700);
  outline-offset: 2px;
}
.mesh-export-error {
  margin: 17px 0 0;
  color: var(--stop, #b32222);
  font-size: 13px;
}
.mesh-export-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 18px;
}
.mesh-export-actions button {
  min-height: 44px;
  border: 1px solid var(--edge, rgb(128 120 140 / 35%));
  border-radius: 10px;
  background: transparent;
  color: inherit;
  font-weight: 700;
}
.mesh-export-actions .mesh-export-primary {
  border-color: transparent;
  background: var(--safelight, #ad5700);
  color: var(--on-accent, white);
}
button:disabled {
  opacity: 0.55;
}
</style>
