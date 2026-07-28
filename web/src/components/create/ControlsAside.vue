<script setup lang="ts">
/*
 * Controls aside (Mold Studio Create) — the right rail. width/height stay the
 * persisted source of truth (see `useGenerateForm`); Shape and Resolution are
 * PROJECTIONS of those pixels. Detail (steps) and Prompt strength (guidance)
 * are direct sliders, Seed exposes Random/Fixed (increment stays reachable in
 * Advanced → Output & seed), Batch is a stepper, and the Advanced button
 * surfaces the "N on" badge and opens the drawer.
 */
import { computed } from "vue";
import { useRouter } from "vue-router";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import Stepper from "@ui/components/Stepper.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import { ASPECTS, dimsForMp } from "@ui/lib/resolution";
import type { GenerateFormState } from "../../types";
import type { OutputMode } from "@studio/lib/sequence";
import { generationCapabilitiesForFamily } from "../../lib/generateCapabilities";
import { projectResolution } from "./resolutionProjection";
import HostRoutingPicker from "./HostRoutingPicker.vue";
import { useHostRouting } from "../../composables/useHostRouting";

const props = withDefaults(
  defineProps<{
    modelValue: GenerateFormState;
    family: string;
    /** Count of active advanced fields (drives the badge). */
    advCount?: number;
    /** Phone surface: the Advanced sheet button shows here; on tablet+ the
     * Advanced sections render inline in the controls region instead. */
    mobile?: boolean;
    /** Seed of the most recent finished print — powers "lock last seed"
     * (desktop InspectorPanel parity). */
    lastSeed?: number | null;
    /** Output is a setting, not a place (mockup 1c/3a): One shot | Sequence
     * lives here between the model picker and Shape. */
    output?: OutputMode;
    /** Clips currently parked on the composer rail (sequence caption). */
    clipCount?: number;
  }>(),
  {
    advCount: 0,
    mobile: false,
    lastSeed: null,
    output: "single",
    clipCount: 0,
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: GenerateFormState];
  "update:output": [value: OutputMode];
  "open-advanced": [];
  /* The rail only knows the form, not the catalog row behind it, so the page
   * owns the actual reset (model defaults + the undo offer). */
  "reset-settings": [];
}>();

const capabilities = computed(() =>
  generationCapabilitiesForFamily(props.family),
);
const sequenceMode = computed(() => props.output === "sequence");
// Edit families (Qwen image edit) render one print at a time; a sequence
// renders one timeline.
const batchLocked = computed(
  () => capabilities.value.forcesBatchSizeOne || sequenceMode.value,
);

const outputSegments = [
  { value: "single", label: "One shot" },
  { value: "sequence", label: "Sequence" },
] as const;

// Reroll: a fresh random seed for the next print without leaving Fixed mode —
// mirrors the desktop inspector's reroll. Switches to Random so the server
// draws a new seed each generate.
function reroll() {
  patch({ seedMode: "random", seed: null });
}

// Generation target (spec §08 multi-host). The picker owns the routing choice —
// Auto, Most capable, or a sticky host — and the submit path in CreatePage
// resolves the same persisted pick through the same singleton, so what the row
// claims and where the job lands can't drift apart.
const router = useRouter();
const routing = useHostRouting();
function openMachines() {
  void router?.push("/machines");
}

const projection = computed(() =>
  projectResolution(props.modelValue.width, props.modelValue.height),
);

const aspectId = computed(() => projection.value.aspectId ?? "");
const mp = computed(() => projection.value.mp);

const currentRatio = computed(() => {
  const a = ASPECTS.find((x) => x.id === projection.value.aspectId);
  if (a) return a.ratio;
  const { width, height } = props.modelValue;
  return height ? width / height : 1;
});

function patch(patch: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...patch });
}

function setAspect(id: string) {
  const a = ASPECTS.find((x) => x.id === id);
  if (!a) return;
  const dims = dimsForMp(mp.value, a.ratio);
  patch({ width: dims.width, height: dims.height });
}

function setMp(nextMp: number) {
  const dims = dimsForMp(nextMp, currentRatio.value);
  patch({ width: dims.width, height: dims.height });
}

// Seed: Random / Fixed segments. "Fixed" covers the persisted static and
// increment modes so an increment set in Advanced survives a re-click.
const seedSegments = [
  { value: "random", label: "Random" },
  { value: "fixed", label: "Fixed" },
] as const;
const seedSegment = computed(() =>
  props.modelValue.seedMode === "random" ? "random" : "fixed",
);
function setSeedSegment(v: "random" | "fixed") {
  if (v === "random") {
    patch({ seedMode: "random" });
  } else if (props.modelValue.seedMode === "random") {
    patch({ seedMode: "static" });
  }
}
function setSeed(value: number) {
  patch({ seed: Number.isFinite(value) ? value : null });
}
// Lock last seed (desktop parity): pin the previous print's seed so the next
// generate reproduces it. Switching to the Fixed segment reveals the input.
function lockLastSeed() {
  if (props.lastSeed === null) return;
  patch({ seedMode: "static", seed: props.lastSeed });
}
</script>

<template>
  <aside class="controls" data-test="controls-aside">
    <div class="controls__head">
      <span class="controls__kicker">Settings</span>
      <button
        type="button"
        class="controls__reset"
        data-test="settings-reset"
        aria-label="Reset settings to model defaults"
        title="Reset settings to model defaults"
        @click="emit('reset-settings')"
      >
        ↺ Reset
      </button>
    </div>

    <div class="controls__group controls__output" data-test="output-card">
      <div class="controls__label">Output</div>
      <SegmentedControl
        data-test="output-mode"
        :model-value="output"
        :options="outputSegments"
        label="Output"
        @update:model-value="emit('update:output', $event as OutputMode)"
      />
      <p v-if="sequenceMode" class="controls__hint" data-test="output-caption">
        {{ clipCount }} clips on the composer rail · switching back keeps clip 1
        and parks the rest.
      </p>
    </div>

    <div class="controls__group">
      <div class="controls__label">Shape</div>
      <ShapePicker
        :model-value="aspectId"
        label="Shape"
        @update:model-value="setAspect"
      />
    </div>

    <div class="controls__group">
      <div class="controls__label">Resolution</div>
      <ResolutionSelector
        :model-value="mp"
        :ratio="currentRatio"
        @update:model-value="setMp"
      />
    </div>

    <div class="controls__group">
      <SliderRow
        label="Detail"
        :model-value="modelValue.steps"
        :min="1"
        :max="60"
        :value-label="`${modelValue.steps} steps`"
        @update:model-value="patch({ steps: $event })"
      />
    </div>

    <div class="controls__group">
      <SliderRow
        label="Prompt strength"
        :model-value="modelValue.guidance"
        :min="0"
        :max="12"
        :step="0.1"
        :value-label="modelValue.guidance.toFixed(1)"
        @update:model-value="patch({ guidance: $event })"
      />
    </div>

    <div class="controls__group">
      <div class="controls__seed-head">
        <span class="controls__label controls__label--inline">Seed</span>
        <button
          type="button"
          class="controls__reroll"
          data-test="seed-reroll"
          title="New random seed next print"
          @click="reroll"
        >
          <Icon name="reroll" :size="13" />
          reroll
        </button>
      </div>
      <SegmentedControl
        data-test="seed-seg"
        :model-value="seedSegment"
        :options="seedSegments"
        label="Seed mode"
        @update:model-value="setSeedSegment"
      />
      <input
        v-if="seedSegment === 'fixed'"
        class="controls__seed"
        data-test="controls-seed"
        type="number"
        min="0"
        placeholder="Seed"
        :value="modelValue.seed ?? ''"
        @input="setSeed(Number(($event.target as HTMLInputElement).value))"
      />
      <p v-if="seedSegment === 'random'" class="controls__hint">
        New seed every print<template v-if="lastSeed !== null">
          ·
          <button
            type="button"
            data-test="lock-last-seed"
            class="controls__lock"
            @click="lockLastSeed"
          >
            lock last ({{ lastSeed }})
          </button></template
        >
      </p>
    </div>

    <div class="controls__group">
      <div class="controls__batch">
        <span class="controls__label controls__label--inline">Batch</span>
        <Stepper
          :model-value="batchLocked ? 1 : modelValue.batchSize"
          :min="1"
          :max="batchLocked ? 1 : 8"
          label="Batch size"
          @update:model-value="patch({ batchSize: $event })"
        />
      </div>
      <p v-if="batchLocked" class="controls__hint" data-test="batch-locked">
        {{
          sequenceMode
            ? "locked to 1 — a sequence renders one timeline."
            : "locked to 1 — edit models render one print at a time."
        }}
      </p>
    </div>

    <button
      v-if="mobile"
      type="button"
      class="controls__advanced"
      data-test="open-advanced"
      @click="emit('open-advanced')"
    >
      <Icon name="sliders" :size="14" />
      Advanced
      <BadgePill v-if="advCount > 0" data-test="adv-badge"
        >{{ advCount }} on</BadgePill
      >
    </button>

    <HostRoutingPicker
      :hosts="routing.hosts.value"
      :target-id="routing.targetId.value"
      @select="routing.setTarget"
      @open-machines="openMachines"
    />
  </aside>
</template>

<style scoped>
.controls {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 18px;
}

.controls__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 9px;
  margin-bottom: 16px;
}

.controls__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}

/* Matches the Advanced card's inline Reset pill so the two read as one
 * family of "put this section back" actions. */
.controls__reset {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 4px 11px;
  border-radius: var(--radius-pill);
  font-size: 11.5px;
  font-weight: 600;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.controls__reset:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}

.controls__group {
  margin-bottom: 20px;
}

/* Output is the one mode-defining setting — a highlighted card so it reads
 * apart from the tuning sliders below it. */
.controls__output {
  border: 1px solid var(--sel-border, var(--ce));
  background: var(--sel-bg, var(--bath));
  border-radius: var(--radius-card);
  padding: 12px;
}

.controls__label {
  font-size: 12px;
  color: var(--ink-2);
  margin-bottom: 9px;
  font-weight: 600;
}

.controls__label--inline {
  margin-bottom: 0;
}

.controls__seed {
  width: 100%;
  box-sizing: border-box;
  margin-top: 9px;
  height: 38px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 13px;
  padding: 0 12px;
  outline: none;
}

.controls__batch {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.controls__seed-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 9px;
}

.controls__reroll {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  cursor: pointer;
  padding: 0;
}
.controls__reroll:hover {
  color: var(--safelight);
}

.controls__hint {
  margin: 8px 0 0;
  font-size: 11px;
  color: var(--ink-3);
  line-height: 1.4;
}

.controls__lock {
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  text-decoration: underline;
  text-underline-offset: 2px;
  cursor: pointer;
  padding: 0;
}
.controls__lock:hover {
  color: var(--safelight);
}

.controls__advanced {
  width: 100%;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: var(--radius-control);
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 16px;
  cursor: pointer;
}
</style>
