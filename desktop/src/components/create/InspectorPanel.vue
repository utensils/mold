<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import Stepper from "@ui/components/Stepper.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import { nearestMp } from "@ui/lib/resolution";
import type { GenerateForm } from "../../lib/generateForm";
import { resetFormToModelDefaults, seedMode } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";
import {
  filterModelsForTarget,
  findInstalledModel,
  mergeInstalledModels,
} from "../../lib/generateModels";
import { modelAvailabilityTag, normalizeTargetHost } from "../../lib/hosts";
import { modelDisplayName } from "../../lib/models";
import { modelSource } from "../../lib/modelSource";
import { generationCapabilitiesForFamily } from "../../lib/capabilities";
import { advancedActiveCount } from "../../lib/advancedCount";
import { applyAspectId, applyMp, aspectIdFor } from "../../lib/resolutions";
import { resolutionValidationError, stepsValidationError } from "../../lib/generateValidation";
import { randomSeed } from "../../stores/generation";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { dragWidth } from "../../lib/panelResize";
import SourceGlyph from "../generate/SourceGlyph.vue";
import AdvancedSettings from "./AdvancedSettings.vue";
import { formatGB } from "../../lib/format";
import PanelResizeHandle from "../shell/PanelResizeHandle.vue";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    /** Seed of the most recent finished print — powers "lock last seed". */
    lastSeed?: number | null;
  }>(),
  { lastSeed: null },
);

const emit = defineEmits<{ "append-word": [word: string] }>();

const formStore = useGenerateFormStore();
const models = useModelStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const appPrefs = useAppPrefsStore();
const router = useRouter();

// The inspector is docked to the window's right edge, so dragging its left
// handle left grows it. Keep pointer moves local and persist only on commit.
const draftInspectorWidth = ref<number | null>(null);
const inspectorWidth = computed(() => draftInspectorWidth.value ?? appPrefs.generateParamsWidth);

function onInspectorResize(dx: number) {
  draftInspectorWidth.value = dragWidth("generateParams", appPrefs.generateParamsWidth, dx, "left");
}

async function onInspectorCommit() {
  const width = draftInspectorWidth.value;
  if (width === null) return;
  if (width !== appPrefs.generateParamsWidth) {
    await appPrefs.update({ generateParamsWidth: width });
  }
  draftInspectorWidth.value = null;
}

function onInspectorReset() {
  draftInspectorWidth.value = null;
  void appPrefs.update({ generateParamsWidth: null });
}

const caps = computed(() => generationCapabilitiesForFamily(props.form.family));
const advancedCount = computed(() => advancedActiveCount(props.form));
const advancedExpanded = ref(false);

// ── Model picker (moved verbatim from the previous inspector) ────────────────
const pickerEl = ref<HTMLDivElement | null>(null);
const pickerOpen = ref(false);

const installedModels = computed(() =>
  mergeInstalledModels(models.installed, hostModels.unionInstalled),
);
const selectedModel = computed<ModelEntry | null>(() =>
  findInstalledModel(installedModels.value, props.form.model),
);

const stickyTarget = computed<string | null>(() =>
  normalizeTargetHost(appPrefs.settings?.generateTargetHost ?? null, hosts.all),
);

const pickerModels = computed<ModelEntry[]>(() => {
  const target = stickyTarget.value;
  const fetched = target && target !== "capable" && (hostModels.byHost[target]?.fetchedAt ?? 0) > 0;
  return filterModelsForTarget(
    installedModels.value,
    target,
    fetched ? new Set(hostModels.installedOn(target).map((m) => m.name)) : null,
  );
});

const pickerFamilies = computed<Map<string, ModelEntry[]>>(() => {
  const byName = new Map<string, ModelEntry>();
  for (const m of pickerModels.value) byName.set(m.name, m);
  const groups = new Map<string, ModelEntry[]>();
  for (const m of byName.values()) {
    const list = groups.get(m.family) ?? [];
    list.push(m);
    groups.set(m.family, list);
  }
  return groups;
});

function availabilityTag(m: ModelEntry): string | null {
  if (!hosts.multiHost) return null;
  const target = stickyTarget.value;
  if (target && target !== "capable") return null;
  return modelAvailabilityTag(hostModels.hostsFor(m.name), hosts.all);
}

const stickyHostMissingModel = computed<string | null>(() => {
  const sel = stickyTarget.value;
  if (!sel || sel === "capable" || !props.form.model) return null;
  const host = hosts.all.find((h) => h.id === sel);
  if (!host) return null;
  const ids = hostModels.hostsFor(props.form.model);
  if (ids.length === 0 || ids.includes(sel)) return null;
  return host.label;
});

const modelDescription = computed(() => {
  const m = selectedModel.value;
  if (!m) return null;
  const parts: string[] = [];
  if (m.description && modelDisplayName(m) === m.name) parts.push(m.description);
  if (m.disk_usage_bytes) parts.push(formatGB(m.disk_usage_bytes));
  if (m.is_loaded) parts.push("loaded");
  return parts.length ? parts.join(" · ") : null;
});

function pickModel(m: ModelEntry) {
  formStore.applyModel(m);
  pickerOpen.value = false;
}

function onDocumentPointerDown(event: PointerEvent) {
  if (!pickerOpen.value || !pickerEl.value) return;
  if (!event.composedPath().includes(pickerEl.value)) pickerOpen.value = false;
}

// Force-fresh availability when the picker opens — a model pulled on an extra
// host by another client then shows up the moment the user looks. The demand
// refresh keyed on the set of ready hosts lives in the view so routing is
// model-aware even before the inspector's picker is first opened.
watch(pickerOpen, (open) => {
  if (open) void hostModels.refresh(true);
});

onMounted(() => document.addEventListener("pointerdown", onDocumentPointerDown));
onBeforeUnmount(() => document.removeEventListener("pointerdown", onDocumentPointerDown));

// ── Shape + resolution projection ────────────────────────────────────────────
const shapeId = computed(() => aspectIdFor(props.form.width, props.form.height) ?? "");
const resolutionMp = computed(() => nearestMp(props.form.width, props.form.height));
const resolutionRatio = computed(() => props.form.width / props.form.height);
const resolutionError = computed(() =>
  resolutionValidationError(props.form.width, props.form.height),
);
const stepsError = computed(() => stepsValidationError(props.form.steps));

function onShape(id: string) {
  applyAspectId(props.form, id);
}
function onResolution(mp: number) {
  applyMp(props.form, mp);
}

// ── Seed (mode is UI-owned to avoid focus loss — see the previous ParamPanel) ─
const uiSeedMode = ref<"random" | "fixed">(seedMode(props.form.seed));
watch(
  () => props.form.seed,
  (seed) => {
    if (seedMode(seed) === "fixed") uiSeedMode.value = "fixed";
  },
);
function setSeedMode(mode: "random" | "fixed") {
  uiSeedMode.value = mode;
  if (mode === "random") {
    props.form.seed = "";
  } else if (seedMode(props.form.seed) === "random") {
    props.form.seed = String(props.lastSeed ?? randomSeed());
  }
}
const seedHint = computed(() => {
  if (uiSeedMode.value !== "fixed") return null;
  const raw = props.form.seed.trim();
  if (raw === "") return "Empty — a random seed will be used.";
  if (!Number.isFinite(Number(raw))) return "Not a number — a random seed will be used.";
  return null;
});
function rerollSeed() {
  props.form.seed = String(randomSeed());
}

const batchMax = computed(() => (caps.value.forcesBatchSizeOne ? 1 : 8));

// Same contract as the Advanced pane's Reset, surfaced without opening it:
// the prompt, the model, and any prepared batch size survive.
function resetSettings() {
  resetFormToModelDefaults(props.form, selectedModel.value);
}
</script>

<template>
  <aside class="ms-inspector" data-test="inspector-panel" :style="{ width: `${inspectorWidth}px` }">
    <PanelResizeHandle
      class="absolute inset-y-0 -left-0.5 z-10"
      label="Resize generation settings"
      @resize="onInspectorResize"
      @commit="onInspectorCommit"
      @reset="onInspectorReset"
    />
    <div class="ms-inspector__scroll">
      <div class="ms-inspector__head">
        <span class="ms-inspector__kicker">Settings</span>
        <button
          type="button"
          class="ms-inspector__reset"
          data-test="settings-reset"
          title="Reset settings to model defaults"
          aria-label="Reset settings to model defaults"
          @click="resetSettings"
        >
          ↺ Reset
        </button>
      </div>

      <!-- Model -->
      <div class="ms-field">
        <div class="ms-field__label">Model</div>
        <div ref="pickerEl" class="ms-model">
          <button
            type="button"
            :aria-expanded="pickerOpen"
            class="ms-model__button"
            @click="pickerOpen = !pickerOpen"
          >
            <span data-test="selected-model-name" class="min-w-0 break-all text-left">{{
              selectedModel ? modelDisplayName(selectedModel) : "Choose a model"
            }}</span>
            <span v-if="selectedModel?.disk_usage_bytes" class="data-mono ms-model__size">
              {{ formatGB(selectedModel.disk_usage_bytes) }}
            </span>
          </button>
          <div v-if="pickerOpen" data-test="model-picker-menu" class="ms-model__menu">
            <template v-for="[family, list] in pickerFamilies" :key="family">
              <div class="ms-model__group">{{ family.toUpperCase() }}</div>
              <button
                v-for="m in list"
                :key="m.name"
                type="button"
                class="ms-model__option"
                @click="pickModel(m)"
              >
                <SourceGlyph :source="modelSource(m)" class="mt-0.5 shrink-0 text-ink-3" />
                <span class="min-w-0 flex-1">
                  <span
                    data-test="model-option-name"
                    class="block break-all text-ink"
                    :title="modelDisplayName(m)"
                  >
                    {{ modelDisplayName(m) }}
                  </span>
                  <span
                    v-if="availabilityTag(m)"
                    data-test="model-availability"
                    class="edge-code mt-0.5 block break-all whitespace-normal"
                  >
                    {{ availabilityTag(m) }}
                  </span>
                </span>
                <span
                  class="ms-model__loaded"
                  :class="m.is_loaded ? 'bg-safelight' : 'bg-transparent'"
                  :title="m.is_loaded ? 'On GPU' : ''"
                />
              </button>
            </template>
            <button
              type="button"
              data-test="browse-catalog"
              class="ms-model__browse"
              @click="
                pickerOpen = false;
                void router.push(caps.supportsVideo ? '/models?type=video' : '/models');
              "
            >
              Browse all models →
            </button>
          </div>
        </div>
        <p v-if="modelDescription" class="ms-field__hint">{{ modelDescription }}</p>
        <p v-if="stickyHostMissingModel" class="ms-field__hint">
          Not on {{ stickyHostMissingModel }} — will download there.
        </p>
      </div>

      <!-- Shape -->
      <div class="ms-field">
        <div class="ms-field__label">Shape</div>
        <ShapePicker :model-value="shapeId" label="Aspect ratio" @update:model-value="onShape" />
      </div>

      <!-- Resolution -->
      <div class="ms-field">
        <div class="ms-field__label">Resolution</div>
        <ResolutionSelector
          :model-value="resolutionMp"
          :ratio="resolutionRatio"
          @update:model-value="onResolution"
        />
        <p v-if="resolutionError" class="ms-field__error" role="alert">{{ resolutionError }}</p>
      </div>

      <!-- Detail (steps) -->
      <div class="ms-field">
        <SliderRow
          :model-value="form.steps"
          :min="1"
          :max="60"
          label="Detail"
          :value-label="`${form.steps} steps`"
          @update:model-value="form.steps = $event"
        />
        <p v-if="stepsError" class="ms-field__error" role="alert">{{ stepsError }}</p>
      </div>

      <!-- Prompt strength (guidance) -->
      <div class="ms-field">
        <SliderRow
          :model-value="form.guidance"
          :min="0"
          :max="12"
          :step="0.1"
          label="Prompt strength"
          :value-label="form.guidance.toFixed(1)"
          @update:model-value="form.guidance = $event"
        />
      </div>

      <!-- Seed -->
      <div class="ms-field">
        <div class="ms-field__label">Seed</div>
        <div class="ms-seg" role="group" aria-label="Seed mode">
          <button
            type="button"
            data-test="seed-mode-random"
            :aria-pressed="uiSeedMode === 'random'"
            class="ms-seg__btn"
            :data-on="uiSeedMode === 'random' ? 'true' : undefined"
            @click="setSeedMode('random')"
          >
            Random
          </button>
          <button
            type="button"
            data-test="seed-mode-fixed"
            :aria-pressed="uiSeedMode === 'fixed'"
            class="ms-seg__btn"
            :data-on="uiSeedMode === 'fixed' ? 'true' : undefined"
            @click="setSeedMode('fixed')"
          >
            Fixed
          </button>
        </div>
        <div v-if="uiSeedMode === 'fixed'" class="ms-seed__value">
          <input
            v-model="form.seed"
            data-selectable
            data-test="seed-input"
            type="text"
            inputmode="numeric"
            aria-label="Seed value"
            class="ms-seed__input data-mono"
          />
          <button
            type="button"
            class="ms-seed__reroll"
            title="Reroll this seed"
            aria-label="Reroll this seed"
            @click="rerollSeed"
          >
            <Icon name="reroll" :size="15" />
          </button>
        </div>
        <p v-if="seedHint" data-test="seed-hint" class="ms-field__hint text-safelight">
          {{ seedHint }}
        </p>
        <p v-if="uiSeedMode === 'random'" class="ms-field__hint">
          New seed every print<template v-if="lastSeed !== null">
            ·
            <button
              type="button"
              data-test="lock-last-seed"
              class="ms-seed__lock"
              @click="form.seed = String(lastSeed)"
            >
              lock last ({{ lastSeed }})
            </button></template
          >
        </p>
      </div>

      <!-- Batch -->
      <div class="ms-field ms-field--row">
        <span class="ms-field__label ms-field__label--inline">Batch</span>
        <Stepper
          :model-value="form.batchSize"
          :min="1"
          :max="batchMax"
          label="Batch size"
          @update:model-value="form.batchSize = $event"
        />
      </div>
      <p v-if="caps.forcesBatchSizeOne" class="ms-field__hint -mt-2">
        Locked to 1 — edit models render one at a time.
      </p>

      <!-- Advanced -->
      <button
        type="button"
        class="ms-advanced"
        data-test="open-advanced"
        :aria-expanded="advancedExpanded"
        aria-controls="desktop-inline-advanced"
        @click="advancedExpanded = !advancedExpanded"
      >
        <span class="ms-advanced__label">
          <Icon name="sliders" :size="14" />
          Advanced
        </span>
        <span class="ms-advanced__meta">
          <BadgePill v-if="advancedCount > 0" tone="accent" data-test="advanced-count"
            >{{ advancedCount }} on</BadgePill
          >
          <Icon :name="advancedExpanded ? 'chevron-up' : 'chevron-down'" :size="15" />
        </span>
      </button>
      <AdvancedSettings
        v-if="advancedExpanded"
        id="desktop-inline-advanced"
        :form="form"
        :selected-model="selectedModel"
        :upscalers="models.upscalers"
        @append-word="emit('append-word', $event)"
      />
    </div>
  </aside>
</template>

<style scoped>
.ms-inspector {
  position: relative;
  min-height: 0;
  flex: 0 0 auto;
  border-left: 1px solid var(--edge);
  background: var(--bench);
}
.ms-inspector__scroll {
  height: 100%;
  overflow-x: hidden;
  overflow-y: auto;
  padding: 20px 18px;
}
.ms-inspector__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 16px;
}
.ms-inspector__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.ms-inspector__reset {
  flex-shrink: 0;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 4px 8px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.ms-inspector__reset:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
  color: var(--rebate);
}
.ms-field {
  margin-bottom: 20px;
}
.ms-field--row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.ms-field__label {
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.ms-field__label--inline {
  margin-bottom: 0;
}
.ms-field__hint {
  font-size: 11px;
  color: var(--ink-3);
  margin-top: 6px;
  line-height: 1.4;
}
.ms-field__error {
  font-size: 11px;
  color: var(--stop);
  margin-top: 6px;
}
.ms-model {
  position: relative;
}
.ms-model__button {
  display: flex;
  min-height: 40px;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  border: 1px solid var(--ce);
  border-radius: 9px;
  background: var(--bath);
  padding: 0 12px;
  font-size: 13px;
  color: var(--rebate);
}
.ms-model__size {
  flex-shrink: 0;
  color: var(--ink-3);
}
.ms-model__menu {
  position: absolute;
  z-index: 10;
  margin-top: 4px;
  max-height: 18rem;
  width: 100%;
  overflow-y: auto;
  border: 1px solid var(--edge);
  border-radius: 12px;
  background: var(--bench);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
}
.ms-model__group {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  padding: 8px 8px 4px;
}
.ms-model__option {
  display: flex;
  width: 100%;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 8px;
  text-align: left;
  font-size: 13px;
  color: var(--ink-2);
}
.ms-model__option:hover {
  background: var(--bath);
  color: var(--rebate);
}
.ms-model__loaded {
  margin-top: 8px;
  height: 6px;
  width: 6px;
  flex-shrink: 0;
  border-radius: 9999px;
}
.ms-model__browse {
  display: flex;
  width: 100%;
  align-items: center;
  border-top: 1px solid var(--edge);
  padding: 8px;
  text-align: left;
  font-size: 13px;
  color: var(--halide);
}
.ms-model__browse:hover {
  background: var(--bath);
}
.ms-seg {
  display: flex;
  gap: 3px;
  padding: 3px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: 9px;
}
.ms-seg__btn {
  flex: 1;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  padding: 7px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
}
.ms-seg__btn[data-on="true"] {
  background: var(--bench);
  color: var(--rebate);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
}
.ms-seed__value {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
}
.ms-seed__input {
  height: 32px;
  width: 100%;
  min-width: 0;
  border: 1px solid var(--ce);
  border-radius: 6px;
  background: var(--bath);
  padding: 0 8px;
  font-size: 13px;
  color: var(--rebate);
}
.ms-seed__reroll {
  flex-shrink: 0;
  color: var(--ink-3);
  background: transparent;
  border: 0;
  cursor: pointer;
}
.ms-seed__reroll:hover {
  color: var(--rebate);
}
.ms-seed__lock {
  color: var(--halide);
}
.ms-seed__lock:hover {
  text-decoration: underline;
}
.ms-advanced {
  width: 100%;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: 9px;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    border-color var(--dur-quick) var(--ease);
}
.ms-advanced:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
}
.ms-advanced[aria-expanded="true"] {
  border-color: color-mix(in srgb, var(--safelight) 45%, var(--ce));
  background: color-mix(in srgb, var(--safelight) 7%, transparent);
  color: var(--rebate);
}
.ms-advanced__label,
.ms-advanced__meta {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ms-advanced__meta {
  color: var(--ink-3);
}
</style>
