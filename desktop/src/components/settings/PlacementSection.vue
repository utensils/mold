<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { modelDisplayName } from "@mold/studio";
import { apiJson } from "../../lib/api/client";
import {
  deleteModelPlacement,
  getModelPlacement,
  putModelPlacement,
} from "../../lib/api/placement";
import {
  defaultPlacement,
  optionToRef,
  refToOption,
  setAdvancedField,
  setTextEncoders,
  supportsAdvanced,
  type AdvancedPlacement,
  type DevicePlacement,
} from "../../lib/placement";
import type { ResourceSnapshot } from "../../lib/api/types";
import { useModelStore } from "../../stores/models";
import { useToastStore } from "../../stores/toasts";

interface GpuEntry {
  ordinal: number;
  name: string;
}

const models = useModelStore();
const toasts = useToastStore();
const { installed } = storeToRefs(models);

const selectedModel = ref<string>("");
const gpus = ref<GpuEntry[]>([]);
const placement = ref<DevicePlacement>(defaultPlacement());
const showAdvanced = ref(false);
const saving = ref(false);

const selectedEntry = computed(() => installed.value.find((m) => m.name === selectedModel.value));
const family = computed(() => selectedEntry.value?.family ?? "");
const tier2 = computed(() => Boolean(selectedEntry.value) && supportsAdvanced(family.value));

// Per-encoder disclosure rows. `null` means "follow the group knob".
const ENCODER_FIELDS: ReadonlyArray<{ field: keyof AdvancedPlacement; label: string }> = [
  { field: "clip_l", label: "CLIP-L" },
  { field: "clip_g", label: "CLIP-G" },
  { field: "t5", label: "T5" },
  { field: "qwen", label: "Qwen" },
];

onMounted(async () => {
  if (models.all.length === 0 && !models.loading) void models.fetch();
  try {
    const snap = await apiJson<ResourceSnapshot>("/api/resources");
    gpus.value = snap.gpus.map((g) => ({ ordinal: g.ordinal, name: g.name }));
  } catch {
    gpus.value = [];
  }
});

// Default to the first installed model once the list arrives.
watch(
  installed,
  (list) => {
    if (!selectedModel.value && list.length > 0) selectedModel.value = list[0]!.name;
  },
  { immediate: true },
);

// Hydrate the editor from the saved default whenever the target changes, so a
// subsequent Save edits the persisted placement instead of clobbering it with
// Auto defaults. Falls back to defaults when nothing is saved (404 → null).
let hydrateToken = 0;
watch(
  selectedModel,
  async (model) => {
    showAdvanced.value = false;
    placement.value = defaultPlacement();
    if (!model) return;
    const token = ++hydrateToken;
    try {
      const saved = await getModelPlacement(model);
      if (token !== hydrateToken) return; // selection changed mid-flight
      if (saved) {
        placement.value = saved;
        if (saved.advanced) showAdvanced.value = true;
      }
    } catch (err) {
      if (token === hydrateToken) toasts.push(String(err), "error");
    }
  },
  { immediate: true },
);

const textEncodersValue = computed(() => refToOption(placement.value.text_encoders));
function onTextEncoders(opt: string) {
  placement.value = setTextEncoders(placement.value, optionToRef(opt));
}

function coreValue(field: "transformer" | "vae"): string {
  return refToOption(placement.value.advanced?.[field] ?? { kind: "auto" });
}
function onCore(field: "transformer" | "vae", opt: string) {
  placement.value = setAdvancedField(placement.value, field, optionToRef(opt));
}

function encoderValue(field: keyof AdvancedPlacement): string {
  const v = placement.value.advanced?.[field];
  return v == null ? "group" : refToOption(v);
}
function onEncoder(field: keyof AdvancedPlacement, opt: string) {
  const ref = opt === "group" ? null : optionToRef(opt);
  placement.value = setAdvancedField(placement.value, field, ref);
}

async function save() {
  if (!selectedModel.value) return;
  saving.value = true;
  try {
    await putModelPlacement(selectedModel.value, placement.value);
    toasts.push(`Saved placement for ${selectedModel.value}`);
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    saving.value = false;
  }
}

async function clear() {
  if (!selectedModel.value) return;
  saving.value = true;
  try {
    await deleteModelPlacement(selectedModel.value);
    placement.value = defaultPlacement();
    showAdvanced.value = false;
    toasts.push(`Cleared placement for ${selectedModel.value}`);
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    saving.value = false;
  }
}
</script>

<template>
  <section class="border-border mb-5 rounded-window border bg-bg p-4" data-test="placement-section">
    <h3 class="text-sm font-semibold text-fg">Device placement</h3>
    <p class="mt-1 text-micro text-fg-dim">
      Pin a model's components to specific GPUs or CPU. Saved as a per-model default in config.toml
      — it applies the next time the model loads.
    </p>

    <div v-if="installed.length === 0" class="mt-3 text-micro text-fg-dim">
      No installed models to configure.
    </div>

    <template v-else>
      <label class="mt-4 block text-micro text-fg-2" for="placement-model">Model</label>
      <select
        id="placement-model"
        v-model="selectedModel"
        class="border-border font-mono text-xs mt-1 h-8 w-full rounded-control border bg-bg-deep px-2 text-fg"
        data-test="placement-model"
      >
        <option v-for="m in installed" :key="m.name" :value="m.name">
          {{ modelDisplayName(m) }}
        </option>
      </select>

      <label class="mt-4 block text-micro text-fg-2" for="placement-te">Text encoders</label>
      <p class="text-micro text-fg-dim">Group knob for T5, CLIP, and Qwen encoders.</p>
      <select
        id="placement-te"
        :value="textEncodersValue"
        class="border-border mt-1 h-8 w-full rounded-control border bg-bg-deep px-2 text-fg"
        data-test="placement-text-encoders"
        @change="onTextEncoders(($event.target as HTMLSelectElement).value)"
      >
        <option value="auto">Auto</option>
        <option value="cpu">CPU</option>
        <option v-for="g in gpus" :key="g.ordinal" :value="`gpu:${g.ordinal}`">
          GPU {{ g.ordinal }} — {{ g.name }}
        </option>
      </select>

      <div v-if="tier2" class="mt-4">
        <button
          type="button"
          class="border-border h-8 w-full rounded-control border px-3 text-left text-sm text-fg-2 hover:text-fg"
          data-test="placement-advanced-toggle"
          :aria-expanded="showAdvanced"
          @click="showAdvanced = !showAdvanced"
        >
          {{ showAdvanced ? "▾" : "▸" }} Advanced — per-component overrides
        </button>

        <div v-if="showAdvanced" class="mt-3 space-y-3" data-test="placement-advanced">
          <div>
            <label class="block text-micro text-fg-2" for="placement-transformer">
              Transformer
            </label>
            <select
              id="placement-transformer"
              :value="coreValue('transformer')"
              class="border-border mt-1 h-8 w-full rounded-control border bg-bg-deep px-2 text-fg"
              @change="onCore('transformer', ($event.target as HTMLSelectElement).value)"
            >
              <option value="auto">Auto</option>
              <option value="cpu">CPU</option>
              <option v-for="g in gpus" :key="g.ordinal" :value="`gpu:${g.ordinal}`">
                GPU {{ g.ordinal }} — {{ g.name }}
              </option>
            </select>
          </div>

          <div>
            <label class="block text-micro text-fg-2" for="placement-vae">VAE</label>
            <select
              id="placement-vae"
              :value="coreValue('vae')"
              class="border-border mt-1 h-8 w-full rounded-control border bg-bg-deep px-2 text-fg"
              @change="onCore('vae', ($event.target as HTMLSelectElement).value)"
            >
              <option value="auto">Auto</option>
              <option value="cpu">CPU</option>
              <option v-for="g in gpus" :key="g.ordinal" :value="`gpu:${g.ordinal}`">
                GPU {{ g.ordinal }} — {{ g.name }}
              </option>
            </select>
          </div>

          <div v-for="enc in ENCODER_FIELDS" :key="enc.field">
            <label class="block text-micro text-fg-2" :for="`placement-${enc.field}`">
              {{ enc.label }}
            </label>
            <select
              :id="`placement-${enc.field}`"
              :value="encoderValue(enc.field)"
              class="border-border mt-1 h-8 w-full rounded-control border bg-bg-deep px-2 text-fg"
              @change="onEncoder(enc.field, ($event.target as HTMLSelectElement).value)"
            >
              <option value="group">Follow text encoders</option>
              <option value="auto">Auto</option>
              <option value="cpu">CPU</option>
              <option v-for="g in gpus" :key="g.ordinal" :value="`gpu:${g.ordinal}`">
                GPU {{ g.ordinal }} — {{ g.name }}
              </option>
            </select>
          </div>
        </div>
      </div>

      <div class="mt-4 flex items-center gap-3">
        <button
          type="button"
          class="h-8 rounded-control bg-accent px-3 text-sm font-semibold text-on-accent hover:brightness-105 active:translate-y-px disabled:opacity-50"
          data-test="placement-save"
          :disabled="saving || !selectedModel"
          @click="save"
        >
          Save as default
        </button>
        <button
          type="button"
          class="border-border h-8 rounded-control border px-3 text-sm text-fg-2 hover:text-fg disabled:opacity-50"
          data-test="placement-clear"
          :disabled="saving || !selectedModel"
          @click="clear"
        >
          Clear
        </button>
      </div>
    </template>
  </section>
</template>
