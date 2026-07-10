<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { apiJson } from "../../lib/api/client";
import { deleteModelPlacement, putModelPlacement } from "../../lib/api/placement";
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

// Reset the in-progress edit whenever the target model changes.
watch(selectedModel, () => {
  placement.value = defaultPlacement();
  showAdvanced.value = false;
});

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
  <section
    class="border-edge mb-5 rounded-chrome border bg-bench p-4"
    data-test="placement-section"
  >
    <h3 class="text-body font-semibold text-ink">Device placement</h3>
    <p class="mt-1 text-caption text-ink-3">
      Pin a model's components to specific GPUs or CPU. Saved as a per-model default in config.toml
      — it applies the next time the model loads.
    </p>

    <div v-if="installed.length === 0" class="mt-3 text-caption text-ink-3">
      No installed models to configure.
    </div>

    <template v-else>
      <label class="mt-4 block text-caption text-ink-2" for="placement-model">Model</label>
      <select
        id="placement-model"
        v-model="selectedModel"
        class="border-edge data-mono mt-1 h-8 w-full rounded-control border bg-bath px-2 text-ink"
        data-test="placement-model"
      >
        <option v-for="m in installed" :key="m.name" :value="m.name">{{ m.name }}</option>
      </select>

      <label class="mt-4 block text-caption text-ink-2" for="placement-te">Text encoders</label>
      <p class="text-caption text-ink-3">Group knob for T5, CLIP, and Qwen encoders.</p>
      <select
        id="placement-te"
        :value="textEncodersValue"
        class="border-edge mt-1 h-8 w-full rounded-control border bg-bath px-2 text-ink"
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
          class="border-edge h-8 w-full rounded-control border px-3 text-left text-body text-ink-2 hover:text-ink"
          data-test="placement-advanced-toggle"
          :aria-expanded="showAdvanced"
          @click="showAdvanced = !showAdvanced"
        >
          {{ showAdvanced ? "▾" : "▸" }} Advanced — per-component overrides
        </button>

        <div v-if="showAdvanced" class="mt-3 space-y-3" data-test="placement-advanced">
          <div>
            <label class="block text-caption text-ink-2" for="placement-transformer">
              Transformer
            </label>
            <select
              id="placement-transformer"
              :value="coreValue('transformer')"
              class="border-edge mt-1 h-8 w-full rounded-control border bg-bath px-2 text-ink"
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
            <label class="block text-caption text-ink-2" for="placement-vae">VAE</label>
            <select
              id="placement-vae"
              :value="coreValue('vae')"
              class="border-edge mt-1 h-8 w-full rounded-control border bg-bath px-2 text-ink"
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
            <label class="block text-caption text-ink-2" :for="`placement-${enc.field}`">
              {{ enc.label }}
            </label>
            <select
              :id="`placement-${enc.field}`"
              :value="encoderValue(enc.field)"
              class="border-edge mt-1 h-8 w-full rounded-control border bg-bath px-2 text-ink"
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
          class="h-8 rounded-control bg-safelight px-3 text-body font-semibold text-[#141110] hover:brightness-105 active:translate-y-px disabled:opacity-50"
          data-test="placement-save"
          :disabled="saving || !selectedModel"
          @click="save"
        >
          Save as default
        </button>
        <button
          type="button"
          class="border-edge h-8 rounded-control border px-3 text-body text-ink-2 hover:text-ink disabled:opacity-50"
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
