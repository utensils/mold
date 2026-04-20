<script setup lang="ts">
import { computed, ref } from "vue";
import type { AdvancedPlacement, DevicePlacement, DeviceRef } from "../types";
import { usePlacement } from "../composables/usePlacement";

interface GpuEntry {
  ordinal: number;
  name: string;
}

const props = defineProps<{
  modelValue: DevicePlacement | null;
  family: string;
  model: string;
  gpus: GpuEntry[];
}>();

const emit = defineEmits<{
  (e: "update:modelValue", v: DevicePlacement | null): void;
}>();

const { supportsAdvanced } = usePlacement();

const tier2 = computed(() => supportsAdvanced(props.family));
const advancedOpen = ref(false);

const tier1Value = computed<DeviceRef>(
  () => props.modelValue?.text_encoders ?? { kind: "auto" },
);

function refToOption(r: DeviceRef): string {
  if (r.kind === "auto") return "auto";
  if (r.kind === "cpu") return "cpu";
  return `gpu:${r.ordinal}`;
}

function optionToRef(opt: string): DeviceRef {
  if (opt === "auto") return { kind: "auto" };
  if (opt === "cpu") return { kind: "cpu" };
  const m = /^gpu:(\d+)$/.exec(opt);
  return m ? { kind: "gpu", ordinal: Number(m[1]) } : { kind: "auto" };
}

function emitTier1(opt: string) {
  const next: DevicePlacement = {
    text_encoders: optionToRef(opt),
    advanced: props.modelValue?.advanced ?? null,
  };
  emit("update:modelValue", next);
}

function emitAdvanced<K extends keyof AdvancedPlacement>(
  field: K,
  opt: string,
) {
  const r = optionToRef(opt);
  const current: AdvancedPlacement = props.modelValue?.advanced ?? {
    transformer: { kind: "auto" },
    vae: { kind: "auto" },
    clip_l: null,
    clip_g: null,
    t5: null,
    qwen: null,
  };
  const nextAdv: AdvancedPlacement = { ...current };
  if (field === "transformer" || field === "vae") {
    nextAdv[field] = r;
  } else {
    nextAdv[field] = r;
  }
  emit("update:modelValue", {
    text_encoders: props.modelValue?.text_encoders ?? { kind: "auto" },
    advanced: nextAdv,
  });
}

async function saveAsDefault() {
  if (!props.modelValue) return;
  const encoded = encodeURIComponent(props.model);
  await fetch(`/api/config/model/${encoded}/placement`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(props.modelValue),
  });
}

const encoderRows = computed(() => {
  switch (props.family) {
    case "flux":
      return ["t5", "clip_l"] as const;
    case "flux2":
    case "flux.2":
    case "flux2-klein":
    case "z-image":
      return ["qwen"] as const;
    case "qwen-image":
    case "qwen_image":
      return ["qwen"] as const;
    case "sd3":
    case "sd3.5":
    case "stable-diffusion-3":
    case "stable-diffusion-3.5":
      return ["t5", "clip_l", "clip_g"] as const;
    default:
      return [] as const;
  }
});

function advancedValue(field: keyof AdvancedPlacement): string {
  const adv = props.modelValue?.advanced;
  if (!adv) return "auto";
  const v = adv[field];
  if (v === null || v === undefined) return "auto";
  return refToOption(v as DeviceRef);
}

const isDirty = computed(() => props.modelValue !== null);
</script>

<template>
  <section class="glass flex flex-col gap-2 rounded-2xl p-3 text-sm">
    <header class="flex items-center justify-between">
      <span class="font-medium text-slate-200">Device placement</span>
      <button
        v-if="isDirty"
        type="button"
        data-test="save-default"
        class="text-xs text-brand-400 hover:underline"
        @click="saveAsDefault"
      >
        Save as default
      </button>
    </header>

    <div v-if="gpus.length > 0" class="flex items-center gap-2">
      <label class="text-slate-400">Text encoders</label>
      <select
        data-test="tier1-select"
        :value="refToOption(tier1Value)"
        class="rounded bg-slate-900 px-2 py-1 text-slate-100"
        @change="emitTier1(($event.target as HTMLSelectElement).value)"
      >
        <option value="auto">Auto</option>
        <option value="cpu">CPU</option>
        <option v-for="g in gpus" :key="g.ordinal" :value="`gpu:${g.ordinal}`">
          GPU {{ g.ordinal }} ({{ g.name }})
        </option>
      </select>
    </div>

    <div class="flex items-center gap-2">
      <button
        type="button"
        data-test="advanced-toggle"
        class="text-xs text-slate-400 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
        :disabled="!tier2"
        :title="
          tier2
            ? undefined
            : `Advanced placement is not yet available for ${family} — Tier 1 controls all encoders as a group.`
        "
        @click="tier2 && (advancedOpen = !advancedOpen)"
      >
        {{ advancedOpen ? "\u25BE" : "\u25B8" }} Advanced
      </button>
    </div>

    <div v-if="tier2 && advancedOpen" class="flex flex-col gap-1 pl-4">
      <div class="flex items-center gap-2">
        <label class="w-24 text-slate-400">Transformer</label>
        <select
          :value="advancedValue('transformer')"
          class="rounded bg-slate-900 px-2 py-1 text-slate-100"
          @change="
            emitAdvanced(
              'transformer',
              ($event.target as HTMLSelectElement).value,
            )
          "
        >
          <option value="auto">Auto</option>
          <option value="cpu">CPU</option>
          <option
            v-for="g in gpus"
            :key="g.ordinal"
            :value="`gpu:${g.ordinal}`"
          >
            GPU {{ g.ordinal }}
          </option>
        </select>
      </div>

      <div class="flex items-center gap-2">
        <label class="w-24 text-slate-400">VAE</label>
        <select
          :value="advancedValue('vae')"
          class="rounded bg-slate-900 px-2 py-1 text-slate-100"
          @change="
            emitAdvanced('vae', ($event.target as HTMLSelectElement).value)
          "
        >
          <option value="auto">Auto</option>
          <option value="cpu">CPU</option>
          <option
            v-for="g in gpus"
            :key="g.ordinal"
            :value="`gpu:${g.ordinal}`"
          >
            GPU {{ g.ordinal }}
          </option>
        </select>
      </div>

      <div
        v-for="field in encoderRows"
        :key="field"
        class="flex items-center gap-2"
      >
        <label class="w-24 text-slate-400">{{ field }}</label>
        <select
          :value="advancedValue(field)"
          class="rounded bg-slate-900 px-2 py-1 text-slate-100"
          @change="
            emitAdvanced(field, ($event.target as HTMLSelectElement).value)
          "
        >
          <option value="auto">Auto (follow group)</option>
          <option value="cpu">CPU</option>
          <option
            v-for="g in gpus"
            :key="g.ordinal"
            :value="`gpu:${g.ordinal}`"
          >
            GPU {{ g.ordinal }}
          </option>
        </select>
      </div>
    </div>
  </section>
</template>
