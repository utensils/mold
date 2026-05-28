<script setup lang="ts">
import { computed } from "vue";
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
  component?: string;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", v: DevicePlacement | null): void;
}>();

const { supportsAdvanced } = usePlacement();

const tier2 = computed(() => supportsAdvanced(props.family));

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

function componentToAdvancedField(
  component: string | undefined,
): keyof AdvancedPlacement | null {
  const key = (component ?? "").toLowerCase().replace(/[-.\s]/g, "_");
  if (key.includes("transformer")) return "transformer";
  if (key.includes("vae")) return "vae";
  if (key.includes("clip_g") || key.includes("clip_2")) return "clip_g";
  if (key === "clip_l" || key.includes("clip_l")) return "clip_l";
  if (key.includes("clip")) return "clip_l";
  if (key === "t5" || key.includes("t5")) return "t5";
  if (key === "qwen" || key.includes("qwen")) return "qwen";
  if (key === "text_encoder" || key.includes("text_encoder")) {
    switch (props.family) {
      case "flux2":
      case "flux.2":
      case "flux2-klein":
      case "z-image":
      case "qwen-image":
      case "qwen_image":
        return "qwen";
      default:
        return "t5";
    }
  }
  return null;
}

const componentField = computed(() =>
  componentToAdvancedField(props.component),
);

function emitComponent(opt: string) {
  const field = componentField.value;
  if (!field) return;
  emitAdvanced(field, opt);
}

function advancedValue(field: keyof AdvancedPlacement): string {
  const adv = props.modelValue?.advanced;
  if (!adv) return "auto";
  const v = adv[field];
  if (v === null || v === undefined) return "auto";
  return refToOption(v as DeviceRef);
}
</script>

<template>
  <section
    v-if="component"
    class="min-w-[7rem]"
    data-test="component-placement"
  >
    <select
      v-if="componentField && tier2"
      :value="advancedValue(componentField)"
      class="w-full rounded bg-slate-950/70 px-2 py-1 text-[11px] text-slate-200"
      data-test="component-placement-select"
      aria-label="Component device placement"
      @change="emitComponent(($event.target as HTMLSelectElement).value)"
    >
      <option value="auto">Auto</option>
      <option value="cpu">CPU</option>
      <option v-for="g in gpus" :key="g.ordinal" :value="`gpu:${g.ordinal}`">
        GPU {{ g.ordinal }}
      </option>
    </select>
  </section>
</template>
