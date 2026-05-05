<script setup lang="ts">
import { ref, watch } from "vue";
import { fetchCatalog } from "../api";
import type { CatalogEntryWire, LoraSelection } from "../types";

const props = defineProps<{
  family: string;
  modelValue: LoraSelection | null;
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: LoraSelection | null): void;
}>();

const loras = ref<CatalogEntryWire[]>([]);
const loading = ref(false);

async function loadLoras(family: string) {
  loras.value = [];
  if (!family) return;
  loading.value = true;
  try {
    const res = await fetchCatalog({ family, kind: "lora", page_size: 200 });
    loras.value = res.entries.filter((e) => e.installed && e.primary_path);
  } catch {
    // silently ignore — picker just stays empty
  } finally {
    loading.value = false;
  }
}

watch(() => props.family, loadLoras, { immediate: true });

function onSelect(event: Event) {
  const path = (event.target as HTMLSelectElement).value;
  if (!path) {
    emit("update:modelValue", null);
  } else {
    emit("update:modelValue", { path, scale: props.modelValue?.scale ?? 1.0 });
  }
}

function onScale(event: Event) {
  if (!props.modelValue) return;
  const scale = Number((event.target as HTMLInputElement).value);
  emit("update:modelValue", { path: props.modelValue.path, scale });
}
</script>

<template>
  <section v-if="loras.length > 0 || modelValue" class="mt-4">
    <label class="text-xs uppercase text-slate-400">LoRA</label>
    <select
      :value="modelValue?.path ?? ''"
      class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
      @change="onSelect"
    >
      <option value="">None</option>
      <option
        v-for="e in loras"
        :key="e.id"
        :value="e.primary_path!"
      >
        {{ e.name }}
      </option>
    </select>
    <div v-if="modelValue" class="mt-2">
      <label class="text-xs text-slate-400"
        >Scale — {{ modelValue.scale.toFixed(2) }}</label
      >
      <input
        type="range"
        min="0"
        max="2"
        step="0.05"
        :value="modelValue.scale"
        class="w-full"
        @input="onScale"
      />
    </div>
  </section>
</template>
