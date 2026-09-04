<script setup lang="ts">
import { ref, watch } from "vue";
import type { GenerateForm } from "../../lib/generateForm";
import { MAX_LORA_STACK } from "../../lib/capabilities";
import { fetchLoras } from "../../lib/api/loras";
import type { LoraInfo } from "../../lib/api/types";
import { cameraMotionLoraPath } from "@studio/lib/cameraMotion";
import type { HostRoute } from "../../stores/hosts";

const props = defineProps<{ form: GenerateForm; model: string; route: HostRoute | null }>();
const emit = defineEmits<{ (e: "append-word", word: string): void }>();

const pickerOpen = ref(false);
const available = ref<LoraInfo[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
let loadEpoch = 0;

async function openPicker() {
  pickerOpen.value = !pickerOpen.value;
  const epoch = ++loadEpoch;
  if (!pickerOpen.value) {
    loading.value = false;
    return;
  }
  loading.value = true;
  error.value = null;
  const route = props.route;
  if (!route) {
    loading.value = false;
    error.value = "The selected machine is unavailable.";
    available.value = [];
    return;
  }
  try {
    const response = await fetchLoras(props.model, route.target);
    if (
      epoch === loadEpoch &&
      props.route?.hostId === route.hostId &&
      (props.route?.instanceId ?? null) === (route.instanceId ?? null)
    )
      available.value = response;
  } catch (err) {
    if (epoch === loadEpoch) {
      error.value = String(err);
      available.value = [];
    }
  } finally {
    if (epoch === loadEpoch) loading.value = false;
  }
}

function alreadyAdded(path: string): boolean {
  return props.form.loras.some((l) => l.path === path);
}

function addLora(l: LoraInfo) {
  if (!props.route || props.form.loras.length >= MAX_LORA_STACK || alreadyAdded(l.path)) return;
  props.form.loras.push({
    path: l.path,
    name: l.name,
    scale: 1,
    trainedWords: l.trained_words,
    hostId: props.route.hostId,
    hostBaseUrl: props.route.target.baseUrl,
    hostInstanceId: props.route.instanceId ?? null,
  });
  pickerOpen.value = false;
}

function removeLora(index: number) {
  const removed = props.form.loras[index];
  props.form.loras.splice(index, 1);
  if (removed?.path === cameraMotionLoraPath(props.form.cameraControl)) {
    props.form.cameraControl = null;
  }
}

// Close and reset the picker when the model changes — the family may differ.
watch(
  () =>
    [
      props.model,
      props.route?.hostId,
      props.route?.instanceId,
      props.route?.target.baseUrl,
    ] as const,
  () => {
    loadEpoch += 1;
    loading.value = false;
    pickerOpen.value = false;
    available.value = [];
  },
);
</script>

<template>
  <div>
    <div class="mb-2 flex items-center gap-2">
      <span class="font-mono text-micro text-fg-dim whitespace-nowrap">Add-on looks</span>
      <span v-if="route" class="text-micro text-fg-dim">{{ route.label }}</span>
      <div class="border-border h-px flex-1 border-t" />
    </div>

    <div v-for="(lora, i) in form.loras" :key="lora.path" class="mb-3">
      <div class="flex items-center justify-between gap-2">
        <span class="truncate text-sm text-fg" :title="lora.name">{{ lora.name }}</span>
        <button
          type="button"
          class="text-fg-dim hover:text-error"
          title="Remove this add-on look"
          aria-label="Remove this add-on look"
          @click="removeLora(i)"
        >
          ✕
        </button>
      </div>
      <div class="mt-1 flex items-center gap-2">
        <input
          v-model.number="lora.scale"
          type="range"
          min="0"
          max="2"
          step="0.05"
          class="w-full accent-accent"
        />
        <span class="font-mono text-xs w-9 text-right text-fg">{{ lora.scale.toFixed(2) }}</span>
      </div>
      <div v-if="lora.trainedWords.length" class="mt-1 flex flex-wrap gap-1">
        <button
          v-for="word in lora.trainedWords"
          :key="word"
          type="button"
          class="border-border rounded-control border bg-bg-deep px-2 py-0.5 text-micro text-sapphire hover:text-fg"
          title="Insert trigger phrase"
          @click="emit('append-word', word)"
        >
          {{ word }}
        </button>
      </div>
    </div>

    <div>
      <button
        type="button"
        class="text-sm text-sapphire hover:text-fg disabled:opacity-40"
        :disabled="form.loras.length >= MAX_LORA_STACK"
        @click="openPicker"
      >
        + Add a look
      </button>
      <div
        v-if="pickerOpen"
        data-test="lora-picker"
        class="border-border mt-2 max-h-60 w-full overflow-y-auto rounded-window border bg-bg shadow-md"
      >
        <p v-if="loading" class="px-2 py-2 text-micro text-fg-dim">Loading…</p>
        <p v-else-if="error" class="px-2 py-2 text-micro text-error">{{ error }}</p>
        <p v-else-if="available.length === 0" class="px-2 py-2 text-micro text-fg-dim">
          No add-on looks here yet for this family.
        </p>
        <button
          v-for="l in available"
          v-else
          :key="l.path"
          type="button"
          class="flex w-full items-center justify-between px-2 py-1.5 text-left text-sm text-fg-2 hover:bg-bg-deep hover:text-fg disabled:opacity-40"
          :disabled="alreadyAdded(l.path)"
          @click="addLora(l)"
        >
          <span class="truncate">{{ l.name }}</span>
          <span
            v-if="alreadyAdded(l.path)"
            class="font-mono text-micro text-fg-dim whitespace-nowrap ml-2"
            >added</span
          >
        </button>
      </div>
    </div>
  </div>
</template>
