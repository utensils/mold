<script setup lang="ts">
import { ref, watch } from "vue";
import Icon from "@ui/components/Icon.vue";
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
    <div class="ms-loras__head">
      <span class="ms-group-label">Add-on looks</span>
      <span v-if="route" class="ms-loras__route">{{ route.label }}</span>
      <span class="ms-loras__spacer" />
      <button
        type="button"
        class="ms-loras__add"
        :disabled="form.loras.length >= MAX_LORA_STACK"
        @click="openPicker"
      >
        Add
      </button>
    </div>

    <div v-for="(lora, i) in form.loras" :key="lora.path" class="ms-lora">
      <div class="ms-lora__row">
        <div class="ms-lora__body">
          <span class="ms-lora__name" :title="lora.name">{{ lora.name }}</span>
          <input
            v-model.number="lora.scale"
            type="range"
            min="0"
            max="2"
            step="0.05"
            class="ms-lora__meter"
            :style="{ '--lora-fill': `${(lora.scale / 2) * 100}%` }"
            :aria-label="`Strength of ${lora.name}`"
          />
        </div>
        <span class="ms-lora__weight">{{ lora.scale.toFixed(2) }}</span>
        <button
          type="button"
          class="ms-lora__remove"
          title="Remove this add-on look"
          aria-label="Remove this add-on look"
          @click="removeLora(i)"
        >
          <Icon name="close" :size="12" />
        </button>
      </div>
      <div v-if="lora.trainedWords.length" class="ms-lora__words">
        <button
          v-for="word in lora.trainedWords"
          :key="word"
          type="button"
          class="ms-lora__word"
          title="Insert trigger phrase"
          @click="emit('append-word', word)"
        >
          {{ word }}
        </button>
      </div>
    </div>

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
</template>

<style scoped>
.ms-loras__head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.ms-loras__route {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
.ms-loras__spacer {
  flex: 1;
}
.ms-loras__add {
  border: 0;
  background: transparent;
  color: var(--mold-blue);
  font-size: var(--mold-fs-micro);
  cursor: pointer;
}
.ms-loras__add:disabled {
  color: var(--mold-text-dim);
  cursor: default;
}
.ms-lora {
  margin-bottom: 6px;
  padding: 8px 9px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
}
.ms-lora__row {
  display: flex;
  align-items: center;
  gap: 9px;
}
.ms-lora__body {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  gap: 4px;
}
.ms-lora__name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--mold-fs-xs);
  font-weight: 500;
  color: var(--mold-text);
}
/* The meter IS the control: a 4px square track whose accent fill is the
   weight, so nothing has to be kept in sync with a second element. */
.ms-lora__meter {
  width: 100%;
  height: 4px;
  appearance: none;
  background: linear-gradient(
    to right,
    var(--mold-blue) var(--lora-fill),
    var(--mold-surface) var(--lora-fill)
  );
  cursor: ew-resize;
}
.ms-lora__meter::-webkit-slider-thumb {
  appearance: none;
  width: 10px;
  height: 10px;
  border-radius: var(--mold-radius-1);
  background: var(--mold-text);
}
.ms-lora__weight {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
.ms-lora__remove {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border: 0;
  border-radius: var(--mold-radius-1);
  background: transparent;
  color: var(--mold-text-dim);
  cursor: pointer;
}
.ms-lora__remove:hover {
  color: var(--mold-error);
}
.ms-lora__words {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 7px;
}
.ms-lora__word {
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  padding: 1px 7px;
  font-size: var(--mold-fs-micro);
  color: var(--mold-sapphire);
  cursor: pointer;
}
.ms-lora__word:hover {
  color: var(--mold-text);
}
</style>
