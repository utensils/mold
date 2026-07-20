<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import type { ApiTarget } from "../lib/api/client";
import { fetchLoras } from "../lib/api/loras";
import type { LoraInfo } from "../lib/api/types";
import { MAX_LORA_STACK, generationCapabilitiesForFamily } from "../lib/capabilities";
import type { GenerateForm } from "../lib/generateForm";

const props = defineProps<{
  form: GenerateForm;
  target: ApiTarget;
}>();

const emit = defineEmits<{ (event: "append-word", word: string): void }>();

const open = ref(props.form.loras.length > 0);
const available = ref<LoraInfo[]>([]);
const loading = ref(false);
const error = ref("");
let loadToken = 0;

const supported = computed(() => generationCapabilitiesForFamily(props.form.family).supportsLora);
const summary = computed(() =>
  props.form.loras.length === 0 ? "Optional" : `${props.form.loras.length} active`,
);

function alreadyAdded(path: string): boolean {
  return props.form.loras.some((lora) => lora.path === path);
}

async function load(): Promise<void> {
  if (!supported.value || !props.form.model) return;
  const token = ++loadToken;
  loading.value = true;
  error.value = "";
  try {
    const response = await fetchLoras(props.form.model, props.target);
    if (token === loadToken) available.value = response;
  } catch (cause) {
    if (token === loadToken) {
      available.value = [];
      error.value = cause instanceof Error ? cause.message : String(cause);
    }
  } finally {
    if (token === loadToken) loading.value = false;
  }
}

function toggle(): void {
  open.value = !open.value;
  if (open.value && available.value.length === 0 && !loading.value) void load();
}

function add(lora: LoraInfo): void {
  if (alreadyAdded(lora.path) || props.form.loras.length >= MAX_LORA_STACK) return;
  props.form.loras.push({
    path: lora.path,
    name: lora.name,
    scale: 1,
    trainedWords: [...lora.trained_words],
  });
}

function remove(index: number): void {
  props.form.loras.splice(index, 1);
}

watch(
  () => [props.form.model, props.target.baseUrl, props.target.apiKey] as const,
  () => {
    loadToken += 1;
    loading.value = false;
    available.value = [];
    error.value = "";
    if (open.value) void load();
  },
);

onMounted(() => {
  if (open.value) void load();
});
</script>

<template>
  <section v-if="supported" class="mobile-generate-disclosure mobile-lora-controls">
    <button
      type="button"
      class="mobile-disclosure-button"
      data-test="mobile-lora-disclosure"
      :aria-expanded="open"
      aria-controls="mobile-lora-panel"
      @click="toggle"
    >
      <span>LoRAs</span>
      <span class="mobile-disclosure-summary">{{ summary }}</span>
      <span aria-hidden="true">{{ open ? "−" : "+" }}</span>
    </button>

    <div v-if="open" id="mobile-lora-panel" class="mobile-disclosure-panel">
      <div
        v-for="(lora, index) in form.loras"
        :key="lora.path"
        class="mobile-lora-row"
        data-test="mobile-lora-row"
      >
        <div class="mobile-lora-row-head">
          <strong>{{ lora.name }}</strong>
          <button
            type="button"
            class="mobile-inline-danger"
            :data-test="`mobile-lora-remove-${index}`"
            :aria-label="`Remove ${lora.name}`"
            @click="remove(index)"
          >
            Remove
          </button>
        </div>
        <label class="mobile-range-field">
          <span
            >Strength <strong>{{ lora.scale.toFixed(2) }}</strong></span
          >
          <input
            v-model.number="lora.scale"
            type="range"
            min="0"
            max="2"
            step="0.05"
            :data-test="`mobile-lora-scale-${index}`"
          />
        </label>
        <div v-if="lora.trainedWords.length" class="mobile-token-list">
          <button
            v-for="(word, wordIndex) in lora.trainedWords"
            :key="word"
            type="button"
            :data-test="`mobile-lora-trigger-${index}-${wordIndex}`"
            @click="emit('append-word', word)"
          >
            {{ word }}
          </button>
        </div>
      </div>

      <p v-if="loading" class="mobile-helper-text" role="status">Loading LoRAs…</p>
      <p v-else-if="error" class="error-text mobile-helper-text" data-test="mobile-lora-error">
        {{ error }}
      </p>
      <p v-else-if="available.length === 0" class="mobile-helper-text">
        No compatible LoRAs are installed on this host.
      </p>
      <div v-else class="mobile-option-list" aria-label="Available LoRAs">
        <button
          v-for="lora in available"
          :key="lora.path"
          type="button"
          data-test="mobile-lora-option"
          :disabled="alreadyAdded(lora.path) || form.loras.length >= MAX_LORA_STACK"
          @click="add(lora)"
        >
          <span>{{ lora.name }}</span>
          <span>{{ alreadyAdded(lora.path) ? "Added" : "Add" }}</span>
        </button>
      </div>
      <p class="mobile-helper-text">Up to {{ MAX_LORA_STACK }} adapters per print.</p>
    </div>
  </section>
</template>
