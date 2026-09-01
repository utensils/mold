<script setup lang="ts">
import { computed, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import { remixPrompt, type StreamTarget } from "../api";
import type {
  ExpandContextWire,
  ExpandTask,
  RemixResponseWire,
  RemixSourceKind,
} from "../types";
import {
  defaultRemixDimensions,
  promptSource,
  remixDimensionsForTask,
  validateRemixVariants,
  type RemixDimension,
} from "@studio/lib/promptTransform";

const props = defineProps<{
  open: boolean;
  prompt: string;
  originalPrompt?: string | null;
  family: string;
  task: ExpandTask;
  /** Generation facts frozen with the request (identity, canvas, frames). */
  context?: ExpandContextWire | null;
  style?: string | null;
  target?: StreamTarget;
}>();
const emit = defineEmits<{
  close: [];
  apply: [payload: { prompt: string; response: RemixResponseWire }];
  prepare: [response: RemixResponseWire];
}>();

const sourceKind = ref<"original" | "current">("original");
const dimensions = ref<RemixDimension[]>([]);
const busy = ref(false);
const error = ref<string | null>(null);
const response = ref<RemixResponseWire | null>(null);
const selected = ref<number[]>([]);
const styleLocked = computed(() => Boolean(props.style?.trim()));
const availableDimensions = computed(() =>
  remixDimensionsForTask(props.task, styleLocked.value),
);
watch(
  () => [props.open, props.task, props.style] as const,
  ([open]) => {
    if (!open) return;
    sourceKind.value = props.originalPrompt?.trim() ? "original" : "current";
    dimensions.value = defaultRemixDimensions(props.task, styleLocked.value);
    response.value = null;
    selected.value = [];
    error.value = null;
  },
  { immediate: true },
);

function toggleDimension(dimension: RemixDimension) {
  dimensions.value = dimensions.value.includes(dimension)
    ? dimensions.value.filter((candidate) => candidate !== dimension)
    : [...dimensions.value, dimension];
}

async function remix() {
  const source = promptSource(
    props.prompt,
    props.originalPrompt,
    sourceKind.value,
  );
  if (!source.prompt || dimensions.value.length === 0) return;
  busy.value = true;
  error.value = null;
  try {
    const result = await remixPrompt(
      {
        source_prompt: source.prompt,
        ...(source.rootPrompt ? { root_prompt: source.rootPrompt } : {}),
        source_kind: source.kind as RemixSourceKind,
        model_family: props.family,
        variations: 3,
        task: props.task,
        ...(props.context ? { context: props.context } : {}),
        ...(props.style?.trim() ? { style: props.style.trim() } : {}),
        dimensions: dimensions.value,
      },
      undefined,
      props.target,
    );
    response.value = {
      ...result,
      variants: validateRemixVariants(result.variants, 3),
    };
    selected.value = [0, 1, 2];
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
  }
}

function toggleSelected(index: number) {
  selected.value = selected.value.includes(index)
    ? selected.value.filter((candidate) => candidate !== index)
    : [...selected.value, index].sort((left, right) => left - right);
}

function prepareSelected() {
  if (!response.value || selected.value.length < 2) return;
  emit("prepare", {
    ...response.value,
    variants: selected.value.map((index) => response.value!.variants[index]!),
  });
}
</script>

<template>
  <div v-if="open" class="fixed inset-0 z-40">
    <ModalPanel
      :open="open"
      :width="620"
      label="Remix prompt"
      @close="emit('close')"
    >
      <div class="space-y-4 p-1 text-ink">
        <header>
          <h2 class="text-lg font-semibold">Remix prompt</h2>
          <p class="text-sm text-ink-2">
            Explore controlled alternatives without changing the central idea.
          </p>
        </header>
        <fieldset v-if="originalPrompt?.trim()" class="space-y-2">
          <legend class="text-xs uppercase tracking-wide text-ink-3">
            Source
          </legend>
          <label class="mr-4 text-sm"
            ><input
              v-model="sourceKind"
              type="radio"
              value="original"
              data-test="remix-source-original"
            />
            Original idea</label
          >
          <label class="text-sm"
            ><input
              v-model="sourceKind"
              type="radio"
              value="current"
              data-test="remix-source-current"
            />
            Current prompt</label
          >
        </fieldset>
        <p
          data-test="remix-source-label"
          class="rounded-control bg-bath p-2 text-sm text-ink-2"
        >
          Remixing
          {{
            sourceKind === "original" && originalPrompt
              ? "original idea"
              : "current prompt"
          }}: “{{
            sourceKind === "original" && originalPrompt
              ? originalPrompt
              : prompt
          }}”
        </p>
        <fieldset>
          <legend class="mb-2 text-xs uppercase tracking-wide text-ink-3">
            Vary
          </legend>
          <div class="flex flex-wrap gap-2">
            <button
              v-for="dimension in availableDimensions"
              :key="dimension"
              type="button"
              class="rounded-control border border-edge px-2 py-1 text-sm capitalize"
              :class="
                dimensions.includes(dimension)
                  ? 'bg-safelight text-on-accent'
                  : ''
              "
              :data-test="`remix-dimension-${dimension}`"
              @click="toggleDimension(dimension)"
            >
              {{ dimension }}
            </button>
          </div>
        </fieldset>
        <p v-if="task !== 'text-to-image'" class="text-xs text-ink-3">
          Attached media remains authoritative; remixes vary only compatible
          creative direction.
        </p>
        <button
          type="button"
          data-test="remix-generate"
          class="rounded-control bg-safelight px-4 py-2 font-semibold text-on-accent disabled:opacity-50"
          :disabled="busy || !prompt.trim() || dimensions.length === 0"
          @click="remix"
        >
          {{ busy ? "Remixing…" : response ? "Re-remix" : "Create 3 remixes" }}
        </button>
        <p v-if="error" role="alert" class="text-sm text-stop">{{ error }}</p>
        <ol v-if="response" class="space-y-2">
          <li
            v-for="(variant, index) in response.variants"
            :key="index"
            class="rounded-control border border-edge bg-bath p-2"
          >
            <label class="mb-2 flex items-center gap-2 text-xs text-ink-3">
              <input
                type="checkbox"
                :checked="selected.includes(index)"
                :data-test="`select-remix-${index}`"
                @change="toggleSelected(index)"
              />
              Include in Batch
            </label>
            <p class="text-sm">{{ variant.prompt }}</p>
            <button
              type="button"
              :data-test="`apply-remix-${index}`"
              class="mt-2 rounded-control border border-edge px-2 py-1 text-sm"
              @click="emit('apply', { prompt: variant.prompt, response })"
            >
              Apply
            </button>
          </li>
        </ol>
        <button
          v-if="response"
          type="button"
          data-test="prepare-remix-batch"
          class="rounded-control bg-safelight px-4 py-2 font-semibold text-on-accent"
          :disabled="selected.length < 2"
          @click="prepareSelected"
        >
          Use {{ selected.length }} as Batch
        </button>
      </div>
    </ModalPanel>
  </div>
</template>
