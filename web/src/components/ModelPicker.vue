<script setup lang="ts">
import { computed, ref } from "vue";
import type { ModelInfoExtended } from "../types";
import { VIDEO_FAMILIES } from "../types";

const props = defineProps<{
  models: ModelInfoExtended[];
  modelValue: string;
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: string): void;
  (e: "select", model: ModelInfoExtended): void;
}>();

const SHOW_ALL_KEY = "mold.generate.showAllModels";
const showAll = ref(localStorage.getItem(SHOW_ALL_KEY) === "true");

function setShowAll(v: boolean) {
  showAll.value = v;
  try {
    localStorage.setItem(SHOW_ALL_KEY, String(v));
  } catch {
    /* ignore */
  }
}

const visibleModels = computed(() =>
  props.models.filter((m) => (showAll.value ? true : m.downloaded)),
);

const imageModels = computed(() =>
  visibleModels.value.filter((m) => !VIDEO_FAMILIES.includes(m.family)),
);
const videoModels = computed(() =>
  visibleModels.value.filter((m) => VIDEO_FAMILIES.includes(m.family)),
);

function onPick(model: ModelInfoExtended) {
  if (!model.downloaded) return;
  emit("update:modelValue", model.name);
  emit("select", model);
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <label
      class="flex items-center justify-between text-xs uppercase text-slate-400"
    >
      <span>Model</span>
      <span class="flex items-center gap-2 normal-case">
        <input
          id="mold-show-all-models"
          type="checkbox"
          :checked="showAll"
          @change="setShowAll(($event.target as HTMLInputElement).checked)"
        />
        <label for="mold-show-all-models">Show all</label>
      </span>
    </label>

    <div class="flex max-h-80 flex-col gap-3 overflow-y-auto pr-1">
      <div>
        <div class="text-xs font-medium text-slate-500">Images</div>
        <ul class="mt-1 flex flex-col gap-1">
          <li v-for="m in imageModels" :key="m.name">
            <button
              type="button"
              class="w-full rounded-xl px-3 py-2 text-left text-sm"
              :class="[
                modelValue === m.name
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200',
                !m.downloaded && 'cursor-not-allowed opacity-50',
              ]"
              :disabled="!m.downloaded"
              :title="
                m.downloaded
                  ? m.description
                  : 'Not downloaded — run mold pull ' + m.name + ' on the host.'
              "
              @click="onPick(m)"
            >
              <div class="flex items-center justify-between gap-2">
                <span>{{ m.name }}</span>
                <span class="text-xs text-slate-400">{{ m.family }}</span>
              </div>
              <div class="text-xs text-slate-400">{{ m.description }}</div>
            </button>
          </li>
        </ul>
      </div>

      <div v-if="videoModels.length">
        <div class="flex items-center gap-2 text-xs font-medium text-slate-500">
          <span>🎬</span><span>Video</span>
        </div>
        <ul class="mt-1 flex flex-col gap-1">
          <li v-for="m in videoModels" :key="m.name">
            <button
              type="button"
              class="w-full rounded-xl px-3 py-2 text-left text-sm"
              :class="[
                modelValue === m.name
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200',
                !m.downloaded && 'cursor-not-allowed opacity-50',
              ]"
              :disabled="!m.downloaded"
              :title="
                m.downloaded
                  ? m.description
                  : 'Not downloaded — run mold pull ' + m.name + ' on the host.'
              "
              @click="onPick(m)"
            >
              <div class="flex items-center justify-between gap-2">
                <span
                  >{{ m.name }}
                  <span class="italic text-xs text-slate-400">video</span></span
                >
                <span class="text-xs text-slate-400">{{ m.family }}</span>
              </div>
              <div class="text-xs text-slate-400">{{ m.description }}</div>
            </button>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>
