<script setup lang="ts">
import { computed, ref } from "vue";
import type { GenerateFormState } from "../types";
import type { ModelInfoExtended } from "../types";
import { modelDisplayNameForId } from "../lib/modelName";
import { requestText } from "../lib/toasts";
import {
  deleteGenerationTemplate,
  loadGenerationTemplates,
  renameGenerationTemplate,
  saveGenerationTemplate,
  sortGenerationTemplates,
  type GenerationTemplate,
  type GenerationTemplateSort,
} from "../lib/generationTemplates";

defineOptions({ name: "GenerationTemplatesPanel" });

const props = defineProps<{
  modelValue: GenerateFormState;
  models?: ModelInfoExtended[];
}>();
const modelLabel = (name: string) =>
  modelDisplayNameForId(name, props.models ?? []);
const emit = defineEmits<{
  (e: "update:modelValue", value: GenerateFormState): void;
}>();

const nameDraft = ref("");
const search = ref("");
const sort = ref<GenerationTemplateSort>("updated-desc");
const templates = ref<GenerationTemplate[]>(
  loadGenerationTemplates(sort.value),
);

const visibleTemplates = computed(() => {
  const q = search.value.trim().toLowerCase();
  const sorted = sortGenerationTemplates(templates.value, sort.value);
  if (!q) return sorted;
  return sorted.filter((template) =>
    [
      template.name,
      template.form.model,
      template.form.prompt,
      template.form.negativePrompt,
      ...template.form.loras.map((lora) => lora.path),
      ...template.mediaReferences.map((ref) => ref.filename),
    ]
      .join(" ")
      .toLowerCase()
      .includes(q),
  );
});

function refreshTemplates() {
  templates.value = loadGenerationTemplates(sort.value);
}

function saveCurrent() {
  const fallback =
    props.modelValue.prompt.trim().slice(0, 48) ||
    props.modelValue.model ||
    "Untitled template";
  saveGenerationTemplate(nameDraft.value || fallback, props.modelValue);
  nameDraft.value = "";
  refreshTemplates();
}

function loadTemplate(template: GenerationTemplate) {
  emit("update:modelValue", template.form);
}

async function renameTemplate(template: GenerationTemplate) {
  const next = await requestText({
    title: "Rename template",
    label: "Template name",
    initial: template.name,
  });
  if (next === null) return;
  renameGenerationTemplate(template.id, next);
  refreshTemplates();
}

function deleteTemplate(template: GenerationTemplate) {
  deleteGenerationTemplate(template.id);
  refreshTemplates();
}
</script>

<template>
  <section
    class="mt-4 rounded-lg border border-edge bg-bath/30 p-2"
    data-test="generation-templates"
  >
    <div class="flex items-center justify-between gap-2">
      <label class="text-xs uppercase text-ink-3">Templates</label>
      <span class="text-[11px] text-ink-3">
        {{ templates.length }}
      </span>
    </div>

    <div class="mt-2 grid grid-cols-[1fr_auto] gap-2">
      <input
        v-model="nameDraft"
        type="text"
        class="min-w-0 rounded-lg bg-bench/60 px-2 py-1 text-xs text-rebate placeholder:text-ink-3"
        placeholder="Template name"
        data-test="template-name"
      />
      <button
        type="button"
        class="rounded-md bg-surface px-2 py-1 text-xs text-ink-2 hover:bg-surface"
        data-test="template-save"
        @click="saveCurrent"
      >
        Save
      </button>
    </div>

    <div class="mt-2 grid grid-cols-[1fr_auto] gap-2">
      <input
        v-model="search"
        type="search"
        class="min-w-0 rounded-lg bg-bench/60 px-2 py-1 text-xs text-rebate placeholder:text-ink-3"
        placeholder="Search templates"
        data-test="template-search"
      />
      <select
        v-model="sort"
        class="rounded-lg bg-bench/60 px-2 py-1 text-xs text-rebate"
        data-test="template-sort"
        @change="refreshTemplates"
      >
        <option value="updated-desc">Newest</option>
        <option value="updated-asc">Oldest</option>
        <option value="name-asc">A-Z</option>
        <option value="name-desc">Z-A</option>
      </select>
    </div>

    <div class="mt-2 max-h-44 overflow-y-auto pr-1">
      <div
        v-for="template in visibleTemplates"
        :key="template.id"
        class="grid grid-cols-[1fr_auto_auto_auto] items-center gap-1 border-t border-edge/70 py-1 first:border-t-0"
      >
        <button
          type="button"
          class="min-w-0 truncate text-left text-xs text-ink-2 hover:text-rebate"
          data-test="template-load"
          @click="loadTemplate(template)"
        >
          <span data-test="template-row-name">{{ template.name }}</span>
          <span class="ml-1 text-ink-3">
            {{
              template.form.model ? modelLabel(template.form.model) : "no model"
            }}
          </span>
        </button>
        <button
          type="button"
          class="rounded px-1.5 py-0.5 text-[11px] text-ink-3 hover:bg-surface hover:text-rebate"
          data-test="template-rename"
          @click="renameTemplate(template)"
        >
          Rename
        </button>
        <button
          type="button"
          class="rounded px-1.5 py-0.5 text-[11px] text-ink-3 hover:bg-surface hover:text-rebate"
          data-test="template-delete"
          @click="deleteTemplate(template)"
        >
          Delete
        </button>
      </div>
      <p v-if="visibleTemplates.length === 0" class="py-2 text-xs text-ink-3">
        No templates
      </p>
    </div>
  </section>
</template>
