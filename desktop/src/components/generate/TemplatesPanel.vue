<script setup lang="ts">
import { computed, ref } from "vue";
import type { GenerateForm } from "../../lib/generateForm";
import { modelDisplayNameForId, type DisplayableModel } from "../../lib/models";
import {
  deleteGenerationTemplateWithMedia,
  fallbackTemplateName,
  loadGenerationTemplates,
  renameGenerationTemplate,
  saveGenerationTemplateWithMedia,
  sortGenerationTemplates,
  type GenerationTemplate,
  type GenerationTemplateSort,
} from "../../lib/generationTemplates";

const props = defineProps<{
  form: GenerateForm;
  models?: DisplayableModel[] | undefined;
}>();
const modelLabel = (name: string) => modelDisplayNameForId(name, props.models ?? []);
const emit = defineEmits<{ (e: "load", template: GenerationTemplate): void }>();

const nameDraft = ref("");
const search = ref("");
const sort = ref<GenerationTemplateSort>("updated-desc");
const templates = ref<GenerationTemplate[]>(loadGenerationTemplates(sort.value));

// Inline rename + two-click delete confirm state, keyed by template id.
const renamingId = ref<string | null>(null);
const renameDraft = ref("");
const confirmingDeleteId = ref<string | null>(null);
const saving = ref(false);
const error = ref("");

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
    ]
      .join(" ")
      .toLowerCase()
      .includes(q),
  );
});

function refresh() {
  templates.value = loadGenerationTemplates(sort.value);
}

async function saveCurrent() {
  if (saving.value) return;
  saving.value = true;
  error.value = "";
  try {
    await saveGenerationTemplateWithMedia(
      nameDraft.value || fallbackTemplateName(props.form),
      props.form,
    );
    nameDraft.value = "";
    refresh();
  } catch (cause) {
    error.value =
      cause instanceof Error ? cause.message : "Couldn’t save the template and its source image.";
  } finally {
    saving.value = false;
  }
}

function loadTemplate(template: GenerationTemplate) {
  emit("load", template);
}

function startRename(template: GenerationTemplate) {
  confirmingDeleteId.value = null;
  renamingId.value = template.id;
  renameDraft.value = template.name;
}
function commitRename(id: string) {
  if (renamingId.value !== id) return;
  renameGenerationTemplate(id, renameDraft.value);
  renamingId.value = null;
  refresh();
}
function cancelRename() {
  renamingId.value = null;
}

async function requestDelete(template: GenerationTemplate) {
  if (confirmingDeleteId.value === template.id) {
    const deletion = deleteGenerationTemplateWithMedia(template);
    confirmingDeleteId.value = null;
    refresh();
    await deletion;
  } else {
    renamingId.value = null;
    confirmingDeleteId.value = template.id;
  }
}
</script>

<template>
  <div data-test="templates-panel">
    <div class="mt-5 mb-2 flex items-center gap-2">
      <span class="edge-code">Templates</span>
      <span class="data-mono text-caption text-ink-3">{{ templates.length }}</span>
      <div class="border-edge h-px flex-1 border-t" />
    </div>

    <!-- Save current form -->
    <div class="grid grid-cols-[1fr_auto] gap-2">
      <input
        v-model="nameDraft"
        data-selectable
        type="text"
        placeholder="Template name"
        data-test="template-name"
        class="border-edge h-8 min-w-0 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3"
        @keydown.enter="saveCurrent"
      />
      <button
        type="button"
        data-test="template-save"
        :disabled="saving"
        class="h-8 rounded-control bg-safelight px-3 text-body font-semibold text-on-accent hover:brightness-105 active:translate-y-px"
        @click="saveCurrent"
      >
        {{ saving ? "Saving…" : "Save" }}
      </button>
    </div>
    <p v-if="error" class="mt-1 text-caption text-stop" role="alert">{{ error }}</p>

    <!-- Search + sort -->
    <div class="mt-2 grid grid-cols-[1fr_auto] gap-2">
      <input
        v-model="search"
        data-selectable
        type="search"
        placeholder="Search templates"
        data-test="template-search"
        class="border-edge h-8 min-w-0 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3"
      />
      <select
        v-model="sort"
        data-test="template-sort"
        aria-label="Sort templates"
        class="border-edge h-8 rounded-control border bg-bath px-1.5 text-body text-ink"
        @change="refresh"
      >
        <option value="updated-desc">Newest</option>
        <option value="updated-asc">Oldest</option>
        <option value="name-asc">A–Z</option>
        <option value="name-desc">Z–A</option>
      </select>
    </div>

    <!-- List -->
    <div class="mt-2 max-h-52 overflow-y-auto pr-1">
      <div
        v-for="template in visibleTemplates"
        :key="template.id"
        class="border-edge grid grid-cols-[1fr_auto_auto] items-center gap-1 border-t py-1 first:border-t-0"
        data-test="template-row"
      >
        <!-- Inline rename -->
        <input
          v-if="renamingId === template.id"
          v-model="renameDraft"
          data-selectable
          type="text"
          data-test="template-rename-input"
          aria-label="Rename template"
          class="border-edge h-7 min-w-0 rounded-control border bg-bath px-2 text-body text-ink"
          @keydown.enter="commitRename(template.id)"
          @keydown.esc="cancelRename"
          @blur="commitRename(template.id)"
        />
        <button
          v-else
          type="button"
          data-test="template-load"
          class="min-w-0 truncate text-left text-body text-ink-2 hover:text-ink"
          @click="loadTemplate(template)"
        >
          <span data-test="template-row-name" class="truncate">{{ template.name }}</span>
          <span class="data-mono ml-1.5 text-caption text-ink-3">
            {{ template.form.model ? modelLabel(template.form.model) : "no model" }}
          </span>
        </button>

        <button
          type="button"
          data-test="template-rename"
          class="rounded-control px-1.5 py-0.5 text-caption text-ink-3 hover:bg-bath hover:text-ink"
          @click="startRename(template)"
        >
          Rename
        </button>
        <button
          type="button"
          data-test="template-delete"
          class="rounded-control px-1.5 py-0.5 text-caption hover:bg-bath"
          :class="confirmingDeleteId === template.id ? 'text-stop' : 'text-ink-3 hover:text-stop'"
          @click="requestDelete(template)"
        >
          {{ confirmingDeleteId === template.id ? "Confirm?" : "Delete" }}
        </button>
      </div>
      <p
        v-if="visibleTemplates.length === 0"
        data-test="template-empty"
        class="py-2 text-caption text-ink-3"
      >
        {{
          search.trim() ? "No templates match." : "No templates yet — Save the current settings."
        }}
      </p>
    </div>
  </div>
</template>
