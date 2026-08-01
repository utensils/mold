<script lang="ts">
import { MOBILE_GENERATION_TEMPLATES_STORAGE_KEY } from "./mobileTemplateStorage";

export { MOBILE_GENERATION_TEMPLATES_STORAGE_KEY };
</script>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  formatTemplateMediaReferences,
  deleteGenerationTemplateWithMedia,
  fallbackTemplateName,
  loadGenerationTemplates,
  renameGenerationTemplate,
  saveGenerationTemplateWithMedia,
  sortGenerationTemplates,
  type GenerationTemplate,
  type GenerationTemplateSort,
  unsnapshottedTemplateMediaReferences,
} from "../lib/generationTemplates";
import type { GenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";
import { modelDisplayNameForId } from "../lib/models";

const props = defineProps<{ form: GenerateForm; hostId: string; models?: ModelEntry[] }>();
const modelLabel = (name: string) => modelDisplayNameForId(name, props.models ?? []);
const emit = defineEmits<{ (event: "load", template: GenerationTemplate): void }>();

const open = ref(false);
const name = ref("");
const search = ref("");
const sort = ref<GenerationTemplateSort>("updated-desc");
const templates = ref<GenerationTemplate[]>([]);
const renamingId = ref<string | null>(null);
const renameDraft = ref("");
const confirmingDeleteId = ref<string | null>(null);
const saving = ref(false);
const error = ref("");

const visible = computed(() => {
  const query = search.value.trim().toLowerCase();
  const sorted = sortGenerationTemplates(templates.value, sort.value);
  if (!query) return sorted;
  return sorted.filter((template) =>
    [template.name, template.form.model, template.form.prompt, template.form.negativePrompt]
      .join(" ")
      .toLowerCase()
      .includes(query),
  );
});

function refresh(): void {
  templates.value = loadGenerationTemplates(sort.value, MOBILE_GENERATION_TEMPLATES_STORAGE_KEY);
}

function toggle(): void {
  open.value = !open.value;
  if (open.value) refresh();
}

async function save(): Promise<void> {
  if (saving.value) return;
  saving.value = true;
  error.value = "";
  try {
    await saveGenerationTemplateWithMedia(
      name.value || fallbackTemplateName(props.form),
      props.form,
      MOBILE_GENERATION_TEMPLATES_STORAGE_KEY,
      props.hostId,
    );
    name.value = "";
    refresh();
  } catch (cause) {
    error.value =
      cause instanceof Error ? cause.message : "Couldn’t save the template and its source image.";
  } finally {
    saving.value = false;
  }
}

function load(template: GenerationTemplate): void {
  emit("load", template);
  open.value = false;
}

function mediaToReAdd(template: GenerationTemplate) {
  return unsnapshottedTemplateMediaReferences(template);
}

function startRename(template: GenerationTemplate): void {
  confirmingDeleteId.value = null;
  renamingId.value = template.id;
  renameDraft.value = template.name;
}

function commitRename(id: string): void {
  if (renamingId.value !== id) return;
  renameGenerationTemplate(id, renameDraft.value, MOBILE_GENERATION_TEMPLATES_STORAGE_KEY);
  renamingId.value = null;
  refresh();
}

async function remove(template: GenerationTemplate): Promise<void> {
  if (confirmingDeleteId.value !== template.id) {
    confirmingDeleteId.value = template.id;
    renamingId.value = null;
    return;
  }
  const deletion = deleteGenerationTemplateWithMedia(
    template,
    MOBILE_GENERATION_TEMPLATES_STORAGE_KEY,
  );
  confirmingDeleteId.value = null;
  refresh();
  await deletion;
}

onMounted(refresh);
</script>

<template>
  <section class="mobile-generate-disclosure mobile-template-controls">
    <button
      type="button"
      class="mobile-disclosure-button"
      data-test="mobile-template-disclosure"
      :aria-expanded="open"
      aria-controls="mobile-template-panel"
      @click="toggle"
    >
      <span>Templates</span>
      <span class="mobile-disclosure-summary"> {{ templates.length }} saved </span>
      <span aria-hidden="true">{{ open ? "−" : "+" }}</span>
    </button>

    <div v-if="open" id="mobile-template-panel" class="mobile-disclosure-panel">
      <div class="mobile-inline-form">
        <input
          v-model="name"
          class="control"
          type="text"
          placeholder="Template name"
          data-test="mobile-template-name"
          @keydown.enter="save"
        />
        <button
          type="button"
          class="secondary-button"
          data-test="mobile-template-save"
          :disabled="saving"
          @click="save"
        >
          {{ saving ? "Saving…" : "Save" }}
        </button>
      </div>
      <p v-if="error" class="mobile-helper-text mobile-inline-danger" role="alert">
        {{ error }}
      </p>

      <div class="mobile-inline-form mobile-template-filter">
        <input
          v-model="search"
          class="control"
          type="search"
          placeholder="Search templates"
          data-test="mobile-template-search"
        />
        <select v-model="sort" class="control" aria-label="Sort templates" @change="refresh">
          <option value="updated-desc">Newest</option>
          <option value="updated-asc">Oldest</option>
          <option value="name-asc">A–Z</option>
          <option value="name-desc">Z–A</option>
        </select>
      </div>

      <div class="mobile-template-list">
        <article
          v-for="template in visible"
          :key="template.id"
          class="mobile-template-row"
          data-test="mobile-template-row"
        >
          <input
            v-if="renamingId === template.id"
            v-model="renameDraft"
            class="control"
            type="text"
            aria-label="Rename template"
            data-test="mobile-template-rename-input"
            @keydown.enter="commitRename(template.id)"
            @keydown.esc="renamingId = null"
          />
          <button
            v-else
            type="button"
            class="mobile-template-load"
            data-test="mobile-template-load"
            @click="load(template)"
          >
            <strong>{{ template.name }}</strong>
            <span>{{ template.form.model ? modelLabel(template.form.model) : "No model" }}</span>
          </button>
          <div class="mobile-template-actions">
            <button type="button" data-test="mobile-template-rename" @click="startRename(template)">
              Rename
            </button>
            <button
              type="button"
              class="mobile-inline-danger"
              data-test="mobile-template-delete"
              @click="remove(template)"
            >
              {{ confirmingDeleteId === template.id ? "Confirm" : "Delete" }}
            </button>
          </div>
          <p v-if="mediaToReAdd(template).length" class="mobile-helper-text">
            Re-add {{ formatTemplateMediaReferences(mediaToReAdd(template)) }} after loading.
          </p>
        </article>
        <p v-if="visible.length === 0" class="mobile-helper-text">
          {{ search.trim() ? "No templates match." : "Save these settings for quick reuse." }}
        </p>
      </div>
    </div>
  </section>
</template>
