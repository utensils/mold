<script setup lang="ts">
import { computed, ref } from "vue";
import {
  formatTemplateMediaReferences,
  loadGenerationTemplates,
  type GenerationTemplate,
} from "../../lib/generationTemplates";
import { modelDisplayNameForId, type DisplayableModel } from "../../lib/models";
import { familyIdentity } from "../../lib/modelFamilyIdentity";

/**
 * Starting points as picture cards (README §5): every one shows a picture, a
 * plain-words label and one line about what comes with it.
 *
 * A saved starting point has no picture of its own — its media is conditioning
 * input, held in the shared IndexedDB store behind an async hydrate, not a
 * result — so the picture is the style's family mark, drawn from the same
 * `familyIdentity` table the catalog placeholder uses.
 */
const props = defineProps<{ models?: DisplayableModel[] | undefined }>();
const emit = defineEmits<{ load: [template: GenerationTemplate] }>();

const templates = ref<GenerationTemplate[]>(loadGenerationTemplates());

const cards = computed(() =>
  templates.value.map((template) => {
    const model = template.form.model;
    const parts = [
      model ? modelDisplayNameForId(model, props.models ?? []) : "no style",
      template.form.recipeCapabilities?.canvasless
        ? "3-D"
        : `${template.form.width}×${template.form.height}`,
    ];
    if (template.mediaReferences.length > 0) {
      parts.push(`from a ${formatTemplateMediaReferences(template.mediaReferences)}`);
    }
    return {
      template,
      identity: familyIdentity(template.form.family),
      note: parts.join(" · "),
    };
  }),
);
</script>

<template>
  <div data-test="starter-list">
    <p v-if="cards.length === 0" class="ms-starters__empty" data-test="starter-empty">
      No starting points yet — Edit… saves the settings you have now.
    </p>
    <button
      v-for="card in cards"
      :key="card.template.id"
      type="button"
      class="ms-starter"
      data-test="starter-card"
      @click="emit('load', card.template)"
    >
      <span class="ms-starter__picture" :data-tone="card.identity.tone" aria-hidden="true">
        {{ card.identity.mark }}
      </span>
      <span class="ms-starter__body">
        <span class="ms-starter__label">{{ card.template.name }}</span>
        <span class="ms-starter__note">{{ card.note }}</span>
      </span>
    </button>
  </div>
</template>

<style scoped>
.ms-starters__empty {
  margin: 0;
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
.ms-starter {
  display: flex;
  width: 100%;
  gap: 10px;
  margin-bottom: 8px;
  padding: 9px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: transparent;
  text-align: left;
  cursor: pointer;
  transition:
    background var(--mold-dur-quick) var(--mold-ease-out),
    border-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-starter:hover {
  border-color: var(--mold-border-focus);
  background: var(--mold-surface);
}
.ms-starter__picture {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 52px;
  height: 52px;
  flex-shrink: 0;
  border: var(--mold-bw) solid var(--mold-border);
  background: color-mix(in srgb, var(--starter-ink) 9%, var(--mold-bg-crust));
  color: color-mix(in srgb, var(--starter-ink) 88%, var(--mold-text));
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-sm);
  font-weight: 600;
  letter-spacing: -0.04em;
  --starter-ink: var(--mold-sapphire);
}
.ms-starter__picture[data-tone="warm"] {
  --starter-ink: var(--mold-blue);
}
.ms-starter__picture[data-tone="neutral"] {
  --starter-ink: color-mix(in srgb, var(--mold-text) 72%, transparent);
}
.ms-starter__body {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  gap: 3px;
}
.ms-starter__label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--mold-fs-xs);
  font-weight: 600;
  color: var(--mold-text);
}
.ms-starter__note {
  font-size: var(--mold-fs-micro);
  line-height: var(--mold-lh-snug);
  color: var(--mold-text-dim);
}
</style>
