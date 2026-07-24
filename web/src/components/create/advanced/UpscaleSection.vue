<script setup lang="ts">
/*
 * "Upscale after generate" — the post-generation Real-ESRGAN pass.
 *
 * The wire contract is a single `GenerateRequest.upscale_model` string, so the
 * always-visible picker uses an explicit Off option followed by the host's
 * available upscalers. This matches the desktop Advanced section exactly.
 *
 * The model list is fetched here (like LoraPicker owns its catalog fetch) so
 * the section is self-contained; a host-aware parent can pass `models` to skip
 * the fetch and scope it to the routed machine.
 */
import { computed, onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import AccordionSection from "@ui/components/AccordionSection.vue";
import { fetchModels } from "../../../api";
import type { ModelInfoExtended } from "../../../types";
import { selectUpscalers, upscaleFactor } from "./upscalers";

const props = withDefaults(
  defineProps<{
    /** The form's `upscaleModel` — "" means the pass is off. */
    modelValue: string;
    /** Pre-fetched `/api/models` rows. Null → this section fetches them. */
    models?: ModelInfoExtended[] | null;
  }>(),
  { models: null },
);

const emit = defineEmits<{ "update:modelValue": [value: string] }>();

const fetched = ref<ModelInfoExtended[] | null>(null);
const failed = ref(false);

onMounted(async () => {
  if (props.models) return;
  try {
    fetched.value = await fetchModels();
  } catch {
    failed.value = true;
    fetched.value = [];
  }
});

/** Null until a listing (injected or fetched) has arrived. */
const listing = computed(() => props.models ?? fetched.value);
const loaded = computed(() => listing.value !== null);
const upscalers = computed(() => selectUpscalers(listing.value ?? []));
const selected = computed(() => props.modelValue.trim());

/*
 * A reused print (or a switched host) can carry an upscaler this listing does
 * not contain. The request would still send it, so the picker keeps it as a
 * row rather than silently disagreeing with what will be submitted.
 */
const options = computed(() => {
  const rows = upscalers.value;
  if (!selected.value || rows.some((m) => m.name === selected.value)) {
    return rows;
  }
  return [{ name: selected.value, downloaded: false, unlisted: true }, ...rows];
});
const none = computed(
  () => loaded.value && upscalers.value.length === 0 && !selected.value,
);
function optionLabel(model: {
  name: string;
  downloaded: boolean;
  unlisted?: boolean;
}): string {
  if (model.unlisted) return `${model.name} — not listed on this server`;
  return model.downloaded
    ? model.name
    : `${model.name} — downloads on first use`;
}

const factor = computed(() => {
  const model = upscalers.value.find((m) => m.name === selected.value);
  return model ? upscaleFactor(model) : null;
});

const summary = computed(() => {
  if (none.value) return "No upscalers on this server";
  if (!selected.value) return "Off";
  return factor.value ? `${selected.value} · ${factor.value}×` : selected.value;
});
</script>

<template>
  <AccordionSection
    icon="upscale"
    title="Upscale after generate"
    :summary="summary"
    :header-interactive="false"
    :open="true"
    data-test="section-upscale"
  >
    <div v-if="none" class="up__empty" data-test="upscale-empty">
      <p class="up__note">
        {{
          failed
            ? "could not read this server's model list."
            : "no upscalers installed on this server."
        }}
      </p>
      <RouterLink to="/models" class="up__link" data-test="upscale-browse">
        Pull one from Models
      </RouterLink>
    </div>

    <div v-else class="up__field">
      <label class="up__label" for="upscale-model-select">Upscaler model</label>
      <select
        id="upscale-model-select"
        class="up__select"
        data-test="upscale-model"
        :value="selected"
        @change="
          emit('update:modelValue', ($event.target as HTMLSelectElement).value)
        "
      >
        <option value="">Off</option>
        <option v-for="u in options" :key="u.name" :value="u.name">
          {{ optionLabel(u) }}
        </option>
      </select>
      <p v-if="factor" class="up__note" data-test="upscale-factor">
        each print is enlarged {{ factor }}× after it renders.
      </p>
      <p v-else class="up__note" data-test="upscale-factor">
        scale is set by the checkpoint you pick.
      </p>
    </div>
  </AccordionSection>
</template>

<style scoped>
.up__label {
  display: block;
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.up__select {
  width: 100%;
  box-sizing: border-box;
  height: 40px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  padding: 0 12px;
  font-size: 13px;
  font-family: var(--f-mono);
}
.up__note {
  font-family: var(--f-mono);
  font-size: 10.5px;
  color: var(--ink-3);
  margin-top: 6px;
}
.up__empty {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
}
.up__empty .up__note {
  margin-top: 0;
}
.up__link {
  border: 1px solid var(--ce);
  border-radius: var(--radius-pill);
  color: var(--ink-2);
  padding: 5px 12px;
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.up__link:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}
</style>
