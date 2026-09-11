<script setup lang="ts">
/*
 * Settings ▸ Per-style defaults: what one style starts with, when it differs
 * from the defaults every style shares.
 *
 * The engine reports these as flat `models.<style>.<field>` rows — 104 of them
 * on a machine with thirteen tuned styles, which flat is most of the page and
 * says nothing about any one style. Grouped, it is one collapsed row per
 * style; the fields arrive only when a style is opened.
 */
import { computed, ref } from "vue";
import PerStyleDefaultsRow from "@studio/components/settings/PerStyleDefaultsRow.vue";
import type { ConfigValue } from "@studio/api/config";
import { groupPerStyleRows } from "@studio/lib/settingsSchema";
import { styleDisplayName } from "@studio/lib/styleLabel";
import { useModelStore } from "../../stores/models";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";

/** Above this many styles the list stops being scannable and earns a filter. */
const FILTER_THRESHOLD = 8;

const config = useSettingsConfigStore();
const models = useModelStore();
const toasts = useToastStore();
const filter = ref("");

const groups = computed(() => groupPerStyleRows(config.perStyleRows));
const filterable = computed(() => groups.value.length > FILTER_THRESHOLD);

const shown = computed(() => {
  const query = filter.value.trim().toLowerCase();
  if (!query || !filterable.value) return groups.value;
  return groups.value.filter(
    (group) =>
      group.style.toLowerCase().includes(query) ||
      (displayName(group.style) ?? "").toLowerCase().includes(query),
  );
});

/** The friendly name this machine knows the style by, if it knows it at all —
 *  a style may be tuned on a machine that no longer lists it. */
function displayName(style: string): string | null {
  const model = models.all.find((entry) => entry.name === style);
  return model ? styleDisplayName(model) : null;
}

async function save(key: string, value: ConfigValue) {
  const error = await config.save(key, value);
  if (error) toasts.push(error, "error");
  else toasts.push(`Saved ${key}`);
}

async function reset(key: string) {
  const error = await config.reset(key);
  if (error) toasts.push(error, "error");
  else toasts.push(`Reset ${key}`);
}
</script>

<template>
  <div>
    <!-- Inset like a row: the section card has no padding of its own. -->
    <div v-if="filterable" class="px-3.5 py-3">
      <input
        v-model="filter"
        data-selectable
        data-test="per-style-filter"
        type="search"
        aria-label="Filter styles"
        placeholder="Filter styles…"
        class="h-7 w-64 rounded-control border border-border bg-bg-deep px-2 text-xs text-fg placeholder:text-fg-dim focus:border-border-focus focus:outline-none"
      />
    </div>

    <PerStyleDefaultsRow
      v-for="group in shown"
      :key="group.style"
      :style="group.style"
      :rows="group.rows"
      :display-name="displayName(group.style)"
      @save="save"
      @reset="reset"
    />

    <p
      v-if="groups.length === 0"
      class="max-w-md px-3.5 py-3 text-micro text-fg-dim"
      data-test="per-style-empty"
    >
      No style has its own defaults yet. Set one with
      <span class="font-mono">mold config set models.&lt;style&gt;.&lt;field&gt;</span>.
    </p>
    <p
      v-else-if="shown.length === 0"
      class="px-3.5 py-3 text-micro text-fg-dim"
      data-test="per-style-no-matches"
    >
      No style matches “{{ filter }}”.
    </p>
  </div>
</template>
