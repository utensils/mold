<script setup lang="ts">
import { computed, ref, watch } from "vue";
import EngineSection from "../components/settings/EngineSection.vue";
import PerformanceSection from "../components/settings/PerformanceSection.vue";
import GenerationSection from "../components/settings/GenerationSection.vue";
import ExpansionSection from "../components/settings/ExpansionSection.vue";
import AccountsSection from "../components/settings/AccountsSection.vue";
import AppSection from "../components/settings/AppSection.vue";
import ProfilesSection from "../components/settings/ProfilesSection.vue";
import AdvancedSection from "../components/settings/AdvancedSection.vue";
import AboutSection from "../components/settings/AboutSection.vue";
import ConfigSettingRow from "../components/settings/ConfigSettingRow.vue";
import {
  ENGINE_KEY_SCHEMAS,
  ENV_KNOB_SCHEMAS,
  matchesSearch,
  SECTIONS,
  type SectionId,
} from "../lib/settingsSchema";
import { useConnectionStore } from "../stores/connection";
import { useModelStore } from "../stores/models";
import { useSettingsConfigStore } from "../stores/settingsConfig";

const conn = useConnectionStore();
const config = useSettingsConfigStore();
const models = useModelStore();

const section = ref<SectionId>("engine");
const query = ref("");

watch(
  () => conn.ready,
  (ready) => {
    if (ready) {
      void config.load();
      void models.fetch();
    }
  },
  { immediate: true },
);

/**
 * Search spans every curated key (engine + env knobs) plus raw advanced
 * rows. Results render as a flat list of curated rows; matches that are
 * whole sections (accounts/app/about) surface as section links.
 */
const searching = computed(() => query.value.trim().length > 0);

const matchedConfigKeys = computed(() =>
  ENGINE_KEY_SCHEMAS.filter((s) => matchesSearch(query.value, s) && config.row(s.key) !== null).map(
    (s) => s.key,
  ),
);
const matchedKnobs = computed(() => ENV_KNOB_SCHEMAS.some((s) => matchesSearch(query.value, s)));
const matchedAdvanced = computed(() =>
  config.advancedRows.filter((r) => matchesSearch(query.value, { key: r.key, label: r.key })),
);

const componentFor: Record<SectionId, unknown> = {
  engine: EngineSection,
  performance: PerformanceSection,
  generation: GenerationSection,
  expansion: ExpansionSection,
  accounts: AccountsSection,
  app: AppSection,
  profiles: ProfilesSection,
  advanced: AdvancedSection,
  about: AboutSection,
};
</script>

<template>
  <div class="flex h-full min-h-0">
    <!-- Section rail -->
    <nav class="border-edge flex w-44 shrink-0 flex-col gap-0.5 border-r bg-bench px-2 py-3">
      <span
        class="font-display px-2 pb-2 text-display-sm font-bold text-ink"
        style="font-stretch: 90%"
      >
        Settings
      </span>
      <button
        v-for="s in SECTIONS"
        :key="s.id"
        type="button"
        class="h-7 rounded-control px-2.5 text-left text-body"
        :class="
          section === s.id && !searching
            ? 'bg-[color-mix(in_srgb,var(--safelight)_14%,transparent)] text-ink'
            : 'text-ink-2 hover:text-ink'
        "
        @click="
          section = s.id;
          query = '';
        "
      >
        {{ s.label }}
      </button>
    </nav>

    <div class="min-h-0 min-w-0 flex-1 overflow-y-auto">
      <header
        class="border-edge sticky top-0 z-10 flex h-11 items-center gap-3 border-b bg-bath px-5"
      >
        <span class="text-body font-medium text-ink">
          {{ searching ? "Search" : SECTIONS.find((s) => s.id === section)?.label }}
        </span>
        <input
          v-model="query"
          data-selectable
          type="search"
          placeholder="Search settings…"
          class="border-edge ml-auto h-7 w-64 rounded-control border bg-bench px-2 text-body text-ink placeholder:text-ink-3"
        />
      </header>

      <div class="max-w-2xl px-5 py-4">
        <!-- Search results: flat curated rows + knob/advanced pointers -->
        <template v-if="searching">
          <p v-if="config.available === false" class="text-caption text-ink-3">
            This engine doesn't expose configuration.
          </p>
          <ConfigSettingRow v-for="key in matchedConfigKeys" :key="key" :schema-key="key" />
          <button
            v-if="matchedKnobs"
            type="button"
            class="border-edge mt-3 h-8 w-full rounded-control border px-3 text-left text-body text-ink-2 hover:text-ink"
            @click="
              section = 'performance';
              query = '';
            "
          >
            Performance knobs match — open Performance →
          </button>
          <div v-if="matchedAdvanced.length" class="mt-3">
            <span class="edge-code">ADVANCED MATCHES</span>
            <AdvancedSection :filter="(r) => matchesSearch(query, { key: r.key, label: r.key })" />
          </div>
          <p
            v-if="matchedConfigKeys.length === 0 && !matchedKnobs && matchedAdvanced.length === 0"
            class="text-caption text-ink-3"
          >
            Nothing matches “{{ query }}”.
          </p>
        </template>

        <!-- Section content -->
        <template v-else>
          <p
            v-if="
              config.available === false &&
              ['generation', 'expansion', 'advanced', 'profiles'].includes(section)
            "
            class="text-caption text-ink-3"
          >
            This engine doesn't expose configuration — it may predate the config API.
          </p>
          <component :is="componentFor[section]" v-else />
        </template>
      </div>
    </div>
  </div>
</template>
