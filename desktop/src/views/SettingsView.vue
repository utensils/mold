<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { useRoute } from "vue-router";
import AccordionSection from "@ui/components/AccordionSection.vue";
import CardSurface from "@ui/components/CardSurface.vue";
import PairingAccessPanel from "@studio/components/PairingAccessPanel.vue";
import AppearanceCard from "../components/settings/AppearanceCard.vue";
import UpdatesSection from "../components/settings/UpdatesSection.vue";
import AboutSection from "../components/settings/AboutSection.vue";
import HostsSection from "../components/settings/HostsSection.vue";
import PerformanceSection from "../components/settings/PerformanceSection.vue";
import GenerationSection from "../components/settings/GenerationSection.vue";
import MediaSection from "../components/settings/MediaSection.vue";
import LibrarySection from "../components/settings/LibrarySection.vue";
import ExpansionSection from "../components/settings/ExpansionSection.vue";
import AccountsSection from "../components/settings/AccountsSection.vue";
import ProfilesSection from "../components/settings/ProfilesSection.vue";
import AdvancedSection from "../components/settings/AdvancedSection.vue";
import {
  ACCORDION_SECTIONS,
  SECTIONS,
  sectionMatchesSearch,
  type SectionId,
} from "../lib/settingsSchema";
import { useConnectionStore } from "../stores/connection";
import { useModelStore } from "../stores/models";
import { useSettingsConfigStore } from "../stores/settingsConfig";

const conn = useConnectionStore();
const config = useSettingsConfigStore();
const models = useModelStore();

const pairingTarget = computed(() =>
  conn.baseUrl ? { baseUrl: conn.baseUrl, apiKey: conn.apiKey } : null,
);
const pairingBaseUrl = computed(() => conn.baseUrl ?? "http://127.0.0.1:7680");

const query = ref("");
/** Which "All settings" accordion is open (one at a time) when not searching. */
const openSection = ref<SectionId | null>(null);
const updatesCard = ref<HTMLElement | null>(null);

// Deep links open accordions or reveal an always-open top card. The native
// Check for Updates menu action uses `section=updates`; the Library trash
// banner uses `section=library`. The view also mounts router-less in tests, so
// the route is optional.
const route = useRoute();
watch(
  () => route?.query.section,
  async (section) => {
    if (typeof section !== "string") return;
    if (ACCORDION_SECTIONS.some((s) => s.id === section)) openSection.value = section as SectionId;
    if (section === "updates") {
      await nextTick();
      updatesCard.value?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  },
  { immediate: true },
);

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

const searching = computed(() => query.value.trim().length > 0);
const advancedKeys = computed(() => config.advancedRows.map((row) => row.key));

const componentFor: Record<string, unknown> = {
  performance: PerformanceSection,
  generation: GenerationSection,
  media: MediaSection,
  library: LibrarySection,
  expansion: ExpansionSection,
  accounts: AccountsSection,
  profiles: ProfilesSection,
  advanced: AdvancedSection,
};

/** While searching, only the matching accordions show; otherwise all of them. */
const visibleSections = computed(() =>
  ACCORDION_SECTIONS.filter(
    (section) => !searching.value || sectionMatchesSearch(query.value, section, advancedKeys.value),
  ),
);

/**
 * Hosts is a doorway to the Machines workspace, not an accordion — but it owns
 * host-scoped engine settings (models_dir, output_dir). Surface the doorway
 * while searching when the query matches those keys/labels (or "Hosts"), so a
 * search for "models_dir" / "Output directory" points the user at where those
 * live instead of returning nothing.
 */
const hostsSection = SECTIONS.find((s) => s.id === "hosts")!;
const hostsMatches = computed(
  () => searching.value && sectionMatchesSearch(query.value, hostsSection),
);

/** Search auto-opens every match; otherwise the single manually-opened one. */
function isOpen(id: SectionId): boolean {
  return searching.value ? true : openSection.value === id;
}

function toggle(id: SectionId): void {
  if (searching.value) return;
  openSection.value = openSection.value === id ? null : id;
}
</script>

<template>
  <div class="flex h-full min-h-0 flex-col">
    <!-- Workspace header -->
    <header class="border-edge flex h-13 shrink-0 items-center gap-3 border-b px-6">
      <h1 class="font-display text-display-sm font-bold text-ink" style="font-stretch: 90%">
        Settings
      </h1>
      <div class="flex-1" />
      <input
        v-model="query"
        data-selectable
        data-test="settings-search"
        type="search"
        aria-label="Search settings"
        placeholder="Search settings…"
        class="border-edge h-8 w-64 rounded-control border bg-bench px-2.5 text-body text-ink placeholder:text-ink-3"
      />
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto">
      <div class="mx-auto flex max-w-2xl flex-col gap-7 px-6 py-6">
        <!-- Always-open top cards — nothing here blocks first use (G7) -->
        <template v-if="!searching">
          <section data-test="appearance-card">
            <div class="edge-code mb-2.5 uppercase">Appearance</div>
            <AppearanceCard />
          </section>

          <section data-test="mobile-pairing-region">
            <div class="edge-code mb-2.5 uppercase">Mobile pairing</div>
            <PairingAccessPanel
              :target="pairingTarget"
              :suggested-base-url="pairingBaseUrl"
              host-label="This device"
            />
          </section>

          <section ref="updatesCard" data-test="updates-card">
            <div class="edge-code mb-2.5 uppercase">Updates</div>
            <UpdatesSection />
          </section>

          <section data-test="about-card">
            <div class="edge-code mb-2.5 uppercase">About</div>
            <CardSurface>
              <AboutSection />
            </CardSurface>
          </section>
        </template>

        <!-- Hosts live in the Machines workspace — this is the doorway. It rides
             at the top normally, and reappears as a search result when the query
             matches host-owned settings (models_dir / output_dir). -->
        <section v-if="!searching || hostsMatches" data-test="hosts-region">
          <div class="edge-code mb-2.5 uppercase">Hosts</div>
          <HostsSection />
        </section>

        <!-- Everything deeper, collapsed until wanted -->
        <section data-test="all-settings">
          <div class="edge-code mb-2.5 uppercase">
            {{ searching ? "Search results" : "All settings" }}
          </div>
          <p v-if="config.available === false" class="mb-3 text-caption text-ink-3">
            This engine doesn't expose configuration — some sections below may be empty.
          </p>
          <div class="flex flex-col gap-2.5">
            <AccordionSection
              v-for="s in visibleSections"
              :key="s.id"
              :data-test="`accordion-${s.id}`"
              :title="s.label"
              :summary="s.summary"
              :icon="s.icon"
              tone="halide"
              :open="isOpen(s.id)"
              @toggle="toggle(s.id)"
            >
              <component :is="componentFor[s.id]" />
            </AccordionSection>
            <p
              v-if="searching && visibleSections.length === 0 && !hostsMatches"
              class="text-caption text-ink-3"
              data-test="no-search-results"
            >
              Nothing matches “{{ query }}”.
            </p>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>
