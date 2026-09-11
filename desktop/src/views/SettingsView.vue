<script setup lang="ts">
/*
 * Settings (README §03): a 200px jump nav — Search settings… then every
 * section in the lexicon — beside one scrolling page of always-open
 * sections. Search narrows the nav and the page to the sections that match;
 * a `?section=` deep link (the Library trash banner, the native Check for
 * Updates action) jumps to its section. Nothing here blocks first use (G7).
 *
 * The frame is the shared kit's `SettingsShell`, so the app and the browser
 * scroll, search and jump identically — including the two observers and the
 * settling hold that this view used to own. What stays here is what is this
 * app's: which sections a desktop shell renders, which body each one gets,
 * the licence machine picker, and the `?section=` route.
 */
import { computed, nextTick, ref, watch } from "vue";
import { useRoute } from "vue-router";
import PairingAccessPanel from "@studio/components/PairingAccessPanel.vue";
import LicenseSettingsPanel from "@studio/components/LicenseSettingsPanel.vue";
import SettingsShell from "@studio/components/settings/SettingsShell.vue";
import { sectionsForSurface, type SectionId } from "@studio/lib/settingsSchema";
import { openExternal } from "../lib/openExternal";
import AppearanceCard from "../components/settings/AppearanceCard.vue";
import UpdatesSection from "../components/settings/UpdatesSection.vue";
import AboutSection from "../components/settings/AboutSection.vue";
import HostsSection from "../components/settings/HostsSection.vue";
import PerformanceSection from "../components/settings/PerformanceSection.vue";
import GenerationSection from "../components/settings/GenerationSection.vue";
import MediaSection from "../components/settings/MediaSection.vue";
import StylesDiskSection from "../components/settings/StylesDiskSection.vue";
import LibrarySection from "../components/settings/LibrarySection.vue";
import ExpansionSection from "../components/settings/ExpansionSection.vue";
import AccountsSection from "../components/settings/AccountsSection.vue";
import CloudSection from "../components/settings/CloudSection.vue";
import PerStyleDefaultsSection from "../components/settings/PerStyleDefaultsSection.vue";
import ProfilesSection from "../components/settings/ProfilesSection.vue";
import AdvancedSection from "../components/settings/AdvancedSection.vue";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useSettingsConfigStore } from "../stores/settingsConfig";

const conn = useConnectionStore();
const config = useSettingsConfigStore();
const models = useModelStore();

const sections = sectionsForSurface("desktop");
const shell = ref<{ jump: (id: SectionId) => void } | null>(null);

const pairingTarget = computed(() =>
  conn.baseUrl ? { baseUrl: conn.baseUrl, apiKey: conn.apiKey } : null,
);
const pairingBaseUrl = computed(() => conn.baseUrl ?? "http://127.0.0.1:7680");

// Licence acceptance is recorded per Mold data root, so it belongs to the
// machine that will do the downloading. Generate-time consent already targets
// whatever machine the render was routed to; without a selector here,
// Settings was the one place that could only ever speak for this device.
const hostsStore = useHostsStore();
const licenseHostId = ref("local");
const licenseHosts = computed(() => hostsStore.all.filter((host) => host.baseUrl));
const licenseHost = computed(
  () =>
    licenseHosts.value.find((host) => host.id === licenseHostId.value) ?? hostsStore.primaryHost,
);
const licenseTarget = computed(() =>
  licenseHost.value?.baseUrl
    ? { baseUrl: licenseHost.value.baseUrl, apiKey: licenseHost.value.apiKey }
    : pairingTarget.value,
);
const licenseHostLabel = computed(() => licenseHost.value?.label ?? "This device");
// The row always names the machine the answers belong to, the way the mock
// does. It only becomes a picker once another machine can answer differently:
// a one-option select is a control that cannot act.
const licensePicker = computed(() => licenseHosts.value.some((host) => host.kind !== "local"));

const componentFor: Partial<Record<SectionId, unknown>> = {
  app: AppearanceCard,
  generation: GenerationSection,
  expansion: ExpansionSection,
  hosts: HostsSection,
  styles: StylesDiskSection,
  media: MediaSection,
  library: LibrarySection,
  performance: PerformanceSection,
  accounts: AccountsSection,
  cloud: CloudSection,
  styleDefaults: PerStyleDefaultsSection,
  profiles: ProfilesSection,
  advanced: AdvancedSection,
};

// The view also mounts router-less in tests, so the route is optional. The
// retired `about` section folded into Updates & about.
const route = useRoute();
watch(
  () => route?.query.section,
  async (section) => {
    if (typeof section !== "string") return;
    const id = section === "about" ? "updates" : section;
    if (!sections.some((s) => s.id === id)) return;
    await nextTick();
    shell.value?.jump(id as SectionId);
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
</script>

<template>
  <div class="flex h-full min-h-0 flex-col gap-2.5 bg-bg p-[18px]">
    <p v-if="config.available === false" class="text-micro text-fg-dim">
      This engine doesn't expose configuration — some sections below may be empty.
    </p>

    <SettingsShell
      ref="shell"
      class="settings-shell min-h-0 flex-1"
      :sections="sections"
      :raw-keys-by-section="config.rawKeysBySection"
    >
      <template #section="{ section, mounted }">
        <template v-if="!mounted" />
        <template v-else-if="section.id === 'licenses'">
          <LicenseSettingsPanel
            :target="licenseTarget"
            :host-label="licenseHostLabel"
            :open-external="openExternal"
          >
            <template #machine>
              <select
                v-if="licensePicker"
                v-model="licenseHostId"
                aria-label="Machine"
                data-test="license-host-select"
                class="h-[26px] shrink-0 rounded-control border border-border bg-bg px-1.5 font-mono text-xs text-fg"
              >
                <option v-for="host in licenseHosts" :key="host.id" :value="host.id">
                  {{ host.label }}
                  {{ host.kind === "local" ? "(this device)" : `(${host.baseUrl})` }}
                </option>
              </select>
              <span v-else class="shrink-0 font-mono text-micro text-fg-2">
                {{ licenseHostLabel }}
              </span>
            </template>
          </LicenseSettingsPanel>
        </template>
        <div v-else-if="section.id === 'pairing'" class="p-3.5">
          <PairingAccessPanel
            :target="pairingTarget"
            :suggested-base-url="pairingBaseUrl"
            host-label="This device"
          />
        </div>
        <template v-else-if="section.id === 'updates'">
          <UpdatesSection />
          <AboutSection />
        </template>
        <component :is="componentFor[section.id as SectionId]" v-else />
      </template>
    </SettingsShell>
  </div>
</template>

<style scoped>
/*
 * The app pane is a fixed height with its own scroller, where a browser page
 * scrolls the window. The shell's scroll-spy observes the content column, so
 * the content column is what has to scroll — a scroller ABOVE it moves the
 * sections and the observer root together and the nav highlight never moves.
 */
.settings-shell {
  align-items: stretch;
}
.settings-shell :deep(.ms-settings-content) {
  min-height: 0;
  overflow-y: auto;
}
</style>
