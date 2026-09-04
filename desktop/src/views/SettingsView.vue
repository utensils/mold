<script setup lang="ts">
/*
 * Settings (README §03): a 200px jump nav — Search settings… then every
 * section in the lexicon — beside one scrolling page of always-open
 * sections. Search narrows the nav and the page to the sections that match;
 * a `?section=` deep link (the Library trash banner, the native Check for
 * Updates action) jumps to its section. Nothing here blocks first use (G7).
 */
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
  type ComponentPublicInstance,
} from "vue";
import { useRoute } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import PairingAccessPanel from "@studio/components/PairingAccessPanel.vue";
import LicenseSettingsPanel from "@studio/components/LicenseSettingsPanel.vue";
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
import ProfilesSection from "../components/settings/ProfilesSection.vue";
import AdvancedSection from "../components/settings/AdvancedSection.vue";
import { SECTIONS, sectionMatchesSearch, type SectionId } from "../lib/settingsSchema";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useSettingsConfigStore } from "../stores/settingsConfig";

const conn = useConnectionStore();
const config = useSettingsConfigStore();
const models = useModelStore();

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

const query = ref("");
const searching = computed(() => query.value.trim().length > 0);
const advancedKeys = computed(() => config.advancedRows.map((row) => row.key));

/** While searching, only the matching sections show; otherwise all of them. */
const visibleSections = computed(() =>
  SECTIONS.filter(
    (section) => !searching.value || sectionMatchesSearch(query.value, section, advancedKeys.value),
  ),
);

/** The nav's highlighted section: the one at the top of the page, or the
 * one last jumped to while that scroll is still settling. */
const active = ref<SectionId>("app");
const sectionEls = new Map<SectionId, HTMLElement>();
const contentEl = ref<HTMLElement | null>(null);
let observer: IntersectionObserver | null = null;
let settling: ReturnType<typeof setTimeout> | null = null;

/** Vue re-invokes a function `:ref` on EVERY patch of its element, so this
 *  must be idempotent: re-registering all fourteen sections on each keystroke
 *  in the search field is what made the nav highlight flicker. */
function bindSection(id: SectionId, el: Element | ComponentPublicInstance | null) {
  const previous = sectionEls.get(id);
  const next = el instanceof HTMLElement ? el : null;
  if (previous === next) return;
  if (previous) observer?.unobserve(previous);
  if (next) {
    sectionEls.set(id, next);
    next.dataset.section = id;
    observer?.observe(next);
  } else sectionEls.delete(id);
}

/** One stable `:ref` callback per section. An inline arrow is a NEW function
 *  every render, which Vue treats as a changed ref: every keystroke in the
 *  search field unobserved and re-observed all fourteen sections, and the
 *  nav highlight flickered as the observer re-fired. */
const sectionBinders = new Map<SectionId, (el: Element | ComponentPublicInstance | null) => void>();
function sectionBinder(id: SectionId) {
  let binder = sectionBinders.get(id);
  if (!binder) {
    binder = (el) => bindSection(id, el);
    sectionBinders.set(id, binder);
  }
  return binder;
}

function jump(id: SectionId) {
  active.value = id;
  // A smooth scroll passes other sections on its way; hold the pick until it lands.
  if (settling) clearTimeout(settling);
  settling = setTimeout(() => (settling = null), 800);
  sectionEls.get(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

onMounted(() => {
  if (typeof IntersectionObserver === "undefined") return;
  observer = new IntersectionObserver(
    (entries) => {
      if (settling) return;
      const top = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      const id = (top?.target as HTMLElement | undefined)?.dataset.section as SectionId | undefined;
      if (id && visibleSections.value.some((s) => s.id === id)) active.value = id;
    },
    { root: contentEl.value, rootMargin: "0px 0px -70% 0px" },
  );
  for (const el of sectionEls.values()) observer.observe(el);
});
onBeforeUnmount(() => {
  observer?.disconnect();
  if (settling) clearTimeout(settling);
});

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
    if (!SECTIONS.some((s) => s.id === id)) return;
    await nextTick();
    jump(id as SectionId);
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
  <div class="flex h-full min-h-0 bg-bg">
    <nav
      class="flex w-[var(--mold-shell-settingsnav-w)] shrink-0 flex-col gap-px overflow-y-auto border-r border-border bg-chrome px-2 py-3"
      aria-label="Settings sections"
    >
      <label
        class="mb-2 flex h-[26px] items-center gap-1.5 rounded-control border border-border bg-bg px-2 focus-within:border-border-focus"
      >
        <Icon name="search" :size="13" class="shrink-0 text-fg-dim" />
        <input
          v-model="query"
          data-selectable
          data-test="settings-search"
          type="search"
          aria-label="Search settings"
          placeholder="Search settings…"
          class="min-w-0 flex-1 bg-transparent text-xs text-fg outline-none placeholder:text-fg-dim"
        />
      </label>
      <button
        v-for="s in visibleSections"
        :key="s.id"
        type="button"
        :data-test="`settings-nav-${s.id}`"
        class="flex min-h-8 items-center rounded-control px-2.5 py-1.5 text-left text-xs transition-colors duration-100"
        :class="active === s.id ? 'bg-accent-tint text-fg' : 'text-fg-2 hover:bg-surface'"
        :aria-current="active === s.id ? 'true' : undefined"
        @click="jump(s.id)"
      >
        {{ s.label }}
      </button>
    </nav>

    <div ref="contentEl" class="flex min-h-0 flex-1 flex-col gap-[18px] overflow-y-auto p-[18px]">
      <p v-if="config.available === false" class="text-micro text-fg-dim">
        This engine doesn't expose configuration — some sections below may be empty.
      </p>

      <section
        v-for="s in visibleSections"
        :key="s.id"
        :ref="sectionBinder(s.id)"
        :data-test="`section-${s.id}`"
        class="flex scroll-mt-[18px] flex-col gap-2.5"
      >
        <div class="flex flex-col gap-1">
          <span class="ms-group-label uppercase">{{ s.label }}</span>
          <span class="text-micro text-fg-dim">{{ s.summary }}</span>
        </div>

        <div class="rounded-control border border-border bg-panel">
          <template v-if="s.id === 'licenses'">
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
          <div v-else-if="s.id === 'pairing'" class="p-3.5">
            <PairingAccessPanel
              :target="pairingTarget"
              :suggested-base-url="pairingBaseUrl"
              host-label="This device"
            />
          </div>
          <template v-else-if="s.id === 'updates'">
            <UpdatesSection />
            <AboutSection />
          </template>
          <component :is="componentFor[s.id]" v-else />
        </div>
      </section>

      <p
        v-if="searching && visibleSections.length === 0"
        class="text-micro text-fg-dim"
        data-test="no-search-results"
      >
        Nothing matches “{{ query }}”.
      </p>
    </div>
  </div>
</template>
