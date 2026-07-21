<script setup lang="ts">
/*
 * Models workspace (spec §03/§06, prototype WEB MODELS lines 1582-1616). One
 * header with an Installed | Discover segmented control. Installed lists the
 * models on disk as tappable rows with a local search + G5 empty states;
 * Discover keeps the full live-catalog behaviour (filters, infinite scroll)
 * restyled as pull cards. Both open the shared model detail drawer.
 */
import { computed, onMounted, ref } from "vue";
import SegmentedControl, {
  type SegmentOption,
} from "@ui/components/SegmentedControl.vue";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import Icon from "@ui/components/Icon.vue";
import { useCatalog, type ModelsTab } from "../composables/useCatalog";
import CatalogSidebar from "../components/CatalogSidebar.vue";
import CatalogTopbar from "../components/CatalogTopbar.vue";
import CatalogCardGrid from "../components/CatalogCardGrid.vue";
import InstalledModelRow from "../components/models/InstalledModelRow.vue";
import ModelDetailDrawer from "../components/models/ModelDetailDrawer.vue";

const cat = useCatalog();

const tabOptions: SegmentOption<ModelsTab>[] = [
  { value: "installed", label: "Installed" },
  { value: "discover", label: "Discover" },
];

const installedQuery = ref("");

const filteredInstalled = computed(() => {
  const q = installedQuery.value.trim().toLowerCase();
  if (!q) return cat.installed.value;
  return cat.installed.value.filter(
    (m) =>
      m.name.toLowerCase().includes(q) || m.family.toLowerCase().includes(q),
  );
});

function clearInstalledSearch() {
  installedQuery.value = "";
}

onMounted(() => {
  void cat.refreshInstalled();
  void cat.refresh();
});
</script>

<template>
  <div class="models">
    <header class="models__header">
      <h1 class="models__title">Models</h1>
      <div class="models__spacer" />
      <SegmentedControl
        class="models__tabs"
        data-test="models-tabs"
        :model-value="cat.tab.value"
        :options="tabOptions"
        label="Models view"
        @update:model-value="cat.setTab"
      />
    </header>

    <!-- Installed -->
    <section v-if="cat.tab.value === 'installed'" data-test="installed-tab">
      <label class="search">
        <Icon name="search" :size="16" class="search__icon" />
        <input
          v-model="installedQuery"
          type="search"
          class="search__input"
          placeholder="Search installed…"
          autocomplete="off"
          spellcheck="false"
          data-test="installed-search"
        />
      </label>

      <div
        v-if="cat.installedLoading.value && cat.installed.value.length === 0"
        class="state"
        data-test="installed-loading"
      >
        Loading…
      </div>

      <div
        v-else-if="cat.installed.value.length === 0"
        class="empty"
        data-test="installed-empty"
      >
        <EmptyStateBlock
          icon="models"
          headline="Nothing installed — pull a model to start."
          guidance="Browse the catalog to download your first model."
        >
          <template #action>
            <button
              type="button"
              class="empty__cta"
              data-test="discover-cta"
              @click="cat.setTab('discover')"
            >
              Discover models
            </button>
          </template>
        </EmptyStateBlock>
      </div>

      <div
        v-else-if="filteredInstalled.length === 0"
        class="empty"
        data-test="installed-no-match"
      >
        <EmptyStateBlock
          icon="search"
          headline="Nothing installed matches."
          guidance="No installed model matches your search."
        >
          <template #action>
            <button
              type="button"
              class="empty__cta"
              data-test="clear-search"
              @click="clearInstalledSearch"
            >
              Clear search
            </button>
          </template>
        </EmptyStateBlock>
      </div>

      <div v-else class="installed-grid">
        <InstalledModelRow
          v-for="model in filteredInstalled"
          :key="model.name"
          :model="model"
          @open="cat.openInstalledDetail(model)"
        />
      </div>
    </section>

    <!-- Discover -->
    <section v-else data-test="discover-tab">
      <CatalogTopbar />
      <div class="discover">
        <CatalogSidebar class="discover__sidebar" />
        <CatalogCardGrid />
      </div>
    </section>

    <ModelDetailDrawer />
  </div>
</template>

<style scoped>
.models {
  max-width: 1400px;
  margin: 0 auto;
  padding: 22px 20px 120px;
}

.models__header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 18px;
}

.models__title {
  margin: 0;
  font-family: var(--f-display);
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}

.models__spacer {
  flex: 1;
}

.models__tabs {
  flex: 0 0 auto;
}

/* ── Local search ─────────────────────────────────────────────────── */
.search {
  position: relative;
  display: block;
  margin-bottom: 16px;
}

.search__icon {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--ink-3);
  pointer-events: none;
}

.search__input {
  width: 100%;
  height: 40px;
  box-sizing: border-box;
  padding: 0 14px 0 36px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 13px;
  outline: none;
}

.search__input::placeholder {
  color: var(--ink-3);
}

.search__input:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 1px;
}

/* ── States ───────────────────────────────────────────────────────── */
.state {
  padding: 48px 0;
  text-align: center;
  font-size: 13px;
  color: var(--ink-3);
}

.empty {
  padding: 40px 0;
}

.empty__cta {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 9px 16px;
  border-radius: var(--radius-control);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.empty__cta:hover {
  border-color: var(--safelight);
  background: color-mix(in srgb, var(--safelight) 8%, transparent);
}

.empty__cta:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

/* ── Installed grid ───────────────────────────────────────────────── */
.installed-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}

@media (min-width: 640px) {
  .installed-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

/* ── Discover layout ──────────────────────────────────────────────── */
.discover {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 16px;
}

@media (min-width: 1024px) {
  .discover {
    flex-direction: row;
    align-items: flex-start;
  }
}

.discover__sidebar {
  display: none;
}

@media (min-width: 1024px) {
  .discover__sidebar {
    display: block;
  }
}
</style>
