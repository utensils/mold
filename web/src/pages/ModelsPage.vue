<script setup lang="ts">
import { workspaceLabel } from "../lib/workspaces";
/*
 * Styles workspace (spec §03/§06, prototype WEB MODELS lines 1582-1616). One
 * header with a Ready to use | Browse more segmented control. Ready to use
 * lists the styles on this machine as tappable rows with a local search, the
 * three kind chips and G5 empty states; Browse more keeps the full
 * live-catalog behaviour (filters, infinite scroll). Both open the shared
 * detail drawer.
 *
 * The kind chips name and partition exactly what the composer's own toolbar
 * does — `@studio/lib/outputKind` is the one authority — so a style filters
 * under the kind it is offered under and a person learns the three words once.
 * They ride `?type=`, which is also the query Browse more reads as its
 * catalog modality, so one link serves both shelves.
 */
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import SegmentedControl, {
  type SegmentOption,
} from "@ui/components/SegmentedControl.vue";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import Icon from "@ui/components/Icon.vue";
import { useCatalog, type ModelsTab } from "../composables/useCatalog";
import { useModelInstallTargets } from "../composables/useModelInstallTargets";
import CatalogSidebar from "../components/CatalogSidebar.vue";
import CatalogTopbar from "../components/CatalogTopbar.vue";
import CatalogCardGrid from "../components/CatalogCardGrid.vue";
import InstalledModelRow from "../components/models/InstalledModelRow.vue";
import ModelDetailDrawer from "../components/models/ModelDetailDrawer.vue";
import ModelInstallTargetDialog from "../components/models/ModelInstallTargetDialog.vue";
import { toast } from "../lib/toasts";
import { modelDisplayName } from "@studio/lib/modelDisplay";
import { styleDisplayName } from "@studio/lib/styleLabel";
import {
  modelsForOutputKind,
  OUTPUT_KIND_LABEL,
  type OutputKind,
} from "../composables/useCreateOutputKind";
import type { CatalogKind, ModelInfoExtended } from "../types";

const cat = useCatalog();
const installTargets = useModelInstallTargets();
const route = useRoute();
const router = useRouter();
watch(
  () => route.query.tab,
  (tab) => {
    if (tab === "installed" || tab === "discover") cat.setTab(tab);
  },
  { immediate: true },
);
watch(
  () => [route.query.type, route.query.kind] as const,
  ([type, kind]) => {
    const modality = type === "image" || type === "video" ? type : undefined;
    const allowedKinds: CatalogKind[] = [
      "checkpoint",
      "lora",
      "vae",
      "text-encoder",
      "tokenizer",
      "clip",
      "control-net",
    ];
    const catalogKind =
      typeof kind === "string" && allowedKinds.includes(kind as CatalogKind)
        ? (kind as CatalogKind)
        : undefined;
    if (modality || catalogKind) {
      cat.setFilter({
        modality,
        kind: catalogKind,
      });
    }
  },
  { immediate: true },
);

const tabOptions: SegmentOption<ModelsTab>[] = [
  { value: "installed", label: "Ready to use" },
  { value: "discover", label: "Browse more" },
];

/* `?type=` carries `mediaTypeFromQuery`'s own values, which is what
 * `OUTPUT_KIND_BROWSE_TARGET` links to from the composer. "All" is the absence
 * of the key rather than a fourth value, so a bare /models is unfiltered. */
const KIND_QUERY: Readonly<Record<OutputKind, string>> = {
  still: "image",
  clip: "video",
  mesh: "mesh",
};
type KindChoice = OutputKind | "all";
const kindOptions: SegmentOption<KindChoice>[] = [
  { value: "all", label: "All" },
  { value: "still", label: OUTPUT_KIND_LABEL.still },
  { value: "clip", label: OUTPUT_KIND_LABEL.clip },
  { value: "mesh", label: OUTPUT_KIND_LABEL.mesh },
];
const activeKind = computed<KindChoice>(() => {
  const type = route.query.type;
  const match = (Object.keys(KIND_QUERY) as OutputKind[]).find(
    (kind) => KIND_QUERY[kind] === type,
  );
  return match ?? "all";
});
function setKind(value: string | number) {
  const query = { ...route.query };
  if (value === "all") delete query.type;
  else query.type = KIND_QUERY[value as OutputKind];
  void router.push({ query });
}

const installedQuery = ref("");

/* Search first, then the kind: a chosen kind narrows what the search found,
 * so clearing the search never resurrects a style of another kind. */
const searchedInstalled = computed(() => {
  const q = installedQuery.value.trim().toLowerCase();
  if (!q) return cat.installed.value;
  return cat.installed.value.filter((m) =>
    [
      modelDisplayName(m),
      styleDisplayName(m),
      m.name,
      m.family,
      m.description ?? "",
    ].some((value) => value.toLowerCase().includes(q)),
  );
});
const filteredInstalled = computed(() =>
  activeKind.value === "all"
    ? searchedInstalled.value
    : modelsForOutputKind(searchedInstalled.value, activeKind.value),
);

/* The no-match state is reachable from the search, from a kind chip, or from
 * both, so its one action drops every narrowing rather than only the one the
 * user happens to remember setting. */
function showAllInstalled() {
  installedQuery.value = "";
  if (activeKind.value !== "all") setKind("all");
}

/** Send an already-installed model to a connected machine that lacks it. */
async function installElsewhere(model: ModelInfoExtended) {
  const choice = await installTargets.chooseInstallTarget({
    modelId: model.name,
    displayName: styleDisplayName(model),
    ownedByOrigin: true,
  });
  if (choice.kind === "cancelled") return;
  try {
    const started = await installTargets.startDownloadOn(
      choice.target,
      model.name,
    );
    if (started.declined) return;
    toast("success", installTargets.queuedMessage(choice.target));
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  }
}

onMounted(() => {
  void cat.refreshInstalled();
  void cat.refresh();
});
</script>

<template>
  <div class="models min-w-0 w-full">
    <header class="models__header">
      <h1 class="models__title">{{ workspaceLabel("models") }}</h1>
      <div class="models__spacer" />
      <SegmentedControl
        class="models__tabs"
        data-test="models-tabs"
        :model-value="cat.tab.value"
        :options="tabOptions"
        label="Styles view"
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
          placeholder="Search your styles…"
          aria-label="Search your styles"
          autocomplete="off"
          spellcheck="false"
          data-test="installed-search"
        />
      </label>

      <SegmentedControl
        class="models__kinds"
        data-test="installed-kinds"
        wrap
        :model-value="activeKind"
        :options="kindOptions"
        label="Kind of style"
        @update:model-value="setKind"
      />

      <div
        v-if="cat.installedError.value"
        class="installed-error"
        role="alert"
        data-test="installed-error"
      >
        <p>
          Could not refresh styles on this server.
          {{ cat.installedError.value }}
        </p>
        <button
          type="button"
          class="empty__cta"
          :disabled="cat.installedLoading.value"
          @click="cat.refreshInstalled()"
        >
          Try again
        </button>
      </div>

      <div
        v-if="cat.installedLoading.value && cat.installed.value.length === 0"
        class="state"
        data-test="installed-loading"
      >
        Loading…
      </div>

      <div
        v-else-if="
          cat.installed.value.length === 0 && !cat.installedError.value
        "
        class="empty"
        data-test="installed-empty"
      >
        <EmptyStateBlock
          icon="models"
          headline="No styles on this machine yet."
          guidance="Browse styles to get one onto this machine."
        >
          <template #action>
            <button
              type="button"
              class="empty__cta"
              data-test="discover-cta"
              @click="cat.setTab('discover')"
            >
              Browse styles
            </button>
          </template>
        </EmptyStateBlock>
      </div>

      <div
        v-else-if="
          filteredInstalled.length === 0 && cat.installed.value.length > 0
        "
        class="empty"
        data-test="installed-no-match"
      >
        <!-- Reached by a search that found nothing AND by a kind chip this
             machine holds no style for; the sentence answers both. -->
        <EmptyStateBlock
          icon="search"
          headline="Nothing here matches."
          guidance="No style on this machine matches what you asked for."
        >
          <template #action>
            <button
              type="button"
              class="empty__cta"
              data-test="clear-search"
              @click="showAllInstalled"
            >
              Show all styles
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
          @install="installElsewhere(model)"
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
    <!-- One picker for the whole workspace: Discover cards, installed rows and
         the detail drawer all resolve their target through it. -->
    <ModelInstallTargetDialog />
  </div>
</template>

<style scoped>
.models {
  box-sizing: border-box;
  max-width: 1400px;
  margin: 0 auto;
  padding: 22px 20px 120px;
}

.models__header {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 14px;
  margin-bottom: 18px;
}

.models__title {
  margin: 0;
  font-family: var(--f-display);
  font-size: 2rem;
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

.models__kinds {
  margin-bottom: 16px;
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
  min-height: 44px;
  box-sizing: border-box;
  padding: 0 14px 0 36px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 1rem;
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
  font-size: 1rem;
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
  font-size: 0.875rem;
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

.installed-error {
  padding: 1rem;
  margin-bottom: 1rem;
  border: 1px solid var(--danger);
  color: var(--ink-2);
  background: var(--bath);
  overflow-wrap: anywhere;
}
.installed-error p {
  margin: 0 0 0.75rem;
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
