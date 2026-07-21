<script setup lang="ts">
/*
 * Global web chrome (spec §04, prototype WEB + MOBILE WEB frames). Wide
 * viewports get a single 56px bar — logo + wordmark, the four workspace
 * pills, search, Downloads, account. Below 640px it collapses to a compact
 * 52px bar (logo, Downloads icon, hamburger) that opens the bottom nav sheet.
 *
 * Downloads opens the existing drawer via the shared `mold:open-downloads`
 * window event (App listens); the ⌘K palette uses the same channel.
 */
import { computed, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import MobileNavSheet from "./MobileNavSheet.vue";
import { useDownloads } from "../../composables/useDownloads";
import {
  markGalleryVisited,
  useNotificationSignals,
} from "../../lib/notifications";
import type { IconName } from "@ui/icons";

const route = useRoute();
const router = useRouter();
const downloads = useDownloads();

// Cross-workspace badge signals (spec §08 G11): a fresh-prints accent dot on
// the Gallery pill (cleared on entering the gallery) and a stop-tinted dot on
// the Machines pill while any registered host is offline.
const { freshPrintCount, hasOfflineHost } = useNotificationSignals();
watch(
  () => route.name,
  (name) => {
    if (name === "gallery") markGalleryVisited();
  },
  { immediate: true },
);

function pillDot(pill: Pill): "accent" | "stop" | null {
  if (pill.name === "gallery" && freshPrintCount.value > 0) return "accent";
  if (pill.name === "machines" && hasOfflineHost.value) return "stop";
  return null;
}

interface Pill {
  name: string;
  label: string;
  icon: IconName;
  path: string;
  match: string[];
}

const pills: Pill[] = [
  {
    name: "gallery",
    label: "Gallery",
    icon: "library",
    path: "/",
    match: ["gallery"],
  },
  {
    name: "create",
    label: "Create",
    icon: "create",
    path: "/create",
    match: ["create"],
  },
  {
    name: "models",
    label: "Models",
    icon: "models",
    path: "/models",
    match: ["models"],
  },
  {
    name: "machines",
    label: "Machines",
    icon: "machines",
    path: "/machines",
    match: ["machines", "machine-detail"],
  },
];

const activeName = computed(() => String(route.name ?? ""));
function isActive(pill: Pill) {
  return pill.match.includes(activeName.value);
}
function go(pill: Pill) {
  if (!isActive(pill)) void router.push(pill.path);
}

// Downloads badge — active + queued jobs, hidden at zero.
const badgeCount = computed(
  () =>
    (downloads.activeJobs?.value.length ?? (downloads.active?.value ? 1 : 0)) +
    (downloads.queued?.value.length ?? 0),
);

function openDownloads() {
  window.dispatchEvent(new CustomEvent("mold:open-downloads"));
}

// Global search — routes to the gallery with a `q` query the page seeds from.
const query = ref(String(route.query.q ?? ""));
watch(
  () => route.query.q,
  (q) => {
    query.value = String(q ?? "");
  },
);
function submitSearch() {
  const q = query.value.trim();
  void router.push({ name: "gallery", query: q ? { q } : {} });
}

const menuOpen = ref(false);
</script>

<template>
  <header class="app-nav">
    <!-- Wide bar (≥640px) -->
    <div class="bar bar--wide">
      <router-link to="/" class="brand" aria-label="mold home">
        <img
          src="/logo.png"
          alt=""
          width="22"
          height="22"
          class="brand__logo"
        />
        <span class="brand__word brand-gradient">mold</span>
      </router-link>

      <nav class="seg-group" aria-label="Workspaces">
        <button
          v-for="pill in pills"
          :key="pill.name"
          type="button"
          class="seg-pill"
          :data-on="isActive(pill) ? 'true' : undefined"
          :data-active="isActive(pill) ? 'true' : 'false'"
          :data-test="`nav-${pill.name}`"
          :aria-current="isActive(pill) ? 'page' : undefined"
          @click="go(pill)"
        >
          {{ pill.label }}
          <span
            v-if="pillDot(pill)"
            class="seg-dot"
            :class="`seg-dot--${pillDot(pill)}`"
            :data-test="`nav-dot-${pill.name}`"
            aria-hidden="true"
          />
        </button>
      </nav>

      <div class="spacer" />

      <form class="search-box" role="search" @submit.prevent="submitSearch">
        <Icon name="search" :size="14" class="search-icon" />
        <input
          v-model="query"
          type="search"
          class="search-input"
          placeholder="Search prompts…"
          aria-label="Search prompts"
          autocomplete="off"
          spellcheck="false"
        />
      </form>

      <button
        type="button"
        class="dl-chip"
        aria-label="Open downloads"
        @click="openDownloads"
      >
        <Icon name="download" :size="15" />
        <span class="dl-chip__label">Downloads</span>
        <span v-if="badgeCount > 0" class="dl-badge">
          <BadgePill tone="accent">{{ badgeCount }}</BadgePill>
        </span>
      </button>

      <router-link to="/settings" class="avatar" aria-label="Settings"
        >M</router-link
      >
    </div>

    <!-- Compact bar (<640px) -->
    <div class="bar bar--compact">
      <router-link to="/" class="brand" aria-label="mold home">
        <img
          src="/logo.png"
          alt=""
          width="20"
          height="20"
          class="brand__logo"
        />
        <span class="brand__word brand__word--sm brand-gradient">mold</span>
      </router-link>

      <div class="spacer" />

      <button
        type="button"
        class="icon-btn"
        aria-label="Open downloads"
        @click="openDownloads"
      >
        <Icon name="download" :size="16" />
        <span v-if="badgeCount > 0" class="dl-badge dl-badge--compact">
          <BadgePill tone="accent">{{ badgeCount }}</BadgePill>
        </span>
      </button>

      <button
        type="button"
        class="icon-btn"
        aria-label="Open menu"
        data-test="nav-hamburger"
        @click="menuOpen = true"
      >
        <Icon name="menu" :size="17" />
      </button>
    </div>

    <MobileNavSheet :open="menuOpen" @close="menuOpen = false" />
  </header>
</template>

<style scoped>
.app-nav {
  position: relative;
  z-index: 30;
  flex: 0 0 auto;
}

.bar {
  align-items: center;
  background: var(--bench);
  border-bottom: 1px solid var(--edge);
}

.bar--wide {
  display: none;
  height: 56px;
  gap: 16px;
  padding: 0 20px;
}

.bar--compact {
  display: flex;
  height: 52px;
  gap: 9px;
  padding: 0 14px;
}

@media (min-width: 640px) {
  .bar--wide {
    display: flex;
  }
  .bar--compact {
    display: none;
  }
}

.spacer {
  flex: 1;
}

/* ── Brand ─────────────────────────────────────────────────────────── */
.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  flex: 0 0 auto;
  text-decoration: none;
}

.brand__logo {
  flex: 0 0 auto;
  object-fit: contain;
  display: block;
}

.brand__word {
  font-family: var(--f-display);
  font-weight: 800;
  font-size: 17px;
  letter-spacing: -0.01em;
}

.brand__word--sm {
  font-size: 16px;
}

/* ── Workspace pills ───────────────────────────────────────────────── */
.seg-group {
  display: flex;
  gap: 3px;
  padding: 3px;
  background: color-mix(in srgb, var(--rebate) 7%, transparent);
  border-radius: var(--radius-control);
}

.seg-pill {
  position: relative;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  padding: 7px 14px;
  border-radius: var(--radius-control-sm);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.seg-dot {
  position: absolute;
  top: 4px;
  right: 5px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
}

.seg-dot--accent {
  background: var(--safelight);
}

.seg-dot--stop {
  background: var(--stop);
}

.seg-pill:hover {
  color: var(--rebate);
}

.seg-pill[data-on="true"] {
  background: var(--sel-bg);
  color: var(--rebate);
}

.seg-pill:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

/* ── Search ────────────────────────────────────────────────────────── */
.search-box {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 36px;
  width: 210px;
  flex: 0 0 auto;
  padding: 0 12px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
}

.search-icon {
  flex: 0 0 auto;
  color: var(--ink-3);
}

.search-input {
  flex: 1;
  min-width: 0;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12px;
  outline: none;
}

.search-input::placeholder {
  color: var(--ink-3);
}

/* ── Downloads chip + badge ────────────────────────────────────────── */
.dl-chip {
  position: relative;
  display: flex;
  align-items: center;
  gap: 7px;
  flex: 0 0 auto;
  height: 36px;
  padding: 0 13px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}

.dl-chip:hover {
  color: var(--rebate);
}

.dl-chip:focus-visible,
.icon-btn:focus-visible,
.avatar:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.dl-badge {
  position: absolute;
  top: -6px;
  right: -6px;
}

.dl-badge--compact {
  top: -3px;
  right: -3px;
}

/* ── Account avatar ────────────────────────────────────────────────── */
.avatar {
  width: 36px;
  height: 36px;
  flex: 0 0 36px;
  border-radius: 50%;
  background: color-mix(in srgb, var(--rebate) 10%, transparent);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--f-mono);
  font-size: 12px;
  color: var(--ink-2);
  text-decoration: none;
}

/* ── Compact icon buttons ──────────────────────────────────────────── */
.icon-btn {
  position: relative;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  border-radius: 50%;
  border: 0;
  background: color-mix(in srgb, var(--rebate) 8%, transparent);
  color: var(--ink-2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}
</style>
