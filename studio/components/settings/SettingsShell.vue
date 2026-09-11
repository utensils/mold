<script setup lang="ts">
/*
 * The Settings frame, shared by web and desktop: a search field and a jump nav
 * beside one scroll of always-open sections. Search narrows the nav and the
 * page together; the caller renders each body through the `#section` slot and
 * owns its own `?section=` deep link — the shell only exposes `jump`.
 *
 * Ported from `desktop/src/views/SettingsView.vue`, whose two observers, 800ms
 * settling hold and idempotent `bindSection` each answer a bug that actually
 * happened.
 *
 * Two layouts, one frame. `layout="scroll"` is the desktop's: every section
 * open on one scroll, the nav a scroll-spy. `layout="pane"` is the browser's:
 * a page that scrolls the window cannot hold fifteen open sections without
 * becoming seven screens long — which is the complaint this kit answers — so
 * the nav is a real navigation, one section on screen at a time, and search
 * stacks every match so a word still finds its row wherever it lives.
 *
 * `scroll` says who scrolls: the page (a browser window) or the content column
 * (the desktop's fixed pane). The scroll-spy observes against whichever it is,
 * because a root that moves with its targets never reports a change.
 *
 * At 900px and up the nav is a sticky 200px column; below, it folds into a
 * horizontally scrolling chip strip, because a browser page has no fixed
 * second pane.
 */
import {
  computed,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
  type ComponentPublicInstance,
} from "vue";
import {
  sectionMatchesSearch,
  type SectionId,
  type SectionInfo,
} from "../../lib/settingsSchema";
import SettingsSection from "./SettingsSection.vue";

const props = withDefaults(
  defineProps<{
    sections: readonly SectionInfo[];
    /** Raw engine rows each section renders, so search can match them. */
    rawKeysBySection?: Partial<Record<SectionId, string[]>>;
    searchPlaceholder?: string;
    /** `scroll`: every section open on one scroll, the nav a scroll-spy.
     *  `pane`: one section on screen at a time, the nav a navigation. */
    layout?: "scroll" | "pane";
    /** Who scrolls: the page (a browser window) or this column (a fixed pane). */
    scroll?: "page" | "content";
  }>(),
  {
    rawKeysBySection: () => ({}),
    searchPlaceholder: "Search settings…",
    layout: "scroll",
    scroll: "page",
  },
);
const emit = defineEmits<{ (e: "update:active", id: SectionId): void }>();

const query = ref("");
const searching = computed(() => query.value.trim().length > 0);

const paned = computed(() => props.layout === "pane");

/** The nav's highlighted section: the one at the top of the page, or the one
 *  last jumped to while that scroll is still settling — or, in a pane, the
 *  one on screen. */
const active = ref<SectionId>(props.sections[0]?.id ?? "app");

/** The sections that match the search; every section when there is none. */
const matchingSections = computed(() =>
  props.sections.filter(
    (section) =>
      !searching.value ||
      sectionMatchesSearch(query.value, section, props.rawKeysBySection),
  ),
);

/** The nav lists every match; the page shows them all on a scroll, and only
 *  the active one in a pane (a search stacks its matches there too). */
const visibleSections = computed(() =>
  paned.value && !searching.value
    ? matchingSections.value.filter((section) => section.id === active.value)
    : matchingSections.value,
);
const sectionEls = new Map<SectionId, HTMLElement>();
const contentEl = ref<HTMLElement | null>(null);
let observer: IntersectionObserver | null = null;
let bodyObserver: IntersectionObserver | null = null;
let settling: ReturnType<typeof setTimeout> | null = null;

watch(active, (id) => emit("update:active", id));

/**
 * Which section bodies have been reached. A body is a live component — Advanced
 * alone opens three HTTP calls and a device subscription on mount, and most
 * visits never scroll to it. A body arrives well before it is looked at and
 * then STAYS, so scrolling back is never a second fetch and a half-typed edit
 * is never thrown away.
 */
const reached = ref<SectionId[]>([]);
function reach(id: SectionId) {
  if (!reached.value.includes(id)) reached.value.push(id);
}

/** While searching, the matches ARE the page: the user asked for them by name,
 *  and there is nothing to scroll past. A pane's one section is on screen. */
function bodyMounted(id: SectionId): boolean {
  return (
    searching.value ||
    (paned.value && id === active.value) ||
    reached.value.includes(id)
  );
}
watch(visibleSections, (sections) => {
  if (searching.value || paned.value)
    for (const section of sections) reach(section.id);
});

/** Vue re-invokes a function `:ref` on EVERY patch of its element, so this must
 *  be idempotent: re-registering every section on each keystroke in the search
 *  field is what made the nav highlight flicker. */
function bindSection(
  id: SectionId,
  el: Element | ComponentPublicInstance | null,
) {
  const previous = sectionEls.get(id);
  const next = el instanceof HTMLElement ? el : null;
  if (previous === next) return;
  if (previous) {
    observer?.unobserve(previous);
    bodyObserver?.unobserve(previous);
  }
  if (next) {
    sectionEls.set(id, next);
    next.dataset.section = id;
    observer?.observe(next);
    bodyObserver?.observe(next);
  } else sectionEls.delete(id);
}

/** One stable `:ref` callback per section. An inline arrow is a NEW function
 *  every render, which Vue treats as a changed ref. */
const sectionBinders = new Map<
  SectionId,
  (el: Element | ComponentPublicInstance | null) => void
>();
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
  // The scroll needs something to land on, so the body comes first.
  reach(id);
  // A pane swaps; there is no scroll to settle.
  if (paned.value) return;
  // A smooth scroll passes other sections on its way; hold the pick until it
  // lands, or the highlight races down the nav.
  if (settling) clearTimeout(settling);
  settling = setTimeout(() => (settling = null), 800);
  sectionEls.get(id)?.scrollIntoView?.({ behavior: "smooth", block: "start" });
}

onMounted(() => {
  // A pane shows what it is told to; nothing scrolls into view.
  if (paned.value) return;
  if (typeof IntersectionObserver === "undefined") {
    // No observer, no scroll signal: an eager page beats an empty one.
    for (const section of props.sections) reach(section.id);
    return;
  }
  // The root is whoever scrolls. A root that moves with its targets never
  // reports a change, which is how the highlight froze on a browser page.
  const root = props.scroll === "content" ? contentEl.value : null;
  // Two observers, two questions. The scroll-spy's band is the top of the page,
  // which is where the nav highlight belongs; a body has to arrive WELL before
  // it is looked at, so it gets its own generous margin.
  bodyObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const id = (entry.target as HTMLElement).dataset.section as
          SectionId | undefined;
        if (id) reach(id);
      }
    },
    { root, rootMargin: "400px 0px 800px 0px" },
  );
  observer = new IntersectionObserver(
    (entries) => {
      if (settling) return;
      const top = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      const id = (top?.target as HTMLElement | undefined)?.dataset.section as
        SectionId | undefined;
      if (id && visibleSections.value.some((section) => section.id === id))
        active.value = id;
    },
    { root, rootMargin: "0px 0px -70% 0px" },
  );
  for (const el of sectionEls.values()) {
    observer.observe(el);
    bodyObserver.observe(el);
  }
});

onBeforeUnmount(() => {
  observer?.disconnect();
  bodyObserver?.disconnect();
  if (settling) clearTimeout(settling);
});

defineExpose({ active, jump, query });
</script>

<template>
  <div
    class="ms-settings-shell"
    :class="{
      'ms-settings-shell--own-scroll': scroll === 'content',
      'ms-settings-shell--pane': paned,
    }"
  >
    <nav class="ms-settings-nav" aria-label="Settings sections">
      <label class="ms-settings-nav__search">
        <input
          v-model="query"
          data-selectable
          data-test="settings-search"
          type="search"
          aria-label="Search settings"
          :placeholder="searchPlaceholder"
        />
      </label>
      <div class="ms-settings-nav__rows">
        <button
          v-for="section in matchingSections"
          :key="section.id"
          type="button"
          class="ms-settings-nav__row"
          :class="{ 'ms-settings-nav__row--active': active === section.id }"
          :data-test="`settings-nav-${section.id}`"
          :aria-current="active === section.id ? 'true' : undefined"
          @click="jump(section.id)"
        >
          {{ section.label }}
        </button>
      </div>
      <slot name="nav-extra" />
    </nav>

    <div ref="contentEl" class="ms-settings-content">
      <section
        v-for="section in visibleSections"
        :key="section.id"
        :ref="sectionBinder(section.id)"
        :data-test="`section-${section.id}`"
        class="ms-settings-content__section"
      >
        <SettingsSection :label="section.label" :summary="section.summary">
          <slot
            name="section"
            :section="section"
            :mounted="bodyMounted(section.id)"
          />
        </SettingsSection>
      </section>

      <p
        v-if="searching && visibleSections.length === 0"
        class="ms-settings-content__empty"
        data-test="no-search-results"
      >
        Nothing matches “{{ query }}”.
      </p>
    </div>
  </div>
</template>

<style scoped>
.ms-settings-shell {
  display: grid;
  grid-template-columns: var(--mold-shell-settingsnav-w, 200px) minmax(0, 1fr);
  gap: var(--mold-sp-5);
  align-items: start;
}
.ms-settings-nav {
  position: sticky;
  top: 0;
  align-self: start;
  display: flex;
  flex-direction: column;
  gap: var(--mold-sp-1);
  max-height: 100vh;
  overflow-y: auto;
}
.ms-settings-nav__search {
  display: flex;
  align-items: center;
  height: var(--mold-ctl-lg, 32px);
  margin-bottom: var(--mold-sp-2);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
}
.ms-settings-nav__search:focus-within {
  border-color: var(--mold-border-focus);
}
.ms-settings-nav__search input {
  min-width: 0;
  flex: 1 1 auto;
  border: none;
  background: transparent;
  color: var(--mold-text);
  font-size: var(--mold-fs-xs);
  outline: none;
}
.ms-settings-nav__search input::placeholder {
  color: var(--mold-text-dim);
}
.ms-settings-nav__rows {
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.ms-settings-nav__row {
  display: flex;
  align-items: center;
  min-height: var(--mold-row-h, 36px);
  padding: 0 var(--mold-sp-2);
  border: none;
  border-radius: var(--mold-radius-2);
  background: none;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  text-align: left;
  cursor: pointer;
  transition: background-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-settings-nav__row:hover {
  background: var(--mold-row-hover, var(--mold-surface));
}
.ms-settings-nav__row--active {
  background: var(--mold-accent-tint);
  color: var(--mold-text);
}
.ms-settings-content {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: var(--mold-sp-5);
}
.ms-settings-content__section {
  scroll-margin-top: var(--mold-sp-4);
}
/* The desktop's fixed pane: the column is the scroller, and it must be as tall
 * as the frame — a column that never overflows never scrolls. */
.ms-settings-shell--own-scroll {
  align-items: stretch;
  min-height: 0;
  height: 100%;
}
.ms-settings-shell--own-scroll .ms-settings-nav {
  position: static;
  max-height: none;
  min-height: 0;
}
.ms-settings-shell--own-scroll .ms-settings-content {
  min-height: 0;
  overflow-y: auto;
}
.ms-settings-content__empty {
  margin: 0;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
}

/* Below 900px the nav folds into a chip strip above the page: a browser page
 * has no fixed second pane, and a 200px column here would leave the rows too
 * narrow to read their own labels. */
@media (max-width: 899px) {
  /* In a column flex layout the align axis is the HORIZONTAL one; `start`
   * shrink-wraps both children to max-content and the page scrolls sideways. */
  .ms-settings-shell {
    display: flex;
    flex-direction: column;
    gap: var(--mold-sp-4);
    align-items: stretch;
  }
  .ms-settings-nav {
    position: static;
    align-self: stretch;
    min-width: 0;
    max-height: none;
    gap: var(--mold-sp-2);
  }
  .ms-settings-nav__rows {
    flex-direction: row;
    gap: var(--mold-sp-1);
    overflow-x: auto;
    scroll-snap-type: x proximity;
  }
  .ms-settings-nav__row {
    flex: none;
    scroll-snap-align: start;
    white-space: nowrap;
    border: var(--mold-bw) solid var(--mold-border);
  }
}
</style>
