<script setup lang="ts">
/*
 * Collections shelf (Library V3 "Shelf" · Collections scope). One card per
 * merged (cross-host) collection: a 2×2 cover mosaic (explicit cover first,
 * else newest prints), display-weight name, a mono meta line with the
 * logical count and the hosts that hold it, and last-updated. The dashed
 * "New collection" card is the one primary action. A card's "…" menu and
 * right-click offer Rename… / Delete collection; the page owns both.
 */
import { ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import { useThumbnailSources } from "../../composables/useThumbnailSources";
import { formatRelativeTime } from "../../util/format";
import type { CollectionCard } from "../../lib/libraryOrganization";

withDefaults(
  defineProps<{
    cards: readonly CollectionCard[];
    /** Some host can create collections. */
    canCreate: boolean;
  }>(),
  {},
);

const emit = defineEmits<{
  (e: "open", slug: string): void;
  (e: "new"): void;
  (e: "rename", slug: string): void;
  (e: "delete", slug: string): void;
}>();

const { srcFor } = useThumbnailSources();
const menuFor = ref<string | null>(null);

function toggleMenu(slug: string, event: Event) {
  event.stopPropagation();
  menuFor.value = menuFor.value === slug ? null : slug;
}
function closeMenu() {
  menuFor.value = null;
}
function metaLine(card: CollectionCard): string {
  const count = `${card.count} ${card.count === 1 ? "print" : "prints"}`;
  return [count, ...card.hostLabels].join(" · ");
}
</script>

<template>
  <section class="shelf" aria-label="Collections" @click="closeMenu">
    <p class="shelf__kicker">
      Collections
      <span class="shelf__kicker-note">merged across hosts by name</span>
    </p>
    <div class="shelf__grid">
      <article
        v-for="card in cards"
        :key="card.slug"
        class="ccard"
        data-test="collection-card"
        :data-slug="card.slug"
        @contextmenu.prevent="menuFor = card.slug"
      >
        <button
          type="button"
          class="ccard__open"
          :aria-label="`Open collection ${card.name}`"
          data-test="collection-open"
          @click="emit('open', card.slug)"
        >
          <span class="ccard__mosaic" :data-n="Math.min(card.covers.length, 4)">
            <img
              v-for="cover in card.covers.slice(0, 4)"
              :key="`${cover.hostId}|${cover.filename}`"
              :src="srcFor(cover)"
              alt=""
              loading="lazy"
              decoding="async"
            />
            <span v-if="card.covers.length === 0" class="ccard__blank">
              <Icon name="collection" :size="22" :stroke-width="1.5" />
            </span>
          </span>
          <span class="ccard__name" data-test="collection-name">{{
            card.name
          }}</span>
          <span class="ccard__meta" data-test="collection-meta">{{
            metaLine(card)
          }}</span>
          <span class="ccard__upd">
            {{
              card.updatedAt
                ? `Updated ${formatRelativeTime(card.updatedAt)}`
                : " "
            }}
          </span>
        </button>
        <button
          type="button"
          class="ccard__more"
          :aria-expanded="menuFor === card.slug"
          aria-haspopup="menu"
          :aria-label="`Collection actions for ${card.name}`"
          data-test="collection-menu"
          @click="toggleMenu(card.slug, $event)"
        >
          <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <circle cx="5" cy="12" r="1.6" />
            <circle cx="12" cy="12" r="1.6" />
            <circle cx="19" cy="12" r="1.6" />
          </svg>
        </button>
        <div
          v-if="menuFor === card.slug"
          class="ccard__menu"
          role="menu"
          data-test="collection-card-menu"
          @click.stop
        >
          <button
            type="button"
            role="menuitem"
            data-test="collection-rename"
            @click="
              closeMenu();
              emit('rename', card.slug);
            "
          >
            Rename…
          </button>
          <button
            type="button"
            role="menuitem"
            class="ccard__menu-danger"
            data-test="collection-delete"
            @click="
              closeMenu();
              emit('delete', card.slug);
            "
          >
            Delete collection…
          </button>
        </div>
      </article>

      <button
        v-if="canCreate"
        type="button"
        class="ccard ccard--new"
        data-test="collection-new-card"
        @click="emit('new')"
      >
        <span class="ccard__mosaic ccard__mosaic--dashed">
          <Icon name="plus" :size="22" />
        </span>
        <span class="ccard__name">New collection</span>
        <span class="ccard__meta"><kbd class="ccard__kbd">⌘⇧N</kbd></span>
        <span class="ccard__upd"
          >Name it, then add prints from the grid or a selection.</span
        >
      </button>
    </div>
  </section>
</template>

<style scoped>
.shelf {
  padding-top: 4px;
}
.shelf__kicker {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin: 0 0 12px;
  font-family: var(--f-display);
  font-size: 13px;
  font-weight: 600;
  color: var(--rebate);
}
.shelf__kicker-note {
  font-family: var(--f-mono);
  font-size: 10px;
  font-weight: 400;
  color: var(--ink-3);
}
.shelf__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 14px;
}
.ccard {
  position: relative;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  background: var(--bench);
  box-shadow: inset 0 1px 0 var(--card-hi);
  text-align: left;
  transition:
    transform var(--dur-quick) var(--ease),
    box-shadow var(--dur-quick) var(--ease);
}
.ccard:hover {
  transform: translateY(-2px);
  box-shadow:
    inset 0 1px 0 var(--card-hi),
    0 10px 24px rgba(0, 0, 0, 0.18);
}
.ccard__open,
.ccard--new {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 100%;
  padding: 10px 12px 12px;
  border: 0;
  border-radius: inherit;
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.ccard--new {
  border: 1px dashed var(--ce);
  background: transparent;
  color: var(--rebate);
}
.ccard__open:focus-visible,
.ccard--new:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.ccard__mosaic {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 2px;
  aspect-ratio: 1;
  margin-bottom: 6px;
  overflow: hidden;
  border-radius: 8px;
  background: var(--print);
}
.ccard__mosaic img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.ccard__mosaic[data-n="1"] {
  grid-template-columns: 1fr;
  grid-template-rows: 1fr;
}
.ccard__mosaic[data-n="2"] {
  grid-template-rows: 1fr;
}
.ccard__mosaic[data-n="3"] img:first-child {
  grid-row: span 2;
}
.ccard__mosaic--dashed {
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: 1.5px dashed var(--ce);
  color: var(--ink-3);
}
.ccard__blank {
  grid-column: 1 / -1;
  grid-row: 1 / -1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--ink-3);
}
.ccard__name {
  font-family: var(--f-display);
  font-size: 15px;
  font-weight: 600;
  color: var(--rebate);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ccard__meta {
  font-family: var(--f-mono);
  font-size: 10.5px;
  color: var(--ink-2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ccard__upd {
  font-size: 11px;
  color: var(--ink-3);
}
.ccard__kbd {
  display: inline-flex;
  align-items: center;
  padding: 1px 5px;
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-sm);
  font: 10px var(--f-mono);
  color: var(--ink-3);
}
.ccard__more {
  position: absolute;
  top: 14px;
  right: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 0;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.5);
  color: var(--on-media);
  opacity: 0;
  cursor: pointer;
  transition: opacity var(--dur-quick) var(--ease);
}
.ccard__more svg {
  width: 16px;
  height: 16px;
}
.ccard:hover .ccard__more,
.ccard__more:focus-visible,
.ccard__more[aria-expanded="true"] {
  opacity: 1;
}
.ccard__menu {
  position: absolute;
  top: 46px;
  right: 16px;
  z-index: 5;
  display: grid;
  min-width: 170px;
  padding: 6px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control-lg);
  background: var(--bench);
  box-shadow: var(--shadow-raised);
}
.ccard__menu button {
  min-height: 34px;
  padding: 0 10px;
  border: 0;
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--rebate);
  font: inherit;
  font-size: 13px;
  text-align: left;
  cursor: pointer;
}
.ccard__menu button:hover {
  background: var(--sel-bg);
}
.ccard__menu-danger {
  color: var(--stop) !important;
}
</style>
