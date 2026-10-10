<script setup lang="ts">
import { computed, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { scopedCollectionInventory } from "@studio/lib/collectionVisibility";
import { useContextMenuStore } from "../../stores/contextMenu";
import { useGalleryStore, type LibraryScope } from "../../stores/gallery";
const gallery = useGalleryStore();
const menu = useContextMenuStore();
const route = useRoute();
const router = useRouter();
function readExpanded(key: string) {
  try {
    return localStorage.getItem(key) !== "closed";
  } catch {
    return true;
  }
}
const expanded = ref(readExpanded("mold.library.navigation"));
const collectionsExpanded = ref(readExpanded("mold.library.collections"));
const collections = computed(() =>
  scopedCollectionInventory(gallery.mergedCollections, gallery.filter),
);
function toggle(which: "library" | "collections") {
  const state = which === "library" ? expanded : collectionsExpanded;
  state.value = !state.value;
  try {
    localStorage.setItem(
      `mold.library.${which === "library" ? "navigation" : "collections"}`,
      state.value ? "open" : "closed",
    );
  } catch {
    /* Expansion remains usable for this session. */
  }
  if (state.value) void gallery.fetchCollections();
}
function navigate(scope: LibraryScope, slug?: string, action?: string) {
  void router.push({ path: "/library", query: { host: route.query.host, scope, c: slug, action } });
}
function manage(event: MouseEvent, slug: string, hidden: boolean) {
  menu.open(event, [
    { label: "Rename…", action: () => navigate("collections", slug, "rename") },
    {
      label: hidden ? "Show in My images" : "Hide from My images",
      action: () => navigate("collections", slug, "visibility"),
    },
    { separator: true },
    {
      label: "Delete collection…",
      danger: true,
      action: () => navigate("collections", slug, "delete"),
    },
  ]);
}
function active(scope: LibraryScope) {
  return (
    route.path === "/library" &&
    (route.query.scope ?? "prints") === scope &&
    (scope !== "collections" || !route.query.c)
  );
}
function drop(event: DragEvent, slug: string) {
  event.preventDefault();
  const raw = event.dataTransfer?.getData("application/x-mold-library-prints");
  if (!raw) return;
  // In-process event carries only keys. The view re-resolves its current selection.
  window.dispatchEvent(
    new CustomEvent("mold-file-library-prints", { detail: { slug, keys: raw } }),
  );
}
</script>
<template>
  <div class="library-navigation text-xs" aria-label="Library sections">
    <button
      class="library-group ms-toolbar-button w-full text-left"
      :aria-expanded="expanded"
      @click="toggle('library')"
    >
      {{ expanded ? "▾" : "▸" }} Library
    </button>
    <div v-if="expanded" class="library-children">
      <button
        class="ms-toolbar-button w-full text-left"
        :aria-current="active('prints') ? 'page' : undefined"
        @click="navigate('prints')"
      >
        Prints <span class="ml-auto">{{ gallery.basePrintCount }}</span>
      </button>
      <button
        v-if="gallery.anyOrganizeCapable"
        class="ms-toolbar-button w-full text-left"
        :aria-current="active('favorites') ? 'page' : undefined"
        @click="navigate('favorites')"
      >
        Favorites
      </button>
      <div v-if="gallery.anyOrganizeCapable" class="flex">
        <button
          class="ms-toolbar-button"
          aria-label="Expand collections"
          :aria-expanded="collectionsExpanded"
          @click="toggle('collections')"
        >
          {{ collectionsExpanded ? "▾" : "▸" }}
        </button>
        <button
          class="ms-toolbar-button flex-1 text-left"
          :aria-current="active('collections') ? 'page' : undefined"
          @click="navigate('collections')"
        >
          Collections
        </button>
        <button
          class="ms-toolbar-button"
          aria-label="New collection"
          @click="navigate('collections', undefined, 'new')"
        >
          +
        </button>
      </div>
      <div
        v-if="collectionsExpanded && gallery.anyOrganizeCapable"
        class="library-collections max-h-[24vh] overflow-y-auto"
      >
        <div
          v-for="collection in collections"
          :key="collection.slug"
          class="flex items-center gap-1"
          @dragover.prevent
          @drop="drop($event, collection.slug)"
        >
          <button
            class="ms-toolbar-button min-w-0 flex-1"
            :aria-current="route.query.c === collection.slug ? 'page' : undefined"
            :class="route.query.c === collection.slug ? 'text-accent' : ''"
            :title="`${collection.name}${collection.hidden ? ' · Hidden' : ''} · ${gallery.collectionAvailability(collection.slug)}`"
            @click="navigate('collections', collection.slug)"
          >
            <span class="truncate">{{ collection.hidden ? "◌ " : "" }}{{ collection.name }}</span>
            <span class="ml-auto font-mono">{{
              gallery.collectionAvailability(collection.slug) === "present"
                ? gallery.collectionCounts(collection.slug)
                : gallery.collectionAvailability(collection.slug) === "absent"
                  ? "Not on machine"
                  : "Unavailable"
            }}</span>
          </button>
          <button
            class="library-more ms-toolbar-button"
            :aria-label="`Manage ${collection.name}`"
            @click="manage($event, collection.slug, collection.hidden === true)"
          >
            ⋯
          </button>
        </div>
      </div>
      <button
        class="ms-toolbar-button w-full text-left"
        :aria-current="active('sources') ? 'page' : undefined"
        @click="navigate('sources')"
      >
        Source Media
      </button>
      <button
        v-if="gallery.anyTrashCapable"
        class="ms-toolbar-button w-full text-left"
        :aria-current="active('trash') ? 'page' : undefined"
        @click="navigate('trash')"
      >
        Trash <span class="ml-auto">{{ gallery.trashCount }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.library-navigation {
  margin: 0.35rem 0 0;
}
.library-navigation .ms-toolbar-button {
  border-color: transparent;
  background: transparent;
  box-shadow: none;
  min-height: 1.85rem;
  color: var(--mold-text-dim);
  font-weight: 400;
  padding-inline: 0.55rem;
  gap: 0.45rem;
}
.library-navigation .ms-toolbar-button:hover {
  background: var(--mold-surface-2);
  color: var(--mold-text);
}
.library-navigation .ms-toolbar-button[aria-current="page"] {
  background: color-mix(in srgb, var(--mold-blue) 12%, transparent);
  color: var(--mold-blue);
  font-weight: 600;
}
.library-navigation .library-group {
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--mold-text-faint);
}
.library-children {
  margin-left: 1.1rem;
  padding-left: 0.3rem;
  border-left: 1px solid var(--mold-border);
}
.library-collections {
  margin-left: 0.65rem;
  padding-left: 0.45rem;
  border-left: 1px solid var(--mold-border);
}
.library-more {
  width: 1.6rem;
  flex-shrink: 0;
}
.library-navigation .font-mono {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-faint);
}
</style>
