<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import Icon from "@ui/components/Icon.vue";
import {
  addTag,
  clearCollection,
  deriveGhostTag,
  effectiveCollection,
  effectiveTags,
  pickCollection,
  removeTag,
  stripTagHash,
  suggestTags,
  validateNewTag,
  type FileUnderCollectionLike,
  type FileUnderState,
} from "@studio/lib/fileUnder";
import { collectionSlug } from "@studio/lib/libraryOrganization";
import type { TagCount } from "../../lib/api/types";
import { previewPrintFilename, previewSequenceFilename } from "../../lib/gallery/printFilename";

/**
 * "File under" — the Create inspector's Library filing group, between the
 * essentials and Advanced.
 *
 * Everything here is presentation over `@studio/lib/fileUnder`'s reducers:
 * the ghost chip and the collection match are DERIVED from the live title on
 * every render, so editing the title in the Create header moves them without a
 * watcher. Nothing in this component creates a collection — a picked name is
 * resolved (or created) by the routed host at develop time, which is what lets
 * one draft file correctly on any machine in the fleet.
 */
const props = defineProps<{
  /** The Create header's live print title. */
  title: string;
  /** The filing draft; every edit emits a NEW state object. */
  state: FileUnderState;
  /** Settings ▸ Library "Tag new prints with their title". */
  autoTagTitle: boolean;
  /** Fleet tags with counts, merged across connected hosts. */
  tags: readonly TagCount[];
  /** Fleet collections, merged by slug. */
  collections: readonly (FileUnderCollectionLike & { count?: number })[];
  /** Resolved model id — the filename preview's middle segment. */
  model: string;
  /** Output extension for the preview (`png`, `mp4`, …). */
  extension: string;
  /** Batch size, so a multi-print batch previews its `-{index}` segment. */
  batchSize?: number;
  /** A sequence's stitched print lands under the chain grammar, not the
   * one-shot one — the preview has to say so. */
  outputKind?: "print" | "sequence";
}>();

const emit = defineEmits<{ "update:state": [state: FileUnderState] }>();

// ── Tags ───────────────────────────────────────────────────────────────────

const ghost = computed(() => deriveGhostTag(props.title, props.autoTagTitle));
const ghostVisible = computed(() => ghost.value !== null && !props.state.ghostRemoved);
/** Every chip currently filed, ghost first — the validation gate's context. */
const activeTags = computed(() => effectiveTags(props.state, props.title, props.autoTagTitle));

const tagDraft = ref("");
const tagError = ref<string | null>(null);
const suggestOpen = ref(false);
const tagAnchor = ref<HTMLElement | null>(null);

const suggestions = computed(() =>
  suggestTags(props.tags, tagDraft.value, activeTags.value).slice(0, 8),
);

function commitTag(raw: string, { fromSuggestion = false } = {}) {
  // A TYPED `#kodak` is the display habit and files as `kodak`; a SUGGESTION
  // is a tag the host actually reported, so `#grain` files as `#grain`.
  const name = fromSuggestion ? raw : stripTagHash(raw);
  if (!name.trim()) {
    tagError.value = null;
    return;
  }
  const problem = validateNewTag(name, activeTags.value);
  if (problem) {
    tagError.value = problem;
    return;
  }
  tagError.value = null;
  tagDraft.value = "";
  suggestOpen.value = false;
  emit("update:state", addTag(props.state, name));
}

function onTagEnter() {
  commitTag(tagDraft.value);
}

function dropTag(name: string) {
  emit("update:state", removeTag(props.state, name, props.title, props.autoTagTitle));
}

/** Retire the derived chip. `removeTag` records the opt-out so the title can
 * keep re-deriving the same slug without the chip coming back. */
function dropGhost() {
  if (ghost.value) dropTag(ghost.value);
}

watch(tagDraft, () => {
  tagError.value = null;
  if (tagDraft.value !== "") suggestOpen.value = true;
});

// ── Collection ─────────────────────────────────────────────────────────────

const collectionMenuOpen = ref(false);
const collectionAnchor = ref<HTMLElement | null>(null);
const newCollectionOpen = ref(false);
const newCollectionName = ref("");

const chosen = computed(() => effectiveCollection(props.state, props.title, props.collections));
const chosenSlug = computed(() => chosen.value?.slug ?? null);

function slugOf(entry: FileUnderCollectionLike): string {
  const slug = entry.slug?.trim();
  return slug && slug.length > 0 ? slug : collectionSlug(entry.name);
}

function countFor(entry: FileUnderCollectionLike & { count?: number }): number | null {
  return typeof entry.count === "number" ? entry.count : null;
}

function choose(entry: FileUnderCollectionLike) {
  closeCollectionMenu();
  emit(
    "update:state",
    pickCollection(props.state, {
      ...(entry.id !== undefined ? { id: entry.id } : {}),
      name: entry.name,
    }),
  );
}

function chooseNone() {
  closeCollectionMenu();
  emit("update:state", clearCollection(props.state, props.title));
}

function clearChosen() {
  emit("update:state", clearCollection(props.state, props.title));
}

function commitNewCollection() {
  const name = newCollectionName.value.trim();
  if (!name) return;
  // Match an existing row by slug so "smurfs" doesn't queue a second
  // collection beside "Smurfs"; the host resolves by slug anyway.
  const slug = collectionSlug(name);
  const existing = props.collections.find((entry) => slugOf(entry) === slug);
  choose(existing ?? { name });
}

function closeCollectionMenu() {
  collectionMenuOpen.value = false;
  newCollectionOpen.value = false;
  newCollectionName.value = "";
}

function toggleCollectionMenu() {
  if (collectionMenuOpen.value) closeCollectionMenu();
  else collectionMenuOpen.value = true;
}

function openNewCollection() {
  newCollectionOpen.value = true;
}

// ── Filename preview ───────────────────────────────────────────────────────

// A grammar preview, not a promise: the host stamps the real timestamp when
// the print lands, so one wall clock read at mount is honest enough.
const previewStamp = Date.now();
const filenamePreview = computed(() =>
  props.outputKind === "sequence"
    ? previewSequenceFilename(props.title)
    : previewPrintFilename({
        model: props.model,
        timestamp: previewStamp,
        ext: props.extension,
        title: props.title,
        batchSize: props.batchSize ?? 1,
        index: 0,
      }),
);
const previewSlug = computed(() => {
  const marker = filenamePreview.value.lastIndexOf("~");
  return marker < 0 ? null : filenamePreview.value.slice(marker);
});
const previewStem = computed(() =>
  previewSlug.value === null
    ? filenamePreview.value
    : filenamePreview.value.slice(0, filenamePreview.value.lastIndexOf("~")),
);

// ── Dismissal ──────────────────────────────────────────────────────────────

function onDocumentPointerDown(event: PointerEvent) {
  const path = event.composedPath();
  if (suggestOpen.value && tagAnchor.value && !path.includes(tagAnchor.value)) {
    suggestOpen.value = false;
  }
  if (
    collectionMenuOpen.value &&
    collectionAnchor.value &&
    !path.includes(collectionAnchor.value)
  ) {
    closeCollectionMenu();
  }
}
function onDocumentKeydown(event: KeyboardEvent) {
  if (event.key !== "Escape") return;
  suggestOpen.value = false;
  closeCollectionMenu();
}
onMounted(() => {
  document.addEventListener("pointerdown", onDocumentPointerDown);
  document.addEventListener("keydown", onDocumentKeydown);
});
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onDocumentPointerDown);
  document.removeEventListener("keydown", onDocumentKeydown);
});
</script>

<template>
  <div class="ms-field ms-fu" data-test="file-under-group">
    <div class="ms-fu__title">
      <Icon name="tag" :size="12" />
      File under
    </div>

    <!-- Tags -->
    <div ref="tagAnchor" class="ms-fu__anchor">
      <div class="ms-fu__tags">
        <span
          v-if="ghostVisible"
          class="ms-fu__chip ms-fu__chip--ghost"
          data-test="file-under-ghost-tag"
          title="Default tag from the title — remove to opt this print out"
        >
          {{ ghost }}
          <span class="ms-fu__chip-src">from title</span>
          <button
            type="button"
            class="ms-fu__chip-x"
            data-test="file-under-ghost-remove"
            :aria-label="`Remove the ${ghost} tag`"
            @click="dropGhost"
          >
            ✕
          </button>
        </span>
        <span
          v-for="tag in state.manualTags"
          :key="tag"
          class="ms-fu__chip"
          data-test="file-under-tag"
        >
          {{ tag }}
          <button
            type="button"
            class="ms-fu__chip-x"
            data-test="file-under-tag-remove"
            :aria-label="`Remove the ${tag} tag`"
            @click="dropTag(tag)"
          >
            ✕
          </button>
        </span>
        <input
          v-model="tagDraft"
          type="text"
          class="ms-fu__tag-input"
          data-test="file-under-tag-input"
          placeholder="Add tag…"
          aria-label="Add tag"
          @focus="suggestOpen = true"
          @keydown.enter.prevent="onTagEnter"
        />
      </div>
      <p v-if="tagError" class="ms-field__error" data-test="file-under-tag-error" role="alert">
        {{ tagError }}
      </p>
      <div
        v-if="suggestOpen && suggestions.length > 0"
        class="ms-fu__pop"
        data-test="file-under-tag-suggestions"
      >
        <button
          v-for="tag in suggestions"
          :key="tag.name"
          type="button"
          class="ms-fu__row"
          data-test="file-under-tag-suggestion"
          @click="commitTag(tag.name, { fromSuggestion: true })"
        >
          <span class="ms-fu__row-name">{{ tag.name }}</span>
          <span class="data-mono ms-fu__row-count">{{ tag.count }}</span>
        </button>
        <p class="ms-fu__pop-foot">↵ adds · new names are created on develop</p>
      </div>
    </div>

    <!-- Collection -->
    <div ref="collectionAnchor" class="ms-fu__anchor">
      <!-- A field box, not a button: the clear ✕ is its own control and may
           not nest inside one. The box forwards stray clicks to the toggle. -->
      <div class="ms-fu__collection" @click="toggleCollectionMenu">
        <button
          type="button"
          class="ms-fu__collection-main"
          data-test="file-under-collection"
          :aria-expanded="collectionMenuOpen"
          aria-label="Collection"
          @click.stop="toggleCollectionMenu"
        >
          <Icon name="collection" :size="14" />
          <span v-if="chosen" class="ms-fu__collection-name">{{ chosen.name }}</span>
          <span v-else class="ms-fu__collection-none">None</span>
          <span
            v-if="chosen?.source === 'title'"
            class="data-mono ms-fu__collection-match"
            data-test="file-under-collection-match"
          >
            · matched to title <span class="ms-fu__collection-tick">✓</span>
          </span>
        </button>
        <button
          v-if="chosen"
          type="button"
          class="ms-fu__chip-x"
          data-test="file-under-collection-clear"
          aria-label="Clear collection"
          title="Clear collection"
          @click.stop="clearChosen"
        >
          ✕
        </button>
        <Icon :name="collectionMenuOpen ? 'chevron-up' : 'chevron-down'" :size="13" />
      </div>
      <div v-if="collectionMenuOpen" class="ms-fu__pop" data-test="file-under-collection-menu">
        <button
          type="button"
          class="ms-fu__row"
          data-test="file-under-collection-none"
          @click="chooseNone"
        >
          <span class="ms-fu__row-tick" />
          <span class="ms-fu__row-name ms-fu__collection-none">None</span>
        </button>
        <button
          v-for="entry in collections"
          :key="slugOf(entry)"
          type="button"
          class="ms-fu__row"
          data-test="file-under-collection-option"
          @click="choose(entry)"
        >
          <span class="ms-fu__row-tick">{{ slugOf(entry) === chosenSlug ? "✓" : "" }}</span>
          <span class="ms-fu__row-name">{{ entry.name }}</span>
          <span v-if="countFor(entry) !== null" class="data-mono ms-fu__row-count">
            {{ countFor(entry) }}
          </span>
        </button>
        <button
          v-if="!newCollectionOpen"
          type="button"
          class="ms-fu__row ms-fu__row--new"
          data-test="file-under-new-collection"
          @click="openNewCollection"
        >
          <Icon name="plus" :size="13" />
          <span class="ms-fu__row-name">New collection…</span>
        </button>
        <div v-else class="ms-fu__row ms-fu__row--new">
          <Icon name="plus" :size="13" />
          <input
            v-model="newCollectionName"
            type="text"
            class="ms-fu__new-input"
            data-test="file-under-new-collection-input"
            placeholder="Collection name"
            aria-label="New collection name"
            @keydown.enter.prevent="commitNewCollection"
            @keydown.esc.prevent="closeCollectionMenu"
          />
        </div>
        <p class="ms-fu__pop-foot">
          {{
            newCollectionOpen
              ? "↵ creates and selects it · Esc cancels"
              : "created on the machine that develops this print"
          }}
        </p>
      </div>
    </div>

    <p class="data-mono ms-fu__filename" data-test="file-under-filename">
      <span class="ms-fu__filename-key">files as</span>
      {{ previewStem }}<b v-if="previewSlug" class="ms-fu__filename-slug">{{ previewSlug }}</b>
    </p>
  </div>
</template>

<style scoped>
/* One card, because the two rows and the preview are a single decision. */
.ms-fu {
  display: flex;
  flex-direction: column;
  gap: 8px;
  border: 1px solid var(--edge);
  border-radius: 8px;
  background: color-mix(in srgb, var(--bench) 55%, var(--bath));
  padding: 10px;
}
.ms-fu__title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  color: var(--safelight);
}
.ms-fu__anchor {
  position: relative;
}
.ms-fu__tags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px;
  min-height: 34px;
  border: 1px solid var(--ce);
  border-radius: 9px;
  background: var(--bath);
  padding: 5px 7px;
}
.ms-fu__chip {
  display: inline-flex;
  min-height: 22px;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--edge);
  border-radius: var(--radius-pill);
  background: var(--bench);
  padding: 1px 5px 1px 8px;
  font-size: 11.5px;
  color: var(--ink-2);
  max-width: 100%;
  overflow-wrap: anywhere;
}
/* Dashed = derived, not typed. The chip is removable, and its removal sticks
   for this draft even as the title keeps re-deriving the same slug. */
.ms-fu__chip--ghost {
  border-style: dashed;
  border-color: var(--ce);
  background: transparent;
}
.ms-fu__chip-src {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  color: var(--ink-3);
}
.ms-fu__chip-x {
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-size: 10px;
  line-height: 1;
  cursor: pointer;
  padding: 0 1px;
}
.ms-fu__chip-x:hover {
  color: var(--rebate);
}
.ms-fu__tag-input {
  flex: 1 1 90px;
  min-width: 80px;
  border: 0;
  background: transparent;
  font-size: 12px;
  color: var(--rebate);
  padding: 2px 0;
}
.ms-fu__tag-input:focus {
  outline: none;
}
.ms-fu__collection {
  display: flex;
  width: 100%;
  min-height: 34px;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--ce);
  border-radius: 9px;
  background: var(--bath);
  padding: 0 9px;
  font-size: 12px;
  color: var(--rebate);
  cursor: pointer;
}
.ms-fu__collection-main {
  display: flex;
  min-width: 0;
  flex: 1;
  align-items: center;
  gap: 6px;
  border: 0;
  background: transparent;
  color: inherit;
  font-size: inherit;
  text-align: left;
  cursor: pointer;
  padding: 0;
}
.ms-fu__collection-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-fu__collection-none {
  color: var(--ink-3);
}
.ms-fu__collection-tick {
  color: var(--safelight);
  font-weight: 700;
}
.ms-fu__collection-match {
  flex-shrink: 0;
  font-size: 9px;
  color: var(--ink-3);
}
/* Anchored to the group, never to the small input: `left/right: 0` keeps the
   popover inside the inspector at every panel width. */
.ms-fu__pop {
  position: absolute;
  top: calc(100% + 5px);
  left: 0;
  right: 0;
  z-index: 30;
  max-height: 16rem;
  overflow-y: auto;
  border: 1px solid var(--edge);
  border-radius: 10px;
  background: var(--bench);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
  padding: 5px 0;
}
.ms-fu__row {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  text-align: left;
  font-size: 12px;
  color: var(--ink-2);
  background: transparent;
  border: 0;
  cursor: pointer;
}
.ms-fu__row:hover {
  background: var(--bath);
  color: var(--rebate);
}
.ms-fu__row--new {
  border-top: 1px solid var(--edge);
  margin-top: 4px;
  padding-top: 7px;
  color: var(--halide);
}
.ms-fu__row-tick {
  width: 10px;
  flex-shrink: 0;
  color: var(--safelight);
  font-size: 10px;
}
.ms-fu__row-name {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-fu__row-count {
  flex-shrink: 0;
  font-size: 10px;
  color: var(--ink-3);
}
.ms-fu__new-input {
  flex: 1;
  min-width: 0;
  border: 0;
  background: transparent;
  font-size: 12px;
  color: var(--rebate);
}
.ms-fu__new-input:focus {
  outline: none;
}
.ms-fu__pop-foot {
  padding: 4px 10px 2px;
  font-size: 9px;
  color: var(--ink-3);
}
/* Dashed, like the ghost chip: a preview of a name the host will stamp. */
.ms-fu__filename {
  border: 1px dashed var(--ce);
  border-radius: 6px;
  background: var(--bath);
  padding: 6px 8px;
  font-size: 10.5px;
  line-height: 1.6;
  color: var(--ink-2);
  overflow-wrap: anywhere;
}
.ms-fu__filename-key {
  margin-right: 5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 8.5px;
  color: var(--ink-3);
}
.ms-fu__filename-slug {
  color: var(--safelight);
  font-weight: 600;
}
</style>
