<script setup lang="ts">
/**
 * "File under" — the iPhone Create stack's Library filing group.
 *
 * The same decision the desktop inspector and the web controls region carry,
 * rendered as the phone's own vocabulary: two 44pt rows (Tags, Collection)
 * whose editors live in the existing `MobileLibrarySheet` bottom sheet, with
 * the compact "files as …" preview underneath.
 *
 * Everything is presentation over `@studio/lib/fileUnder`'s reducers: the
 * ghost chip and the title match are DERIVED from the live title on every
 * render, so editing the Create title moves them with no watcher. Nothing
 * here creates a collection — a picked name is resolved (or created) by the
 * routed machine at develop time, which is what lets one draft file correctly
 * on any machine in the fleet.
 */
import { computed, ref, watch } from "vue";
import {
  addTag,
  clearCollection,
  deriveGhostTag,
  effectiveCollection,
  effectiveTags,
  pickCollection,
  removeTag,
  requestTagKey,
  stripTagHash,
  suggestTags,
  validateNewTag,
  type FileUnderCollectionLike,
  type FileUnderState,
} from "@studio/lib/fileUnder";
import { collectionSlug } from "@studio/lib/libraryOrganization";
import type { TagCount } from "../lib/api/types";
import { previewPrintFilename } from "../lib/gallery/printFilename";
import MobileLibrarySheet from "./MobileLibrarySheet.vue";

const props = defineProps<{
  /** The Create stack's live print title. */
  title: string;
  /** The filing draft; every edit emits a NEW state object. */
  state: FileUnderState;
  /** Settings ▸ Library "Tag new prints with their title". */
  autoTagTitle: boolean;
  /** Fleet tags with counts, merged across connected machines. */
  tags: readonly TagCount[];
  /** Fleet collections, merged by slug. */
  collections: readonly (FileUnderCollectionLike & { count?: number })[];
  /** Resolved model id — the filename preview's middle segment. */
  model: string;
  /** Output extension for the preview (`png`, `mp4`, …). */
  extension: string;
  /** Batch size, so a multi-print batch previews its `-{index}` segment. */
  batchSize?: number;
}>();

const emit = defineEmits<{ "update:state": [state: FileUnderState] }>();

/** One sheet at a time — the phone has no room for two. */
const sheet = ref<"tags" | "collection" | null>(null);

// ── Tags ───────────────────────────────────────────────────────────────────

const ghost = computed(() => deriveGhostTag(props.title, props.autoTagTitle));
const ghostVisible = computed(() => ghost.value !== null && !props.state.ghostRemoved);
/** Every chip currently filed, ghost first — the validation gate's context. */
const activeTags = computed(() => effectiveTags(props.state, props.title, props.autoTagTitle));

/**
 * The chips the row draws after the ghost.
 *
 * A manual tag can become case-insensitively equal to the ghost when the title
 * is typed AFTER the tag (`validateNewTag` only guards the other order), and
 * the row would then draw the same name twice. `effectiveTags` already dedupes
 * for the wire and the sheet; the row has to as well, because `removeTag`
 * retires the ghost AND drops the identical manual tag, so a ✕ on either
 * duplicate would delete a tag the user only meant to de-duplicate.
 */
const rowTags = computed(() => {
  const ghostKey = ghostVisible.value && ghost.value ? requestTagKey(ghost.value) : null;
  return props.state.manualTags.filter((tag) => requestTagKey(tag) !== ghostKey);
});

const tagDraft = ref("");
const tagError = ref<string | null>(null);

/** What `Add` would actually file: people type the `#` out of habit. */
const tagDraftName = computed(() => stripTagHash(tagDraft.value).trim());
// Query with the stripped name too — `requestTagKey` keeps a leading `#`, so a
// typed `#kod` would otherwise match neither `kodak` nor a host's `#kodak`.
const suggestions = computed(() =>
  suggestTags(props.tags, tagDraftName.value, activeTags.value).slice(0, 12),
);

/**
 * Add a tag. `stripTagHash` is deliberately NOT part of `addTag`, so the split
 * lives at the entry point: TYPED text loses a leading `#` (people type it out
 * of habit and Rust would file the literal `#kodak`), while a SUGGESTION the
 * host reported is added VERBATIM — stripping it there would file `kodak`
 * when the user picked the host's own `#kodak`. The phone's third entry point
 * is that both live in the SAME sheet, so each names its source explicitly and
 * there is no default.
 */
function commitTag(raw: string, source: "typed" | "suggestion"): void {
  const name = source === "typed" ? stripTagHash(raw) : raw;
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
  emit("update:state", addTag(props.state, name));
}

function dropTag(name: string): void {
  emit("update:state", removeTag(props.state, name, props.title, props.autoTagTitle));
}

/** Retire the derived chip. `removeTag` records the opt-out so the title can
 * keep re-deriving the same slug without the chip coming back. */
function dropGhost(): void {
  if (ghost.value) dropTag(ghost.value);
}

function openTagSheet(): void {
  tagDraft.value = "";
  tagError.value = null;
  sheet.value = "tags";
}

watch(tagDraft, () => {
  tagError.value = null;
});

// ── Collection ─────────────────────────────────────────────────────────────

const newCollectionOpen = ref(false);
const newCollectionName = ref("");

const chosen = computed(() => effectiveCollection(props.state, props.title, props.collections));
const chosenSlug = computed(() => chosen.value?.slug ?? null);

function slugOf(entry: FileUnderCollectionLike): string {
  const slug = entry.slug?.trim();
  return slug && slug.length > 0 ? slug : collectionSlug(entry.name);
}

function openCollectionSheet(): void {
  newCollectionOpen.value = false;
  newCollectionName.value = "";
  sheet.value = "collection";
}

function choose(entry: FileUnderCollectionLike): void {
  emit(
    "update:state",
    pickCollection(props.state, {
      ...(entry.id !== undefined ? { id: entry.id } : {}),
      name: entry.name,
    }),
  );
  closeSheet();
}

function chooseNone(): void {
  emit("update:state", clearCollection(props.state, props.title));
  closeSheet();
}

/** The row's own ✕ — same reducer as None, without opening the sheet. */
function clearChosen(): void {
  emit("update:state", clearCollection(props.state, props.title));
}

function commitNewCollection(): void {
  const name = newCollectionName.value.trim();
  if (!name) return;
  // Match an existing row by slug so "smurfs" doesn't queue a second
  // collection beside "Smurfs"; the host resolves by slug anyway.
  const slug = collectionSlug(name);
  const existing = props.collections.find((entry) => slugOf(entry) === slug);
  choose(existing ?? { name });
}

function closeSheet(): void {
  sheet.value = null;
  newCollectionOpen.value = false;
  newCollectionName.value = "";
}

// ── Filename preview ───────────────────────────────────────────────────────

// A grammar preview, not a promise: the host stamps the real timestamp when
// the print lands, so one wall clock read at mount is honest enough.
const previewStamp = Date.now();
const filenamePreview = computed(() =>
  previewPrintFilename({
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
</script>

<template>
  <section class="mobile-file-under" data-test="mobile-file-under" aria-label="File under">
    <p class="mobile-file-under-kicker">File under</p>

    <!-- Tags -->
    <div class="mobile-file-under-row">
      <span class="mobile-file-under-label">Tags</span>
      <span class="mobile-file-under-chips">
        <span
          v-if="ghostVisible"
          class="mobile-file-under-chip is-ghost"
          data-test="mobile-file-under-ghost"
        >
          <span>{{ ghost }}</span>
          <span class="mobile-file-under-chip-source">from title</span>
          <button
            type="button"
            :aria-label="`Remove the ${ghost} tag`"
            title="Default tag from the title — remove to opt this print out"
            data-test="mobile-file-under-ghost-remove"
            @click="dropGhost"
          >
            ✕
          </button>
        </span>
        <span
          v-for="tag in rowTags"
          :key="tag"
          class="mobile-file-under-chip"
          data-test="mobile-file-under-tag"
        >
          <span>{{ tag }}</span>
          <button
            type="button"
            :aria-label="`Remove the ${tag} tag`"
            data-test="mobile-file-under-tag-remove"
            @click="dropTag(tag)"
          >
            ✕
          </button>
        </span>
        <span
          v-if="activeTags.length === 0"
          class="mobile-file-under-none"
          data-test="mobile-file-under-no-tags"
          >None</span
        >
      </span>
      <button
        type="button"
        class="mobile-file-under-add"
        data-test="mobile-file-under-add-tag"
        @click="openTagSheet"
      >
        Add tag ›
      </button>
    </div>

    <!-- Collection. A row, not a button: the clear ✕ is its own control and
         may not nest inside one. -->
    <div class="mobile-file-under-row">
      <button
        type="button"
        class="mobile-file-under-collection"
        :aria-expanded="sheet === 'collection'"
        data-test="mobile-file-under-collection"
        @click="openCollectionSheet"
      >
        <span class="mobile-file-under-label">Collection</span>
        <span v-if="chosen" class="mobile-file-under-value">{{ chosen.name }}</span>
        <span v-else class="mobile-file-under-none">None</span>
        <span
          v-if="chosen?.source === 'title'"
          class="mobile-file-under-match"
          data-test="mobile-file-under-collection-match"
        >
          · matched to title ✓
        </span>
      </button>
      <button
        v-if="chosen"
        type="button"
        class="mobile-file-under-clear"
        aria-label="Clear collection"
        data-test="mobile-file-under-collection-clear"
        @click="clearChosen"
      >
        ✕
      </button>
    </div>

    <p class="mobile-file-under-filename" data-test="mobile-file-under-filename">
      <span class="mobile-file-under-filename-key">files as</span>
      {{ previewStem
      }}<b v-if="previewSlug" class="mobile-file-under-filename-slug">{{ previewSlug }}</b>
    </p>

    <MobileLibrarySheet
      :open="sheet === 'tags'"
      title="Tags"
      test-id="mobile-file-under-tag-sheet"
      @close="closeSheet"
    >
      <template v-if="sheet === 'tags'">
        <div class="mobile-file-under-sheet-chips" data-test="mobile-file-under-sheet-tags">
          <span v-if="activeTags.length === 0" class="mobile-empty-note">No tags yet.</span>
          <span v-for="tag in activeTags" :key="tag" class="mobile-file-under-chip">
            <span>{{ tag }}</span>
            <button
              type="button"
              :aria-label="`Remove tag ${tag}`"
              data-test="mobile-file-under-sheet-remove"
              @click="dropTag(tag)"
            >
              ✕
            </button>
          </span>
        </div>
        <form class="mobile-library-sheet-form" @submit.prevent="commitTag(tagDraft, 'typed')">
          <label class="field">
            <span>Add a tag</span>
            <input
              v-model="tagDraft"
              class="control"
              autocomplete="off"
              autocapitalize="off"
              enterkeyhint="done"
              placeholder="kodak"
              data-test="mobile-file-under-tag-input"
            />
          </label>
          <button
            class="primary-button"
            type="submit"
            :disabled="!tagDraftName"
            data-test="mobile-file-under-tag-add"
          >
            Add
          </button>
        </form>
        <p
          v-if="tagError"
          class="status-line error-text"
          role="alert"
          data-test="mobile-file-under-tag-error"
        >
          {{ tagError }}
        </p>
        <div
          v-if="suggestions.length"
          class="mobile-file-under-sheet-chips"
          data-test="mobile-file-under-tag-suggestions"
        >
          <button
            v-for="tag in suggestions"
            :key="tag.name"
            class="mobile-library-chip"
            type="button"
            data-test="mobile-file-under-tag-suggestion"
            @click="commitTag(tag.name, 'suggestion')"
          >
            {{ tag.name }}<span class="mobile-library-chip-count">{{ tag.count }}</span>
          </button>
        </div>
        <p class="mobile-file-under-note">New names are created on develop.</p>
      </template>
    </MobileLibrarySheet>

    <MobileLibrarySheet
      :open="sheet === 'collection'"
      title="Collection"
      test-id="mobile-file-under-collection-sheet"
      @close="closeSheet"
    >
      <template v-if="sheet === 'collection'">
        <ul
          class="mobile-library-checklist"
          role="radiogroup"
          aria-label="Collection"
          data-test="mobile-file-under-collection-list"
        >
          <!-- `role="presentation"` on the items so the radios stay direct
               children of the radiogroup for assistive technology. -->
          <li role="presentation">
            <button
              type="button"
              role="radio"
              :aria-checked="!chosen"
              data-test="mobile-file-under-collection-none"
              @click="chooseNone"
            >
              <span class="mobile-library-check" aria-hidden="true">{{ chosen ? "" : "✓" }}</span>
              <span class="mobile-collection-copy"><strong>None</strong></span>
            </button>
          </li>
          <li v-for="entry in collections" :key="slugOf(entry)" role="presentation">
            <button
              type="button"
              role="radio"
              :aria-checked="slugOf(entry) === chosenSlug"
              data-test="mobile-file-under-collection-option"
              @click="choose(entry)"
            >
              <span class="mobile-library-check" aria-hidden="true">{{
                slugOf(entry) === chosenSlug ? "✓" : ""
              }}</span>
              <span class="mobile-collection-copy">
                <strong>{{ entry.name }}</strong>
                <span v-if="typeof entry.count === 'number'" class="mobile-collection-count">{{
                  entry.count
                }}</span>
              </span>
            </button>
          </li>
        </ul>
        <button
          v-if="!newCollectionOpen"
          type="button"
          class="mobile-file-under-new"
          data-test="mobile-file-under-new-collection"
          @click="newCollectionOpen = true"
        >
          ＋ New collection…
        </button>
        <form v-else class="mobile-library-sheet-form" @submit.prevent="commitNewCollection">
          <label class="field">
            <span>New collection</span>
            <input
              v-model="newCollectionName"
              class="control"
              autocomplete="off"
              enterkeyhint="done"
              placeholder="Collection name"
              data-test="mobile-file-under-new-collection-input"
            />
          </label>
          <button
            class="primary-button"
            type="submit"
            :disabled="!newCollectionName.trim()"
            data-test="mobile-file-under-new-collection-create"
          >
            Create
          </button>
        </form>
        <p class="mobile-file-under-note">Created on the machine that develops this print.</p>
      </template>
    </MobileLibrarySheet>
  </section>
</template>

<style scoped>
.mobile-file-under {
  display: grid;
  gap: 6px;
  border: 1px solid var(--edge);
  border-radius: 16px;
  background: var(--bench);
  padding: 10px 12px 12px;
}

.mobile-file-under-kicker {
  margin: 0;
  color: var(--ink-3);
  font-family: var(--font-utility);
  font-size: var(--text-data);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.mobile-file-under-row {
  display: flex;
  min-height: 44px;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.mobile-file-under-label {
  color: var(--ink-3);
  font-size: 14px;
}

.mobile-file-under-value {
  color: var(--rebate);
  font-size: 15px;
}

.mobile-file-under-none {
  color: var(--ink-3);
  font-size: 15px;
}

.mobile-file-under-match {
  color: var(--safelight);
  font-family: var(--font-utility);
  font-size: var(--text-data);
}

.mobile-file-under-chips {
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.mobile-file-under-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 1px solid var(--control-edge);
  border-radius: var(--radius-pill);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--font-utility);
  font-size: var(--text-data);
  padding: 0 2px 0 12px;
}

/* Derived, not typed — dashed so the default reads as an offer, and always
   removable so the print can opt out before Generate. */
.mobile-file-under-chip.is-ghost {
  border-style: dashed;
  color: var(--ink-2);
}

.mobile-file-under-chip-source {
  color: var(--ink-3);
  font-size: 10px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.mobile-file-under-chip button {
  display: grid;
  min-width: 44px;
  min-height: 44px;
  place-items: center;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-size: 15px;
}

.mobile-file-under-add {
  display: inline-flex;
  min-height: 44px;
  align-items: center;
  gap: 4px;
  border: 0;
  background: transparent;
  color: var(--safelight);
  font-size: 15px;
  padding: 0 2px;
}

.mobile-file-under-collection {
  display: flex;
  min-height: 44px;
  min-width: 0;
  flex: 1 1 auto;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  border: 0;
  background: transparent;
  color: var(--rebate);
  text-align: left;
}

.mobile-file-under-clear {
  display: grid;
  min-width: 44px;
  min-height: 44px;
  place-items: center;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-size: 15px;
}

.mobile-file-under-filename {
  margin: 0;
  overflow-wrap: anywhere;
  color: var(--ink-3);
  font-family: var(--font-utility);
  font-size: 11px;
  line-height: 1.5;
}

.mobile-file-under-filename-key {
  margin-right: 6px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.mobile-file-under-filename-slug {
  color: var(--safelight);
  font-weight: 600;
}

.mobile-file-under-sheet-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.mobile-file-under-new {
  display: flex;
  width: 100%;
  min-height: 44px;
  align-items: center;
  gap: 8px;
  border: 1px dashed var(--control-edge);
  border-radius: 12px;
  background: transparent;
  color: var(--safelight);
  font-size: 15px;
  padding: 0 14px;
}

.mobile-file-under-note {
  margin: 0;
  color: var(--ink-3);
  font-size: 13px;
}

/* iOS refuses to leave a focused field alone below 16px. */
.mobile-file-under input {
  font-size: 16px;
}
</style>
