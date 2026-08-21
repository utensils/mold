<script setup lang="ts">
/*
 * "File under" — Create-time Library organization (design pitch §1–§3, §6).
 *
 * One capability-gated group in the Create controls region, between the
 * essentials and Advanced. Two rows and a preview line:
 *
 *   Tags        a dashed, removable GHOST chip derived from the print title
 *               ("smurfs · from title"), the real chips the user added, and
 *               an `Add tag…` field whose popover suggests the fleet's own
 *               tags with counts.
 *   Collection  None / a picked collection / the collection whose slug
 *               equals the title's ("· matched to title"), always clearable.
 *               The match NEVER creates anything; only an explicit
 *               "New collection…" names one, and the host creates it by name
 *               at develop time.
 *   files as    the creation-time filename this print will land under.
 *
 * Every rule is the shared `@studio/lib/fileUnder` contract — this component
 * renders it and reports intent. The parent owns the state object.
 */
import { computed, nextTick, ref, watch } from "vue";
import Icon from "@ui/components/Icon.vue";
import Popover from "@ui/components/Popover.vue";
import {
  addTag,
  clearCollection,
  deriveGhostTag,
  effectiveCollection,
  effectiveTags,
  pickCollection,
  removeTag,
  requestTagKey,
  restoreGhostTag,
  stripTagHash,
  suggestTags,
  validateNewTag,
  type FileUnderCollectionLike,
  type FileUnderState,
} from "@studio/lib/fileUnder";
import { fileUnderPreviewName } from "../../lib/fileUnder";
import type { TagCount } from "../../types";

/** Enough to be useful in a 340px rail without becoming a list. */
const MAX_SUGGESTIONS = 8;

const props = withDefaults(
  defineProps<{
    state: FileUnderState;
    /** The live Create title — the ghost chip and the match derive from it. */
    title?: string | null;
    /** Settings ▸ Library ▸ "Tag new prints with their title". */
    autoTag: boolean;
    /** The fleet's tag vocabulary with counts (merged across hosts). */
    suggestions?: readonly TagCount[];
    /** Collections merged across hosts by slug. */
    collections?: readonly (FileUnderCollectionLike & { count?: number })[];
    /** Resolved model id, for the filename preview. */
    model: string;
    /** Output format for the filename preview. */
    ext: string;
    /** Creation stamp for the preview. Frozen so the line does not tick. */
    timestamp?: number;
  }>(),
  {
    title: null,
    suggestions: () => [],
    collections: () => [],
    timestamp: () => Date.now(),
  },
);

const emit = defineEmits<{ "update:state": [value: FileUnderState] }>();

// ── Tags ────────────────────────────────────────────────────────────────────

const ghost = computed(() => deriveGhostTag(props.title, props.autoTag));
const ghostShown = computed(() => !props.state.ghostRemoved && !!ghost.value);
const tags = computed(() =>
  effectiveTags(props.state, props.title, props.autoTag),
);
/** `effectiveTags` puts the ghost first, so the flag is positional. */
const chips = computed(() =>
  tags.value.map((name) => ({
    name,
    ghost:
      ghostShown.value &&
      ghost.value !== null &&
      requestTagKey(name) === requestTagKey(ghost.value),
  })),
);

const draft = ref("");
const tagError = ref("");
const suggestOpen = ref(false);
const tagInput = ref<HTMLInputElement | null>(null);

const matches = computed(() =>
  suggestTags(props.suggestions, draft.value, tags.value).slice(
    0,
    MAX_SUGGESTIONS,
  ),
);

/**
 * Add a tag. `typed` text loses a leading `#` (people type it out of habit
 * and Rust would file the literal `#kodak`); a suggestion the host reported
 * is added VERBATIM, because stripping it there would file a different tag.
 */
function add(raw: string, source: "typed" | "suggestion") {
  const name = source === "typed" ? stripTagHash(raw) : raw;
  const problem = validateNewTag(name, tags.value);
  if (problem) {
    tagError.value = problem;
    return;
  }
  tagError.value = "";
  draft.value = "";
  emit("update:state", addTag(props.state, name));
}

function onTagKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" || event.key === ",") {
    event.preventDefault();
    if (draft.value.trim()) add(draft.value, "typed");
    return;
  }
  if (event.key === "Escape") {
    draft.value = "";
    tagError.value = "";
    suggestOpen.value = false;
    tagInput.value?.blur();
  }
}

function onRemoveTag(name: string) {
  tagError.value = "";
  emit(
    "update:state",
    removeTag(props.state, name, props.title, props.autoTag),
  );
}

function onRestoreGhost() {
  emit("update:state", restoreGhostTag(props.state));
}

watch(draft, () => {
  if (tagError.value) tagError.value = "";
});

// ── Collection ──────────────────────────────────────────────────────────────

const collection = computed(() =>
  effectiveCollection(props.state, props.title, props.collections),
);
const pickerOpen = ref(false);
const creating = ref(false);
const newName = ref("");
const newInput = ref<HTMLInputElement | null>(null);

const rows = computed(() =>
  props.collections.map((entry) => ({
    key: entry.slug ?? entry.name,
    name: entry.name,
    count: entry.count ?? null,
    selected: collection.value
      ? collection.value.name.toLowerCase() === entry.name.toLowerCase()
      : false,
  })),
);

function closePicker() {
  pickerOpen.value = false;
  creating.value = false;
  newName.value = "";
}

function choose(name: string) {
  emit("update:state", pickCollection(props.state, { name }));
  closePicker();
}

function chooseNone() {
  emit("update:state", clearCollection(props.state, props.title));
  closePicker();
}

function onClear() {
  emit("update:state", clearCollection(props.state, props.title));
}

async function startCreating() {
  creating.value = true;
  await nextTick();
  newInput.value?.focus();
}

function commitNew() {
  const name = newName.value.trim();
  if (!name) return;
  choose(name);
}

function onNewKeydown(event: KeyboardEvent) {
  if (event.key === "Enter") {
    event.preventDefault();
    commitNew();
  } else if (event.key === "Escape") {
    event.preventDefault();
    creating.value = false;
    newName.value = "";
  }
}

// ── Filename preview ────────────────────────────────────────────────────────

const preview = computed(() =>
  fileUnderPreviewName({
    model: props.model,
    title: props.title,
    ext: props.ext,
    timestamp: props.timestamp,
  }),
);
</script>

<template>
  <div class="fu" data-test="file-under-group">
    <div class="fu__label">File under</div>

    <!-- Tags -->
    <div class="fu__chips" data-test="file-under-tags">
      <span
        v-for="chip in chips"
        :key="chip.name"
        class="fu__chip"
        :class="{ 'fu__chip--ghost': chip.ghost }"
        :data-test="chip.ghost ? 'file-under-ghost' : 'file-under-tag'"
      >
        <span class="fu__chip-name">{{ chip.name }}</span>
        <span v-if="chip.ghost" class="fu__chip-note">· from title</span>
        <button
          type="button"
          class="fu__x"
          :aria-label="
            chip.ghost
              ? `Do not tag this print with ${chip.name}`
              : `Remove tag ${chip.name}`
          "
          :data-test="
            chip.ghost ? 'file-under-ghost-remove' : 'file-under-tag-remove'
          "
          @click="onRemoveTag(chip.name)"
        >
          <Icon name="close" :size="11" :stroke-width="2.2" />
        </button>
      </span>

      <Popover
        :open="suggestOpen && matches.length > 0"
        label="Tag suggestions"
        class="fu__addwrap"
        @update:open="suggestOpen = $event"
      >
        <template #trigger>
          <input
            ref="tagInput"
            v-model="draft"
            class="fu__input"
            type="text"
            placeholder="Add tag…"
            aria-label="Add tag"
            data-test="file-under-tag-input"
            @keydown="onTagKeydown"
            @focus="suggestOpen = true"
            @blur="suggestOpen = false"
          />
        </template>
        <div class="fu__suggest" data-test="file-under-suggestions">
          <button
            v-for="tag in matches"
            :key="tag.name"
            type="button"
            class="fu__sug"
            data-test="file-under-suggestion"
            @mousedown.prevent="add(tag.name, 'suggestion')"
          >
            <span class="fu__sug-name">{{ tag.name }}</span>
            <span class="fu__sug-n">{{ tag.count }}</span>
          </button>
          <p class="fu__sug-foot" data-test="file-under-suggest-foot">
            ↵ adds · new names are created on develop
          </p>
        </div>
      </Popover>
    </div>

    <p
      v-if="state.ghostRemoved && ghost"
      class="fu__hint"
      data-test="file-under-ghost-restored"
    >
      Not tagging this print with “{{ ghost }}”.
      <button
        type="button"
        class="fu__link"
        data-test="file-under-ghost-restore"
        @click="onRestoreGhost"
      >
        Undo
      </button>
    </p>
    <p
      v-if="tagError"
      class="fu__hint fu__hint--error"
      role="alert"
      data-test="file-under-tag-error"
    >
      {{ tagError }}
    </p>

    <!-- Collection -->
    <div class="fu__row" data-test="file-under-collection">
      <Popover
        :open="pickerOpen"
        label="Collection"
        class="fu__pickwrap"
        @update:open="pickerOpen = $event ? true : (closePicker(), false)"
      >
        <template #trigger>
          <button
            type="button"
            class="fu__field"
            :aria-expanded="pickerOpen"
            aria-label="Collection"
            data-test="file-under-collection-open"
            @click="pickerOpen ? closePicker() : (pickerOpen = true)"
          >
            <Icon name="library" :size="12" />
            <span v-if="!collection" class="fu__field-none">None</span>
            <template v-else>
              <span class="fu__field-name">{{ collection.name }}</span>
              <span
                v-if="collection.source === 'title'"
                class="fu__field-note"
                data-test="file-under-collection-matched"
                >· matched to title ✓</span
              >
            </template>
          </button>
        </template>
        <div class="fu__picker">
          <button
            type="button"
            class="fu__opt"
            :data-on="collection ? undefined : 'true'"
            data-test="file-under-collection-none"
            @click="chooseNone"
          >
            None
          </button>
          <button
            v-for="row in rows"
            :key="row.key"
            type="button"
            class="fu__opt"
            :data-on="row.selected ? 'true' : undefined"
            data-test="file-under-collection-option"
            @click="choose(row.name)"
          >
            <span class="fu__opt-name">{{ row.name }}</span>
            <span v-if="row.count !== null" class="fu__opt-n">{{
              row.count
            }}</span>
          </button>
          <button
            v-if="!creating"
            type="button"
            class="fu__opt fu__opt--new"
            data-test="file-under-collection-new"
            @click="startCreating"
          >
            <Icon name="plus" :size="12" /> New collection…
          </button>
          <div v-else class="fu__new">
            <input
              ref="newInput"
              v-model="newName"
              class="fu__new-input"
              type="text"
              placeholder="Collection name"
              aria-label="New collection name"
              data-test="file-under-collection-new-input"
              @keydown="onNewKeydown"
            />
            <p class="fu__new-foot">↵ creates and selects it · Esc cancels</p>
          </div>
        </div>
      </Popover>
      <button
        v-if="collection"
        type="button"
        class="fu__x fu__x--field"
        aria-label="Clear collection"
        data-test="file-under-collection-clear"
        @click="onClear"
      >
        <Icon name="close" :size="11" :stroke-width="2.2" />
      </button>
    </div>

    <p class="fu__preview" data-test="file-under-preview">
      <span class="fu__preview-k">files as</span> {{ preview }}
    </p>
  </div>
</template>

<style scoped>
.fu {
  margin-bottom: 20px;
}

.fu__label {
  font-size: 12px;
  color: var(--ink-2);
  margin-bottom: 9px;
  font-weight: 600;
}

.fu__chips {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  min-height: 32px;
  padding: 4px 6px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
}
.fu__chips:focus-within {
  outline: 2px solid var(--safelight);
  outline-offset: 1px;
}

.fu__chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  max-width: 100%;
  height: 22px;
  padding: 0 4px 0 8px;
  border-radius: var(--radius-pill);
  background: var(--sel-bg);
  color: var(--sel-ink);
  font-size: 11.5px;
  font-weight: 600;
}
/* The auto-derived chip reads as a default, not a decision the user made. */
.fu__chip--ghost {
  background: transparent;
  border: 1px dashed var(--sel-border);
}

.fu__chip-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fu__chip-note {
  font-family: var(--f-mono);
  font-size: 9.5px;
  font-weight: 400;
  color: var(--ink-3);
}

.fu__x {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: inherit;
  cursor: pointer;
}
.fu__x:hover {
  background: color-mix(in srgb, var(--rebate) 12%, transparent);
}
.fu__x--field {
  width: 20px;
  height: 20px;
  color: var(--ink-3);
}

.fu__addwrap {
  flex: 1;
  min-width: 90px;
}
.fu__addwrap :deep(.ms-popover__trigger) {
  width: 100%;
}

.fu__input {
  width: 100%;
  min-width: 80px;
  height: 22px;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
  outline: none;
}
.fu__input::placeholder {
  color: var(--ink-3);
}

.fu__suggest {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 190px;
}
.fu__sug {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  min-height: 26px;
  padding: 0 7px;
  border: 0;
  border-radius: var(--radius-control-sm);
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
  text-align: left;
  cursor: pointer;
}
.fu__sug:hover {
  background: var(--sel-bg);
  color: var(--sel-ink);
}
.fu__sug-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fu__sug-n {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}
.fu__sug-foot {
  margin: 4px 0 0;
  padding: 0 7px;
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}

.fu__row {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 8px;
}
.fu__pickwrap {
  flex: 1;
  min-width: 0;
}
.fu__pickwrap :deep(.ms-popover__trigger) {
  width: 100%;
}

.fu__field {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  min-height: 32px;
  padding: 0 10px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
  text-align: left;
  cursor: pointer;
}
.fu__field:hover {
  border-color: var(--safelight);
}
.fu__field-none {
  color: var(--ink-3);
}
.fu__field-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fu__field-note {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}

.fu__picker {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 200px;
}
.fu__opt {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  min-height: 28px;
  padding: 0 7px;
  border: 0;
  border-radius: var(--radius-control-sm);
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
  text-align: left;
  cursor: pointer;
}
.fu__opt:hover {
  background: color-mix(in srgb, var(--rebate) 5%, transparent);
}
.fu__opt[data-on="true"] {
  color: var(--sel-ink);
  font-weight: 600;
}
.fu__opt-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fu__opt-n {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}
.fu__opt--new {
  justify-content: flex-start;
  gap: 6px;
  margin-top: 2px;
  color: var(--safelight);
  font-weight: 600;
}
.fu__opt--new:hover {
  background: var(--sel-bg);
}

.fu__new {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 4px;
}
.fu__new-input {
  height: 30px;
  padding: 0 8px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
  outline: none;
}
.fu__new-input:focus {
  border-color: var(--safelight);
}
.fu__new-foot {
  margin: 0;
  padding: 0 2px;
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}

.fu__hint {
  margin: 8px 0 0;
  font-size: 11px;
  color: var(--ink-3);
  line-height: 1.4;
}
.fu__hint--error {
  color: var(--stop);
}
.fu__link {
  border: 0;
  background: transparent;
  color: var(--safelight);
  font-size: 11px;
  text-decoration: underline;
  text-underline-offset: 2px;
  cursor: pointer;
  padding: 0;
}

.fu__preview {
  margin: 8px 0 0;
  font-family: var(--f-mono);
  font-size: 10px;
  line-height: 1.5;
  color: var(--ink-3);
  overflow-wrap: anywhere;
}
.fu__preview-k {
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 9px;
}
</style>
