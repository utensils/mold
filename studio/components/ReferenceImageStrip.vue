<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import {
  imageInputFormatOfBase64,
  referenceImageMimeTypes,
} from "../lib/referenceImagesProfile";
import {
  REFERENCE_CANVAS_NOTE,
  referenceStripItems,
} from "../lib/referenceStrip";

/**
 * The ordered reference strip shared by web, desktop and iPhone/Android: one
 * thumbnail per reference, in request order, numbered the way the prompt
 * addresses it ("image 1", "image 2" — `studio/lib/referenceStrip.ts`), with
 * per-picture remove and reorder (drag, or the keyboard-reachable earlier /
 * later buttons), and a "Sets canvas" mark on the picture a
 * `canvas: last-reference` recipe sizes the canvas from.
 *
 * Every thumbnail sits on the kit's `.ms-alpha-bed` checkerboard drawn on the
 * image box itself, so a transparent PNG/WebP reference shows exactly which
 * pixels carry no colour and an opaque one covers it.
 *
 * The strip only reports intent: surfaces own their lists (web keeps
 * `SourceImageState` rows, desktop and the phone raw base64) and apply
 * `move`/`remove` with `reorderReference`. It renders
 * `data-drop-target="references"` so a shell that hit-tests an OS drag can
 * name it (`imageDropRouting`), and an HTML5 file drop is marked handled and
 * handed over as `files` — the surface APPENDS them, never replaces.
 */
export interface ReferenceStripImage {
  /** Raw base64 without a data-URI prefix; empty/null renders a placeholder. */
  data: string | null;
  mimeType?: string | null;
  filename?: string | null;
}

const props = withDefaults(
  defineProps<{
    images: readonly ReferenceStripImage[];
    /** The first picture is the edit Target (`primary_is_target`). */
    firstIsTarget?: boolean;
    /** The recipe's canvas follows the LAST picture (`stripSetsCanvas`). */
    setsCanvas?: boolean;
    /** Images the request carries ahead of the strip (`referenceOrdinalBase`). */
    ordinalBase?: number;
    /** The advertised ceiling; `null` is unbounded. */
    max?: number | null;
    disabled?: boolean;
    /** iPhone/Android: 44pt controls, no dragging (there is no drag there). */
    touchFriendly?: boolean;
    /** The add tile's wording. */
    addLabel?: string;
    /** The add tile's wording while the strip is empty; it then spans the
     * strip as its drop zone. */
    emptyLabel?: string | null;
    /** The recipe renders from references: the empty add tile says so. */
    required?: boolean;
    /** The strip container's own test hook (web `reference-strip`, desktop
     * `attachment-strip`). */
    stripTestId?: string;
    /** Prefix for every per-tile test hook. */
    testIdPrefix?: string;
  }>(),
  {
    firstIsTarget: false,
    setsCanvas: false,
    ordinalBase: 0,
    max: null,
    disabled: false,
    touchFriendly: false,
    addLabel: "Add",
    emptyLabel: null,
    required: false,
    stripTestId: "reference-strip",
    testIdPrefix: "",
  },
);

const emit = defineEmits<{
  /** Move the picture at `from` to `to` (indices into `images`). */
  move: [from: number, to: number];
  remove: [index: number];
  /** Open the surface's picker. */
  add: [];
  /** Files dropped on the strip, in drop order, to APPEND. */
  files: [files: File[]];
}>();

/** The MIME a tile drag carries, so a reorder never reads as a file drop. */
const REORDER_MIME = "application/x-mold-reference-index";

const items = computed(() =>
  referenceStripItems({
    count: props.images.length,
    firstIsTarget: props.firstIsTarget,
    setsCanvas: props.setsCanvas,
    ordinalBase: props.ordinalBase,
  }),
);
const full = computed(
  () => props.max !== null && props.images.length >= props.max,
);

function thumbUrl(image: ReferenceStripImage): string | null {
  if (!image.data) return null;
  if (image.data.startsWith("data:")) return image.data;
  const mime =
    image.mimeType ||
    referenceImageMimeTypes([imageInputFormatOfBase64(image.data) ?? "png"])[0];
  return `data:${mime};base64,${image.data}`;
}

function tid(name: string): string {
  return `${props.testIdPrefix}${name}`;
}

// ── Keyboard focus follows the moved picture ──────────────────────────────
const root = ref<HTMLElement | null>(null);
const pendingFocus = ref<{ index: number; dir: "earlier" | "later" } | null>(
  null,
);

function move(index: number, dir: "earlier" | "later"): void {
  const to = dir === "earlier" ? index - 1 : index + 1;
  if (to < 0 || to >= props.images.length) return;
  pendingFocus.value = { index: to, dir };
  emit("move", index, to);
}

watch(
  () => props.images,
  async () => {
    const pending = pendingFocus.value;
    pendingFocus.value = null;
    if (!pending) return;
    await nextTick();
    const other = pending.dir === "earlier" ? "later" : "earlier";
    const find = (dir: string) =>
      root.value?.querySelector<HTMLButtonElement>(
        `[data-ris-move="${dir}-${pending.index}"]`,
      );
    const button = find(pending.dir);
    // At an end the same-direction button is disabled; the other one is the
    // natural next press.
    (button && !button.disabled ? button : find(other))?.focus();
  },
);

// ── Drag to reorder, drop to append ───────────────────────────────────────
const dragFrom = ref<number | null>(null);
const dragOver = ref<number | null>(null);

function onTileDragStart(index: number, event: DragEvent): void {
  if (props.touchFriendly || props.disabled) return;
  dragFrom.value = index;
  event.dataTransfer?.setData(REORDER_MIME, String(index));
  if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
}
function isReorder(event: DragEvent): boolean {
  const types = event.dataTransfer?.types;
  return (
    dragFrom.value !== null ||
    (types ? Array.from(types).includes(REORDER_MIME) : false)
  );
}
function onTileDrop(index: number, event: DragEvent): void {
  dragOver.value = null;
  if (!isReorder(event)) return; // a file: let the strip take it
  event.preventDefault();
  event.stopPropagation();
  const raw = event.dataTransfer?.getData(REORDER_MIME);
  const from = raw && !Number.isNaN(Number(raw)) ? Number(raw) : dragFrom.value;
  dragFrom.value = null;
  if (from === null || from === index) return;
  emit("move", from, index);
}
function onTileDragEnd(): void {
  dragFrom.value = null;
  dragOver.value = null;
}
function onStripDrop(event: DragEvent): void {
  dragOver.value = null;
  const files = Array.from(event.dataTransfer?.files ?? []);
  if (files.length === 0) return;
  event.preventDefault();
  if (props.disabled) return;
  emit("files", files);
}
</script>

<template>
  <div
    ref="root"
    class="ris"
    :class="{ 'ris--touch': touchFriendly }"
    :data-test="stripTestId"
    data-drop-target="references"
    @dragover.prevent
    @drop="onStripDrop"
  >
    <ol class="ris__list" aria-label="Ordered reference images">
      <li
        v-for="item in items"
        :key="`${item.index}-${images[item.index]?.data?.slice(-16) ?? ''}`"
        class="ris__tile"
        :class="{
          'ris__tile--over': dragOver === item.index,
          'ris__tile--canvas': item.setsCanvas,
        }"
        :draggable="touchFriendly || disabled ? 'false' : 'true'"
        :data-test="tid(`reference-tile-${item.index}`)"
        :data-sets-canvas="item.setsCanvas || undefined"
        @dragstart="onTileDragStart(item.index, $event)"
        @dragend="onTileDragEnd"
        @dragover.prevent="dragOver = item.index"
        @dragleave="dragOver = null"
        @drop="onTileDrop(item.index, $event)"
      >
        <div class="ris__thumb">
          <img
            v-if="thumbUrl(images[item.index]!)"
            class="ris__img ms-alpha-bed"
            :src="thumbUrl(images[item.index]!)!"
            :alt="`${item.label}, ${item.role.toLowerCase()}${images[item.index]?.filename ? ` (${images[item.index]!.filename})` : ''}`"
            :data-test="tid(`reference-thumb-${item.index}`)"
            draggable="false"
          />
          <span v-else class="ris__img ris__img--missing">No preview</span>
          <span
            class="ris__ordinal"
            aria-hidden="true"
            :data-test="tid(`reference-ordinal-${item.index}`)"
            >{{ item.ordinal }}</span
          >
          <span
            v-if="item.setsCanvas"
            class="ris__canvas"
            data-test="reference-sets-canvas"
            >Sets canvas</span
          >
        </div>
        <div class="ris__meta">
          <span
            class="ris__label"
            :data-test="tid(`reference-label-${item.index}`)"
            >{{ item.label }}</span
          >
          <span
            class="ris__role"
            :data-test="tid(`reference-role-${item.index}`)"
            >{{ item.role }}</span
          >
        </div>
        <span
          v-if="images[item.index]?.filename"
          class="ris__filename"
          :title="images[item.index]!.filename!"
          >{{ images[item.index]!.filename }}</span
        >
        <div class="ris__actions">
          <button
            type="button"
            class="ris__action"
            :disabled="disabled || item.index === 0"
            :aria-label="`Move ${item.label.toLowerCase()} earlier`"
            :data-ris-move="`earlier-${item.index}`"
            :data-test="tid(`reference-earlier-${item.index}`)"
            @click="move(item.index, 'earlier')"
          >
            ‹
          </button>
          <button
            type="button"
            class="ris__action"
            :disabled="disabled || item.index === images.length - 1"
            :aria-label="`Move ${item.label.toLowerCase()} later`"
            :data-ris-move="`later-${item.index}`"
            :data-test="tid(`reference-later-${item.index}`)"
            @click="move(item.index, 'later')"
          >
            ›
          </button>
          <button
            type="button"
            class="ris__action ris__action--danger"
            :disabled="disabled"
            :aria-label="`Remove ${item.label.toLowerCase()}`"
            :data-test="tid(`reference-remove-${item.index}`)"
            @click="emit('remove', item.index)"
          >
            ✕
          </button>
        </div>
      </li>
      <li
        class="ris__tile ris__tile--add"
        :class="{ 'ris__tile--empty': images.length === 0 }"
      >
        <button
          type="button"
          class="ris__add"
          :disabled="disabled || full"
          :aria-required="(required && images.length === 0) || undefined"
          :data-test="tid('reference-add')"
          @click="emit('add')"
        >
          <span aria-hidden="true" class="ris__add-glyph">＋</span>
          <span>{{
            full
              ? `Full · ${max}`
              : images.length === 0 && emptyLabel
                ? emptyLabel
                : addLabel
          }}</span>
        </button>
      </li>
    </ol>
    <p
      v-if="setsCanvas && images.length > 0"
      class="ris__note"
      data-test="reference-canvas-note"
    >
      {{ REFERENCE_CANVAS_NOTE }}
    </p>
  </div>
</template>

<style scoped>
.ris {
  --ris-tile: 112px;
  --ris-control: 24px;
  display: grid;
  gap: 6px;
  min-width: 0;
}
.ris--touch {
  --ris-tile: 136px;
  --ris-control: 44px;
}
/* The strip WRAPS rather than scrolling: order is the point, so every
 * picture (up to ten on Qwen Image 2.1) stays in view at once. */
.ris__list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.ris__tile {
  position: relative;
  display: grid;
  flex: 0 0 var(--ris-tile);
  align-content: start;
  gap: 4px;
  width: var(--ris-tile);
  padding: 4px;
  border: 1px solid var(--mold-border, #ddd);
  border-radius: var(--mold-radius-3);
  background: var(--mold-bg-deep, transparent);
}
.ris__tile--over {
  border-color: var(--mold-blue, #b45309);
}
.ris__tile--canvas {
  border-color: color-mix(in srgb, var(--mold-blue, #b45309) 60%, transparent);
}
.ris__thumb {
  position: relative;
  aspect-ratio: 4 / 3;
  overflow: hidden;
  border-radius: var(--mold-radius-2);
  background: var(--mold-media-bed, #111);
}
.ris__img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  --ms-alpha-cell: 8px;
}
.ris__img--missing {
  display: grid;
  place-items: center;
  color: var(--mold-text-dim, #737373);
  font-size: var(--mold-fs-micro, 0.6875rem);
}
.ris__ordinal {
  position: absolute;
  top: 4px;
  left: 4px;
  display: grid;
  place-items: center;
  min-width: 20px;
  height: 20px;
  padding: 0 5px;
  border-radius: 999px; /* literal: a pill is round in every theme */
  background: color-mix(in srgb, var(--mold-media-bed, #111) 80%, transparent);
  color: var(--mold-text, #fff);
  font-family: var(--mold-font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro, 0.6875rem);
  font-weight: 700;
}
/* Textual, never colour alone: the canvas mark must survive monochrome. */
.ris__canvas {
  position: absolute;
  right: 4px;
  bottom: 4px;
  left: 4px;
  overflow: hidden;
  padding: 1px 4px;
  text-align: center;
  text-overflow: ellipsis;
  border-radius: var(--mold-radius-1);
  background: var(--mold-blue, #b45309);
  color: var(--mold-on-accent, #fff);
  font-family: var(--mold-font-mono, ui-monospace, monospace);
  font-size: calc(var(--mold-fs-micro, 0.6875rem) * 0.85);
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
}
/* The role wraps under the ordinal rather than truncating: "Reference" is
 * the word a user reads to tell it from the Target. */
.ris__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  column-gap: 6px;
  min-width: 0;
}
.ris__label {
  color: var(--mold-text, inherit);
  font-size: var(--mold-fs-xs, 0.75rem);
  font-weight: 600;
  white-space: nowrap;
}
.ris__role {
  color: var(--mold-text-dim, #737373);
  font-family: var(--mold-font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro, 0.6875rem);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  white-space: nowrap;
}
.ris__filename {
  overflow: hidden;
  color: var(--mold-text-dim, #737373);
  font-size: var(--mold-fs-micro, 0.6875rem);
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ris__actions {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 3px;
}
.ris__action {
  min-height: var(--ris-control);
  padding: 0;
  border: 1px solid var(--mold-border, #ddd);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-text-2, var(--mold-text-dim, #737373));
  font-size: var(--mold-fs-sm, 0.8125rem);
  line-height: 1;
  cursor: pointer;
}
.ris__action:hover:not(:disabled) {
  color: var(--mold-text, inherit);
}
.ris__action--danger:hover:not(:disabled) {
  color: var(--mold-error, #b42318);
}
.ris__action:disabled {
  cursor: default;
  opacity: 0.35;
}
.ris__action:focus-visible,
.ris__add:focus-visible {
  outline: 2px solid var(--mold-blue, #b45309);
  outline-offset: 1px;
}
.ris__tile--add {
  padding: 0;
  border: 0;
  background: none;
}
/* Empty, the add tile IS the strip: one full-width drop zone. */
.ris__tile--empty {
  flex: 1 1 100%;
  width: auto;
}
.ris__add {
  display: grid;
  place-content: center;
  gap: 2px;
  width: 100%;
  height: 100%;
  min-height: calc(var(--ris-tile) * 0.75);
  border: 1px dashed var(--mold-border, #bbb);
  border-radius: var(--mold-radius-3);
  background: transparent;
  color: var(--mold-text-dim, #737373);
  font-size: var(--mold-fs-xs, 0.75rem);
  cursor: pointer;
}
.ris__add:hover:not(:disabled) {
  border-color: var(--mold-blue, #b45309);
  color: var(--mold-blue, #b45309);
}
.ris__add:disabled {
  cursor: default;
  opacity: 0.6;
}
.ris__add-glyph {
  font-size: var(--mold-fs-md, 1rem);
}
.ris__note {
  margin: 0;
  color: var(--mold-text-dim, #737373);
  font-size: var(--mold-fs-xs, 0.75rem);
  line-height: 1.45;
}
</style>
