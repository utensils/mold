<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import {
  imageInputFormatOfBase64,
  referenceImageMimeTypes,
} from "../lib/referenceImagesProfile";
import {
  pastReorderThreshold,
  REFERENCE_CANVAS_NOTE,
  referenceDropSide,
  referenceIndexAtPoint,
  referenceMoveAnnouncement,
  referenceStripItems,
} from "../lib/referenceStrip";

/**
 * The ordered reference strip shared by web, desktop and iPhone/Android: one
 * thumbnail per reference, in request order, numbered the way the prompt
 * addresses it ("image 1", "image 2" — `studio/lib/referenceStrip.ts`), with
 * per-picture remove and reorder (a pointer drag, or the keyboard-reachable
 * earlier / later buttons), and a "Sets canvas" mark on the picture a
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
 *
 * Reordering is Pointer Events only, never HTML5 drag-and-drop: Tauri's
 * default `dragDropEnabled` can swallow HTML5 DnD inside the webview
 * (WebView2 on Windows), and an HTML5 drag is how an OS FILE arrives — so the
 * strip's only HTML5 drop is a file, and a reorder can never read as one.
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

// ── Keyboard focus follows the moved picture, or the gap Remove left ──────
const root = ref<HTMLElement | null>(null);
const pendingFocus = ref<{ index: number; dir: "earlier" | "later" } | null>(
  null,
);
/** Set by Remove: the removed picture's index, which after the surface
 * applies the removal is where the NEXT picture now sits. */
const pendingRemoveFocus = ref<number | null>(null);

function move(index: number, dir: "earlier" | "later"): void {
  const to = dir === "earlier" ? index - 1 : index + 1;
  if (to < 0 || to >= props.images.length) return;
  pendingFocus.value = { index: to, dir };
  emit("move", index, to);
}

function remove(index: number): void {
  pendingRemoveFocus.value = index;
  emit("remove", index);
}

function findRemoveButton(index: number): HTMLButtonElement | null {
  return (
    root.value?.querySelector<HTMLButtonElement>(
      `[data-ris-remove="${index}"]`,
    ) ?? null
  );
}

watch(
  () => props.images,
  async () => {
    const pending = pendingFocus.value;
    pendingFocus.value = null;
    if (pending) {
      await nextTick();
      const other = pending.dir === "earlier" ? "later" : "earlier";
      const find = (dir: string) =>
        root.value?.querySelector<HTMLButtonElement>(
          `[data-ris-move="${dir}-${pending.index}"]`,
        );
      const button = find(pending.dir);
      // At an end the same-direction button is disabled; the other one is
      // the natural next press.
      (button && !button.disabled ? button : find(other))?.focus();
      return;
    }
    const removed = pendingRemoveFocus.value;
    pendingRemoveFocus.value = null;
    if (removed === null) return;
    await nextTick();
    // The picture that used to follow the removed one now sits at the same
    // index; failing that, the one before it; failing that (the strip is
    // empty), the add tile — focus never lands on <body>.
    (
      findRemoveButton(removed) ??
      findRemoveButton(removed - 1) ??
      root.value?.querySelector<HTMLButtonElement>(
        `[data-test="${tid("reference-add")}"]`,
      )
    )?.focus();
  },
);

// ── Pointer drag to reorder ───────────────────────────────────────────────
// pointerdown arms; past the threshold the tile captures the pointer and the
// tile under it is hit-tested on every move; pointerup commits one `move`.
// Escape, pointercancel and unmount abort. A press that starts on a control
// belongs to that control. Touch keeps the ‹ › buttons (a finger drag is a
// scroll there), and so does the phone's `touchFriendly` strip.
interface PointerDrag {
  pointerId: number;
  from: number;
  startX: number;
  startY: number;
  tile: HTMLElement;
  active: boolean;
}
let drag: PointerDrag | null = null;
/** The lifted picture while a drag is live. */
const dragFrom = ref<number | null>(null);
/** The tile the picture would land on. */
const dragOver = ref<number | null>(null);
/** The polite live region's sentence after a pointer reorder. */
const announcement = ref("");

const canDrag = computed(() => !props.touchFriendly && !props.disabled);

function tileIndexAt(x: number, y: number): number | null {
  const tiles = root.value?.querySelectorAll<HTMLElement>("[data-ris-tile]");
  if (!tiles) return null;
  return referenceIndexAtPoint(
    Array.from(tiles, (tile) => tile.getBoundingClientRect()),
    x,
    y,
  );
}

function onTilePointerDown(index: number, event: PointerEvent): void {
  if (!canDrag.value || drag) return;
  if (event.pointerType === "touch" || event.button !== 0) return;
  if ((event.target as Element | null)?.closest("button, a, input")) return;
  drag = {
    pointerId: event.pointerId,
    from: index,
    startX: event.clientX,
    startY: event.clientY,
    tile: event.currentTarget as HTMLElement,
    active: false,
  };
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("pointercancel", onPointerCancel);
  window.addEventListener("keydown", onKeyDown);
}

function onPointerMove(event: PointerEvent): void {
  if (!drag || event.pointerId !== drag.pointerId) return;
  if (!drag.active) {
    if (
      !pastReorderThreshold(
        event.clientX - drag.startX,
        event.clientY - drag.startY,
      )
    )
      return;
    drag.active = true;
    dragFrom.value = drag.from;
    try {
      drag.tile.setPointerCapture?.(event.pointerId);
    } catch {
      // Capture is a nicety; the window listeners still see every move.
    }
  }
  event.preventDefault();
  const over = tileIndexAt(event.clientX, event.clientY);
  dragOver.value = over === drag.from ? null : over;
}

function onPointerUp(event: PointerEvent): void {
  if (!drag || event.pointerId !== drag.pointerId) return;
  const { from, active } = drag;
  const to = dragOver.value;
  endDrag();
  if (!active || to === null || to === from) return;
  announcement.value = referenceMoveAnnouncement(from, to, props.ordinalBase);
  emit("move", from, to);
}

function onPointerCancel(event: PointerEvent): void {
  if (drag && event.pointerId === drag.pointerId) endDrag();
}

function onKeyDown(event: KeyboardEvent): void {
  if (event.key !== "Escape" || !drag) return;
  if (drag.active) event.preventDefault();
  endDrag();
}

function endDrag(): void {
  if (drag) {
    try {
      if (drag.tile.hasPointerCapture?.(drag.pointerId))
        drag.tile.releasePointerCapture(drag.pointerId);
    } catch {
      // Already released (the element left the DOM).
    }
  }
  drag = null;
  dragFrom.value = null;
  dragOver.value = null;
  window.removeEventListener("pointermove", onPointerMove);
  window.removeEventListener("pointerup", onPointerUp);
  window.removeEventListener("pointercancel", onPointerCancel);
  window.removeEventListener("keydown", onKeyDown);
}

onBeforeUnmount(endDrag);

function dropSideClass(index: number): string | null {
  if (dragOver.value !== index || dragFrom.value === null) return null;
  return `ris__tile--over-${referenceDropSide(dragFrom.value, index)}`;
}

// ── Drop files to append ──────────────────────────────────────────────────
function onStripDrop(event: DragEvent): void {
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
        :class="[
          {
            'ris__tile--over': dragOver === item.index,
            'ris__tile--lifted': dragFrom === item.index,
            'ris__tile--canvas': item.setsCanvas,
          },
          dropSideClass(item.index),
        ]"
        data-ris-tile
        :data-reorderable="canDrag || undefined"
        :data-test="tid(`reference-tile-${item.index}`)"
        :data-sets-canvas="item.setsCanvas || undefined"
        @pointerdown="onTilePointerDown(item.index, $event)"
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
            :data-ris-remove="item.index"
            :data-test="tid(`reference-remove-${item.index}`)"
            @click="remove(item.index)"
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
      class="ris__sr"
      role="status"
      aria-live="polite"
      data-test="reference-announce"
    >
      {{ announcement }}
    </p>
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
/* A mouse/pen press on the tile body (not its controls) drags it. */
.ris__tile[data-reorderable] {
  cursor: grab;
  user-select: none;
}
.ris__tile--lifted {
  cursor: grabbing;
  opacity: 0.55;
}
.ris__tile--over {
  border-color: var(--mold-blue, #b45309);
}
/* The insertion mark sits in the 8px gap on the side the picture lands. */
.ris__tile--over-before {
  box-shadow: -4px 0 0 -1px var(--mold-blue, #b45309);
}
.ris__tile--over-after {
  box-shadow: 4px 0 0 -1px var(--mold-blue, #b45309);
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
.ris__sr {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}
.ris__note {
  margin: 0;
  color: var(--mold-text-dim, #737373);
  font-size: var(--mold-fs-xs, 0.75rem);
  line-height: 1.45;
}
</style>
