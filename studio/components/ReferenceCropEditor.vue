<script setup lang="ts">
/*
 * Reference crop editor — the shared CONTENT of the Crop action on a MiniMax
 * H3 Ref2VA image reference (a dialog on desktop/web, a bottom sheet on
 * iPhone). Pointer + touch drag of the box and its four 44 pt corner handles,
 * keyboard nudging, the Free / 1:1 / 4:3 / 3:2 / 16:9 presets, the pad-cost
 * hint, Reset, and Apply. Every rectangle rule lives in
 * `@studio/lib/referenceCrop`; this component only maps pointers to it.
 */
import { computed, onBeforeUnmount, ref, watch } from "vue";
import {
  REFERENCE_CROP_ASPECTS,
  fullReferenceCrop,
  moveReferenceCrop,
  referenceCropAspectId,
  referenceCropIsIdentity,
  referenceCropForAspect,
  referencePadEstimate,
  resizeReferenceCropFromCorner,
  type ReferenceCrop,
  type ReferenceCropAspectId,
  type ReferenceCropCorner,
} from "../lib/referenceCrop";

const props = withDefaults(
  defineProps<{
    image: { data: string; mimeType: string; width: number; height: number };
    crop: ReferenceCrop | null;
    /** iPhone: larger controls. Handles are 44 pt on every surface. */
    large?: boolean;
  }>(),
  { large: false },
);

const emit = defineEmits<{
  /** The normalized rect, or `null` when the crop is the whole image. */
  apply: [crop: ReferenceCrop | null];
  cancel: [];
}>();

const size = computed(() => ({
  width: props.image.width,
  height: props.image.height,
}));
const working = ref<ReferenceCrop>(props.crop ?? fullReferenceCrop(size.value));
const aspect = ref<ReferenceCropAspectId>(
  referenceCropAspectId(working.value, size.value),
);
watch(
  () => props.crop,
  (crop) => {
    working.value = crop ?? fullReferenceCrop(size.value);
    aspect.value = referenceCropAspectId(working.value, size.value);
  },
);

const dataUrl = computed(
  () =>
    `data:${props.image.mimeType || "image/png"};base64,${props.image.data}`,
);
const boxStyle = computed(() => ({
  left: `${(working.value.x / size.value.width) * 100}%`,
  top: `${(working.value.y / size.value.height) * 100}%`,
  width: `${(working.value.width / size.value.width) * 100}%`,
  height: `${(working.value.height / size.value.height) * 100}%`,
}));
const lockedRatio = computed(() => {
  const entry = REFERENCE_CROP_ASPECTS.find((a) => a.id === aspect.value);
  if (!entry || entry.ratio === null) return null;
  return size.value.height > size.value.width ? 1 / entry.ratio : entry.ratio;
});

const format = (value: number) => value.toLocaleString("en-US");
const estimate = computed(() =>
  referencePadEstimate(size.value, working.value),
);
const uncropped = computed(() => referencePadEstimate(size.value));
const hint = computed(() => {
  const w = working.value;
  const cropLabel = `${w.width}×${w.height} of ${size.value.width}×${size.value.height}`;
  if (referenceCropIsIdentity(w, size.value)) {
    return `${cropLabel} · whole image · about ${format(uncropped.value.pads)} vision pads`;
  }
  return `${cropLabel} · about ${format(estimate.value.pads)} vision pads (${format(
    uncropped.value.pads,
  )} uncropped)`;
});

const CORNERS: { id: ReferenceCropCorner; label: string }[] = [
  { id: "nw", label: "top-left" },
  { id: "ne", label: "top-right" },
  { id: "sw", label: "bottom-left" },
  { id: "se", label: "bottom-right" },
];

// ── Pointer mapping ──────────────────────────────────────────────────────
const stage = ref<HTMLElement | null>(null);

/** Displayed pixels → source pixels. Unlaid-out stages (tests) map 1:1. */
function scale(): number {
  const width = stage.value?.clientWidth ?? 0;
  return width > 0 ? width / size.value.width : 1;
}

interface Drag {
  kind: "move" | ReferenceCropCorner;
  startX: number;
  startY: number;
  origin: ReferenceCrop;
}
let drag: Drag | null = null;

function corner(crop: ReferenceCrop, id: ReferenceCropCorner) {
  return {
    x: id === "nw" || id === "sw" ? crop.x : crop.x + crop.width,
    y: id === "nw" || id === "ne" ? crop.y : crop.y + crop.height,
  };
}

function onPointerMove(event: PointerEvent | MouseEvent): void {
  if (!drag) return;
  const factor = scale();
  const dx = (event.clientX - drag.startX) / factor;
  const dy = (event.clientY - drag.startY) / factor;
  if (drag.kind === "move") {
    working.value = moveReferenceCrop(drag.origin, dx, dy, size.value);
    return;
  }
  const from = corner(drag.origin, drag.kind);
  working.value = resizeReferenceCropFromCorner(
    drag.origin,
    drag.kind,
    { x: from.x + dx, y: from.y + dy },
    size.value,
    lockedRatio.value,
  );
}

function endDrag(): void {
  drag = null;
  window.removeEventListener("pointermove", onPointerMove);
  window.removeEventListener("pointerup", endDrag);
  window.removeEventListener("pointercancel", endDrag);
}

function beginDrag(kind: Drag["kind"], event: PointerEvent | MouseEvent): void {
  if ("button" in event && event.button !== 0) return;
  event.preventDefault();
  drag = {
    kind,
    startX: event.clientX,
    startY: event.clientY,
    origin: working.value,
  };
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);
}

onBeforeUnmount(endDrag);

// Arrow keys nudge by 8 source px (32 with Shift) — the vision-pad grid's
// own quarter and whole cell.
function onKeydown(event: KeyboardEvent): void {
  const step = event.shiftKey ? 32 : 8;
  const delta: Record<string, [number, number]> = {
    ArrowLeft: [-step, 0],
    ArrowRight: [step, 0],
    ArrowUp: [0, -step],
    ArrowDown: [0, step],
  };
  const move = delta[event.key];
  if (!move) return;
  event.preventDefault();
  working.value = moveReferenceCrop(
    working.value,
    move[0],
    move[1],
    size.value,
  );
}

function pick(id: ReferenceCropAspectId): void {
  aspect.value = id;
  if (id !== "free") working.value = referenceCropForAspect(size.value, id);
}

function reset(): void {
  aspect.value = "free";
  working.value = fullReferenceCrop(size.value);
}

function apply(): void {
  emit(
    "apply",
    referenceCropIsIdentity(working.value, size.value)
      ? null
      : { ...working.value },
  );
}
</script>

<template>
  <div
    class="crop-editor"
    :class="{ 'crop-editor--large': large }"
    data-test="reference-crop-editor"
  >
    <div
      ref="stage"
      class="crop-editor__stage"
      :style="{
        aspectRatio: `${size.width} / ${size.height}`,
        '--crop-ratio': String(size.width / size.height),
      }"
    >
      <img class="crop-editor__image" :src="dataUrl" alt="" draggable="false" />
      <div
        class="crop-editor__box"
        data-test="crop-box"
        role="group"
        tabindex="0"
        aria-label="Crop rectangle; arrow keys move it, Shift for larger steps"
        :style="boxStyle"
        @pointerdown="beginDrag('move', $event)"
        @keydown="onKeydown"
      >
        <button
          v-for="handle in CORNERS"
          :key="handle.id"
          type="button"
          class="crop-editor__handle"
          :class="`crop-editor__handle--${handle.id}`"
          :data-test="`crop-handle-${handle.id}`"
          :aria-label="`Resize crop from the ${handle.label} corner`"
          @pointerdown.stop="beginDrag(handle.id, $event)"
        />
      </div>
    </div>

    <div
      class="crop-editor__aspects"
      role="radiogroup"
      aria-label="Crop aspect"
    >
      <button
        v-for="entry in REFERENCE_CROP_ASPECTS"
        :key="entry.id"
        type="button"
        role="radio"
        class="crop-editor__aspect"
        :aria-checked="aspect === entry.id ? 'true' : 'false'"
        :data-test="`crop-aspect-${entry.id}`"
        @click="pick(entry.id)"
      >
        {{ entry.label }}
      </button>
    </div>

    <p class="crop-editor__hint" data-test="crop-hint" aria-live="polite">
      {{ hint }}
    </p>

    <div class="crop-editor__actions">
      <button
        type="button"
        class="crop-editor__secondary"
        data-test="crop-reset"
        @click="reset"
      >
        Reset
      </button>
      <span class="crop-editor__spacer" />
      <button
        type="button"
        class="crop-editor__secondary"
        data-test="crop-cancel"
        @click="emit('cancel')"
      >
        Cancel
      </button>
      <button
        type="button"
        class="crop-editor__primary"
        data-test="crop-apply"
        @click="apply"
      >
        Apply crop
      </button>
    </div>
  </div>
</template>

<style scoped>
.crop-editor {
  display: grid;
  gap: 12px;
  min-width: 0;
  color: var(--mold-text);
}
.crop-editor__stage {
  position: relative;
  /* The stage IS the source's aspect: its width is whichever is smaller,
     the column or the width a 60vh-tall image of this ratio would have, so
     the image fills it exactly and box percentages / pointer scale map onto
     the pixels that are actually shown. Never clamp the height alone — a
     letterboxed image would put the box over pixels it does not cover. */
  width: min(100%, calc(60vh * var(--crop-ratio, 1)));
  margin: 0 auto;
  overflow: hidden;
  border-radius: 10px;
  background: var(--well, rgba(128, 128, 128, 0.14));
  touch-action: none;
  user-select: none;
  -webkit-user-select: none;
}
.crop-editor__image {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  pointer-events: none;
}
.crop-editor__box {
  position: absolute;
  box-sizing: border-box;
  border: 2px solid var(--accent, #fff);
  /* The shade outside the crop: one shadow instead of four rectangles. */
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.55);
  cursor: move;
  touch-action: none;
}
.crop-editor__box:focus-visible {
  outline: 2px solid var(--accent, #fff);
  outline-offset: 2px;
}
.crop-editor__handle {
  position: absolute;
  width: 44px;
  height: 44px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: nwse-resize;
  touch-action: none;
}
.crop-editor__handle::after {
  content: "";
  position: absolute;
  inset: 14px;
  border-radius: 50%;
  background: var(--accent, #fff);
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.5);
}
.crop-editor__handle--nw {
  top: -22px;
  left: -22px;
}
.crop-editor__handle--ne {
  top: -22px;
  right: -22px;
  cursor: nesw-resize;
}
.crop-editor__handle--sw {
  bottom: -22px;
  left: -22px;
  cursor: nesw-resize;
}
.crop-editor__handle--se {
  bottom: -22px;
  right: -22px;
}
.crop-editor__aspects {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.crop-editor__aspect,
.crop-editor__secondary,
.crop-editor__primary {
  min-width: 44px;
  min-height: 36px;
  padding: 0 12px;
  border: 1px solid var(--mold-border, #bbb);
  border-radius: 8px;
  background: var(--mold-bg, transparent);
  color: inherit;
  font: inherit;
  cursor: pointer;
}
.crop-editor__aspect[aria-checked="true"],
.crop-editor__primary {
  background: var(--accent, #333);
  border-color: var(--accent, #333);
  color: var(--mold-on-accent, #fff);
}
.crop-editor__hint {
  margin: 0;
  color: var(--mold-text-dim, #737373);
  font-size: 12px;
  line-height: 1.45;
}
.crop-editor__actions {
  display: flex;
  gap: 8px;
  align-items: center;
}
.crop-editor__spacer {
  flex: 1;
}
.crop-editor--large .crop-editor__aspect,
.crop-editor--large .crop-editor__secondary,
.crop-editor--large .crop-editor__primary {
  min-height: 44px;
}
</style>
