<script setup lang="ts">
/*
 * Inpaint mask editor (spec §06 Create). Paint/erase a white mask over the
 * source print and hand the flattened PNG back to the composer.
 *
 * Mold Studio migration: was a hand-rolled `fixed inset-0` panel with an emoji
 * close glyph, Tailwind utility colors and no Escape handling. Now @ui
 * ModalPanel + SegmentedControl + Icon, with `useOverlayFocus` supplying the
 * Tab trap and opener restoration the kit leaves to the host.
 */
import { computed, nextTick, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import Icon from "@ui/components/Icon.vue";
import { useOverlayFocus } from "../composables/useOverlayFocus";
import type { SourceImageState } from "../types";

const props = defineProps<{
  open: boolean;
  sourceImage: SourceImageState | null;
  initialMask?: SourceImageState | null;
}>();

const emit = defineEmits<{
  (e: "apply", v: SourceImageState): void;
  (e: "close"): void;
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
const host = ref<HTMLElement | null>(null);
const visible = computed(() => props.open && props.sourceImage !== null);
const { onKeydown } = useOverlayFocus(visible, host);

const modeOptions = [
  { value: "brush" as const, label: "Brush" },
  { value: "erase" as const, label: "Erase" },
];
const mode = ref<"brush" | "erase">("brush");
const brushSize = ref(32);
const drawing = ref(false);
const undoStack = ref<ImageData[]>([]);
const redoStack = ref<ImageData[]>([]);

function context(): CanvasRenderingContext2D | null {
  return canvasRef.value?.getContext("2d") ?? null;
}

function sourceUrl(image: SourceImageState | null): string {
  return image ? `data:image/png;base64,${image.base64}` : "";
}

function ensureCanvasSize(width = 512, height = 512) {
  const canvas = canvasRef.value;
  if (!canvas) return;
  canvas.width = width;
  canvas.height = height;
}

function drawInitialMask() {
  const canvas = canvasRef.value;
  const ctx = context();
  if (!canvas || !ctx || !props.initialMask) return;
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = sourceUrl(props.initialMask);
}

function snapshot(): ImageData | null {
  const canvas = canvasRef.value;
  const ctx = context();
  if (!canvas || !ctx) return null;
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function pushUndo() {
  const image = snapshot();
  if (!image) return;
  undoStack.value = [...undoStack.value, image].slice(-20);
}

function restore(image: ImageData | null) {
  if (!image) return;
  const ctx = context();
  if (!ctx) return;
  ctx.putImageData(image, 0, 0);
}

async function resetCanvas() {
  await nextTick();
  ensureCanvasSize();
  undoStack.value = [];
  redoStack.value = [];
  const canvas = canvasRef.value;
  const ctx = context();
  if (!canvas || !ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawInitialMask();
}

watch(
  () => [props.open, props.sourceImage?.base64, props.initialMask?.base64],
  () => {
    if (props.open) void resetCanvas();
  },
  { immediate: true },
);

function point(event: PointerEvent): { x: number; y: number } {
  const canvas = canvasRef.value;
  if (!canvas) return { x: 0, y: 0 };
  const rect = canvas.getBoundingClientRect();
  const scaleX = rect.width > 0 ? canvas.width / rect.width : 1;
  const scaleY = rect.height > 0 ? canvas.height / rect.height : 1;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function drawPoint(event: PointerEvent) {
  const ctx = context();
  if (!ctx) return;
  const p = point(event);
  ctx.save();
  ctx.globalCompositeOperation =
    mode.value === "erase" ? "destination-out" : "source-over";
  ctx.fillStyle = "rgba(255,255,255,1)";
  ctx.beginPath();
  ctx.arc(p.x, p.y, brushSize.value / 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function startDrawing(event: PointerEvent) {
  drawing.value = true;
  pushUndo();
  redoStack.value = [];
  drawPoint(event);
}

function keepDrawing(event: PointerEvent) {
  if (!drawing.value) return;
  drawPoint(event);
}

function stopDrawing() {
  drawing.value = false;
}

function clearMask() {
  const canvas = canvasRef.value;
  const ctx = context();
  if (!canvas || !ctx) return;
  pushUndo();
  redoStack.value = [];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function invertMask() {
  const canvas = canvasRef.value;
  const ctx = context();
  if (!canvas || !ctx) return;
  pushUndo();
  redoStack.value = [];
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < image.data.length; i += 4) {
    image.data[i] = 255 - image.data[i];
    image.data[i + 1] = 255 - image.data[i + 1];
    image.data[i + 2] = 255 - image.data[i + 2];
    image.data[i + 3] = image.data[i + 3] === 0 ? 255 : image.data[i + 3];
  }
  ctx.putImageData(image, 0, 0);
}

function undo() {
  const previous = undoStack.value.at(-1);
  if (!previous) return;
  const current = snapshot();
  if (current) redoStack.value = [...redoStack.value, current];
  undoStack.value = undoStack.value.slice(0, -1);
  restore(previous);
}

function redo() {
  const next = redoStack.value.at(-1);
  if (!next) return;
  const current = snapshot();
  if (current) undoStack.value = [...undoStack.value, current];
  redoStack.value = redoStack.value.slice(0, -1);
  restore(next);
}

function maskFilename(): string {
  const filename = props.sourceImage?.filename ?? "source.png";
  const stem = filename.replace(/\.[^.]+$/, "");
  return `${stem}-mask.png`;
}

function onSourceLoaded(event: Event) {
  const image = event.target as HTMLImageElement;
  const width = image.naturalWidth || 512;
  const height = image.naturalHeight || 512;
  ensureCanvasSize(width, height);
  const ctx = context();
  const canvas = canvasRef.value;
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawInitialMask();
}

function applyMask() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const dataUrl = canvas.toDataURL("image/png");
  emit("apply", {
    kind: "upload",
    filename: maskFilename(),
    base64: dataUrl.replace(/^data:image\/png;base64,/, ""),
  });
}
</script>

<template>
  <!-- Viewport-fixed host: ModalPanel is `position: absolute; inset: 0` and
       Create is a long scrolling column, so mounting inline would draw the
       panel at the top of the document. Same host pattern as
       ModelDetailDrawer. -->
  <div
    v-if="visible && sourceImage"
    ref="host"
    class="me-host"
    data-test="mask-editor-host"
    @keydown="onKeydown"
  >
    <ModalPanel
      :open="true"
      :width="900"
      label="Mask editor"
      @close="emit('close')"
    >
      <div class="me">
        <header class="me__head">
          <h2 class="me__title">
            <Icon name="mask" :size="17" />
            <span>Mask editor</span>
          </h2>
          <button
            type="button"
            class="me__close"
            aria-label="Close"
            data-test="mask-close"
            @click="emit('close')"
          >
            <Icon name="close" :size="15" />
          </button>
        </header>

        <div class="me__tools">
          <div class="me__mode" data-test="mask-mode">
            <SegmentedControl
              v-model="mode"
              :options="modeOptions"
              label="Mask tool"
            />
          </div>
          <label class="me__size">
            <span>Size</span>
            <input
              v-model.number="brushSize"
              type="range"
              min="4"
              max="128"
              step="1"
              data-test="mask-brush-size"
            />
            <span class="me__size-val">{{ brushSize }}</span>
          </label>
          <span class="me__spacer" />
          <button
            type="button"
            class="me__tool"
            data-test="mask-clear"
            @click="clearMask"
          >
            Clear
          </button>
          <button type="button" class="me__tool" @click="invertMask">
            Invert
          </button>
          <button
            type="button"
            class="me__tool"
            :disabled="undoStack.length === 0"
            data-test="mask-undo"
            @click="undo"
          >
            Undo
          </button>
          <button
            type="button"
            class="me__tool"
            :disabled="redoStack.length === 0"
            data-test="mask-redo"
            @click="redo"
          >
            Redo
          </button>
        </div>

        <div class="me__stage">
          <div class="me__canvas-wrap">
            <img
              :src="sourceUrl(sourceImage)"
              :alt="sourceImage.filename"
              class="me__source"
              draggable="false"
              @load="onSourceLoaded"
            />
            <canvas
              ref="canvasRef"
              class="me__canvas"
              data-test="mask-canvas"
              @pointerdown="startDrawing"
              @pointermove="keepDrawing"
              @pointerup="stopDrawing"
              @pointerleave="stopDrawing"
            ></canvas>
          </div>
        </div>
      </div>
      <template #footer>
        <span class="me__spacer" />
        <button type="button" class="me__cancel" @click="emit('close')">
          Cancel
        </button>
        <button
          type="button"
          class="me__apply"
          data-test="mask-apply"
          @click="applyMask"
        >
          Apply
        </button>
      </template>
    </ModalPanel>
  </div>
</template>

<style scoped>
.me-host {
  position: fixed;
  inset: 0;
  z-index: 50;
}

.me {
  display: flex;
  flex-direction: column;
  gap: 14px;
  color: var(--rebate);
}

.me__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.me__title {
  display: flex;
  align-items: center;
  gap: 9px;
  margin: 0;
  min-width: 0;
  font-family: var(--f-display);
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.me__title svg {
  color: var(--ink-3);
  flex: none;
}

.me__title span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.me__close {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  transition:
    color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.me__close:hover {
  color: var(--rebate);
  background: var(--surface);
}

.me__tools {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 12px 0;
  border-top: 1px solid var(--edge);
  border-bottom: 1px solid var(--edge);
}

.me__mode {
  width: 168px;
}

.me__size {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--f-mono);
  font-size: 11px;
  letter-spacing: 0.02em;
  color: var(--ink-3);
}

.me__size input {
  width: 120px;
  accent-color: var(--safelight);
}

.me__size-val {
  min-width: 22px;
  color: var(--ink-2);
}

.me__spacer {
  flex: 1;
}

.me__tool {
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--ink-2);
  padding: 7px 13px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.me__tool:hover:not(:disabled) {
  background: var(--surface);
  color: var(--rebate);
}

.me__tool:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.me__stage {
  display: flex;
  justify-content: center;
  min-height: 0;
  overflow: auto;
}

.me__canvas-wrap {
  position: relative;
  max-width: 100%;
  max-height: 58vh;
  overflow: hidden;
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-lg);
  background: var(--bath);
}

.me__source {
  display: block;
  max-width: 100%;
  max-height: 58vh;
  object-fit: contain;
  user-select: none;
}

.me__canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  cursor: crosshair;
  touch-action: none;
  opacity: 0.7;
}

.me__cancel {
  border: 1px solid var(--ce);
  border-radius: var(--radius-control-lg);
  background: transparent;
  color: var(--ink-2);
  padding: 10px 16px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.me__apply {
  border: 0;
  border-radius: var(--radius-control-lg);
  background: var(--safelight);
  color: var(--on-accent);
  padding: 10px 20px;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}
</style>
