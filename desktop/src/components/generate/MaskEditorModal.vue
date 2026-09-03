<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
import { invertPixels, pointerToCanvas, pushCapped } from "../../lib/maskEditor";
import { base64ToDataUrl } from "../../lib/image";

const props = defineProps<{
  open: boolean;
  /** Source image, base64 (no data-URI prefix) — the desktop form's shape. */
  source: string | null;
  /** Source filename, used only for the alt text. */
  filename?: string | null;
  /** Existing mask to seed the canvas, base64 (no data-URI prefix). */
  initialMask?: string | null;
}>();

const emit = defineEmits<{
  /** Emits the painted mask as base64 (no data-URI prefix). */
  (e: "apply", mask: string): void;
  (e: "close"): void;
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
// Focus the close button on open and hand focus back to the opener on close,
// so the dialog is keyboard-operable and doesn't strand focus (matches Lightbox).
const closeBtn = ref<HTMLButtonElement | null>(null);
let restoreFocusEl: HTMLElement | null = null;
const mode = ref<"brush" | "erase">("brush");
const brushSize = ref(32);
const drawing = ref(false);
const undoStack = ref<ImageData[]>([]);
const redoStack = ref<ImageData[]>([]);

function context(): CanvasRenderingContext2D | null {
  return canvasRef.value?.getContext("2d") ?? null;
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
  img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  img.src = base64ToDataUrl(props.initialMask);
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
  undoStack.value = pushCapped(undoStack.value, image);
}

function restore(image: ImageData | null) {
  if (!image) return;
  context()?.putImageData(image, 0, 0);
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
  () => [props.open, props.source, props.initialMask],
  () => {
    if (props.open) void resetCanvas();
  },
  { immediate: true },
);

watch(
  () => props.open,
  (open) => {
    if (open) {
      restoreFocusEl = document.activeElement as HTMLElement | null;
      void nextTick(() => closeBtn.value?.focus());
    } else {
      restoreFocusEl?.focus?.();
      restoreFocusEl = null;
    }
  },
);

function point(event: PointerEvent): { x: number; y: number } {
  const canvas = canvasRef.value;
  if (!canvas) return { x: 0, y: 0 };
  return pointerToCanvas(event.clientX, event.clientY, canvas.getBoundingClientRect(), {
    width: canvas.width,
    height: canvas.height,
  });
}

function drawPoint(event: PointerEvent) {
  const ctx = context();
  if (!ctx) return;
  const p = point(event);
  ctx.save();
  ctx.globalCompositeOperation = mode.value === "erase" ? "destination-out" : "source-over";
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
  invertPixels(image.data);
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

function onSourceLoaded(event: Event) {
  const image = event.target as HTMLImageElement;
  ensureCanvasSize(image.naturalWidth || 512, image.naturalHeight || 512);
  const canvas = canvasRef.value;
  const ctx = context();
  if (!canvas || !ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawInitialMask();
}

function applyMask() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const dataUrl = canvas.toDataURL("image/png");
  emit("apply", dataUrl.replace(/^data:image\/png;base64,/, ""));
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open && source"
      class="mask-editor-backdrop fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      @click.self="emit('close')"
      @keydown.esc="emit('close')"
    >
      <div
        class="mask-editor-dialog border-border flex max-h-[92vh] w-full max-w-4xl flex-col overflow-hidden rounded-window border bg-bg p-4"
        role="dialog"
        aria-modal="true"
        aria-label="Mask editor"
      >
        <div class="flex items-center justify-between gap-3">
          <h2 class="truncate text-base font-semibold text-fg">Mask editor</h2>
          <button
            ref="closeBtn"
            type="button"
            class="text-fg-dim hover:text-fg"
            aria-label="Close"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <div
          class="mask-editor-toolbar border-border mt-4 flex flex-wrap items-center gap-2 border-y py-3"
        >
          <button
            type="button"
            class="rounded-control px-3 py-1 text-sm transition-colors duration-100"
            :class="
              mode === 'brush'
                ? 'bg-accent font-semibold text-on-accent'
                : 'border-border border text-fg-2 hover:text-fg'
            "
            data-test="mask-mode-brush"
            @click="mode = 'brush'"
          >
            Brush
          </button>
          <button
            type="button"
            class="rounded-control px-3 py-1 text-sm transition-colors duration-100"
            :class="
              mode === 'erase'
                ? 'bg-accent font-semibold text-on-accent'
                : 'border-border border text-fg-2 hover:text-fg'
            "
            data-test="mask-mode-erase"
            @click="mode = 'erase'"
          >
            Erase
          </button>
          <label class="ml-2 flex items-center gap-2 text-micro text-fg-2">
            Size
            <input
              v-model.number="brushSize"
              type="range"
              min="4"
              max="128"
              step="1"
              class="w-32 accent-accent"
              data-test="mask-brush-size"
            />
          </label>
          <button
            type="button"
            class="border-border rounded-control border px-3 py-1 text-sm text-fg-2 hover:text-fg"
            data-test="mask-clear"
            @click="clearMask"
          >
            Clear
          </button>
          <button
            type="button"
            class="border-border rounded-control border px-3 py-1 text-sm text-fg-2 hover:text-fg"
            @click="invertMask"
          >
            Invert
          </button>
          <button
            type="button"
            class="border-border rounded-control border px-3 py-1 text-sm text-fg-2 hover:text-fg disabled:opacity-40"
            :disabled="undoStack.length === 0"
            data-test="mask-undo"
            @click="undo"
          >
            Undo
          </button>
          <button
            type="button"
            class="border-border rounded-control border px-3 py-1 text-sm text-fg-2 hover:text-fg disabled:opacity-40"
            :disabled="redoStack.length === 0"
            data-test="mask-redo"
            @click="redo"
          >
            Redo
          </button>
        </div>

        <p class="mt-3 text-micro text-fg-dim">White repaints, black preserves.</p>

        <div class="mask-editor-stage mt-2 flex min-h-0 justify-center overflow-auto">
          <div
            class="border-border relative max-h-[62vh] max-w-full overflow-hidden rounded-inner border"
          >
            <img
              :src="base64ToDataUrl(source)"
              :alt="filename ?? 'source'"
              class="max-h-[62vh] max-w-full select-none object-contain"
              draggable="false"
              @load="onSourceLoaded"
            />
            <canvas
              ref="canvasRef"
              class="absolute inset-0 h-full w-full cursor-crosshair touch-none opacity-70"
              data-test="mask-canvas"
              @pointerdown="startDrawing"
              @pointermove="keepDrawing"
              @pointerup="stopDrawing"
              @pointerleave="stopDrawing"
            ></canvas>
          </div>
        </div>

        <div class="mask-editor-actions mt-4 flex justify-end gap-2">
          <button
            type="button"
            class="border-border rounded-control border px-4 py-2 text-sm text-fg-2 hover:text-fg"
            @click="emit('close')"
          >
            Cancel
          </button>
          <button
            type="button"
            class="rounded-control bg-accent px-4 py-2 text-sm font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px"
            data-test="mask-apply"
            @click="applyMask"
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
