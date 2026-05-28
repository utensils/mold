<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
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
  <Teleport to="body">
    <div
      v-if="open && sourceImage"
      class="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 p-4 backdrop-blur-sm"
      @click.self="emit('close')"
    >
      <div
        class="glass flex max-h-[92vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl p-4"
      >
        <div class="flex items-center justify-between gap-3">
          <h2 class="truncate text-lg font-semibold text-slate-100">
            Mask editor
          </h2>
          <button
            type="button"
            class="text-slate-400 hover:text-slate-100"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <div
          class="mt-4 flex flex-wrap items-center gap-2 border-y border-slate-800 py-3"
        >
          <button
            type="button"
            class="rounded-lg px-3 py-1 text-sm"
            :class="
              mode === 'brush'
                ? 'bg-brand-500 text-white'
                : 'bg-slate-900/60 text-slate-200'
            "
            data-test="mask-mode-brush"
            @click="mode = 'brush'"
          >
            Brush
          </button>
          <button
            type="button"
            class="rounded-lg px-3 py-1 text-sm"
            :class="
              mode === 'erase'
                ? 'bg-brand-500 text-white'
                : 'bg-slate-900/60 text-slate-200'
            "
            data-test="mask-mode-erase"
            @click="mode = 'erase'"
          >
            Erase
          </button>
          <label class="ml-2 flex items-center gap-2 text-sm text-slate-300">
            Size
            <input
              v-model.number="brushSize"
              type="range"
              min="4"
              max="128"
              step="1"
              class="w-32"
              data-test="mask-brush-size"
            />
          </label>
          <button
            type="button"
            class="rounded-lg bg-slate-900/60 px-3 py-1 text-sm text-slate-200"
            data-test="mask-clear"
            @click="clearMask"
          >
            Clear
          </button>
          <button
            type="button"
            class="rounded-lg bg-slate-900/60 px-3 py-1 text-sm text-slate-200"
            @click="invertMask"
          >
            Invert
          </button>
          <button
            type="button"
            class="rounded-lg bg-slate-900/60 px-3 py-1 text-sm text-slate-200 disabled:opacity-40"
            :class="{ 'opacity-40': undoStack.length === 0 }"
            data-test="mask-undo"
            @click="undo"
          >
            Undo
          </button>
          <button
            type="button"
            class="rounded-lg bg-slate-900/60 px-3 py-1 text-sm text-slate-200 disabled:opacity-40"
            :class="{ 'opacity-40': redoStack.length === 0 }"
            data-test="mask-redo"
            @click="redo"
          >
            Redo
          </button>
        </div>

        <div class="mt-4 flex min-h-0 justify-center overflow-auto">
          <div class="relative max-h-[65vh] max-w-full overflow-hidden">
            <img
              :src="sourceUrl(sourceImage)"
              :alt="sourceImage.filename"
              class="max-h-[65vh] max-w-full select-none object-contain"
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

        <div class="mt-4 flex justify-end gap-2">
          <button
            type="button"
            class="rounded-lg bg-slate-900/60 px-4 py-2 text-sm text-slate-200"
            @click="emit('close')"
          >
            Cancel
          </button>
          <button
            type="button"
            class="rounded-lg bg-brand-500 px-4 py-2 text-sm text-white"
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
