<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import {
  grainParams,
  makeGrainTile,
  tint,
  type DevelopPhase,
  type GrainParams,
} from "../lib/grain";

const props = withDefaults(
  defineProps<{
    seed: number | string;
    /** 0–1 denoise progress. */
    progress: number;
    phase: DevelopPhase;
  }>(),
  { progress: 0 },
);

const canvas = ref<HTMLCanvasElement | null>(null);
let tileCanvas: HTMLCanvasElement | null = null;
let raf = 0;
let current: GrainParams = grainParams(0, "latent");

const reducedMotion =
  typeof matchMedia !== "undefined" &&
  matchMedia("(prefers-reduced-motion: reduce)").matches;

const target = computed(() => grainParams(props.progress, props.phase));

function buildTile() {
  const data = makeGrainTile(props.seed);
  tileCanvas = document.createElement("canvas");
  tileCanvas.width = data.width;
  tileCanvas.height = data.height;
  tileCanvas.getContext("2d")!.putImageData(data, 0, 0);
}

function draw() {
  const el = canvas.value;
  const ctx = el?.getContext("2d");
  if (!el || !ctx || !tileCanvas) return;
  const { width: w, height: h } = el;
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, w, h);
  if (current.amplitude <= 0.005) return;

  // Two passes: raw 1:1 grain (tiled pattern) fades out while magnified
  // blotches (a shrinking source window blown up) organize as the
  // correlation length grows — white noise → smooth low-frequency field.
  const coarseScale = Math.max(1, current.cell / 2);
  const fineWeight = Math.max(0, 1 - (current.cell - 2) / 22);

  const pattern = ctx.createPattern(tileCanvas, "repeat");
  if (pattern) {
    ctx.globalAlpha = current.amplitude * (0.35 + 0.55 * fineWeight);
    ctx.fillStyle = pattern;
    ctx.fillRect(0, 0, w, h);
  }
  const srcWindow = Math.max(3, tileCanvas.width / coarseScale);
  ctx.globalAlpha = current.amplitude * (0.25 + 0.6 * (1 - fineWeight));
  ctx.drawImage(tileCanvas, 0, 0, srcWindow, srcWindow, 0, 0, w, h);

  // Temperature wash: cold Halide latent → warm Safelight while developing.
  const c = tint(current.warmth, props.phase === "stopped");
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 1;
  ctx.fillStyle = `rgb(${c.r} ${c.g} ${c.b})`;
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = "source-over";
}

function tick() {
  const t = target.value;
  const ease = reducedMotion ? 1 : 0.12;
  current = {
    amplitude: current.amplitude + (t.amplitude - current.amplitude) * ease,
    cell: current.cell + (t.cell - current.cell) * ease,
    warmth: current.warmth + (t.warmth - current.warmth) * ease,
  };
  draw();
  const settled =
    Math.abs(current.amplitude - t.amplitude) < 0.004 &&
    Math.abs(current.cell - t.cell) < 0.1 &&
    Math.abs(current.warmth - t.warmth) < 0.004;
  // Stop the loop once eased to the target — params only move on real
  // denoise events, and a 60fps multiply-composite redraw while the GPU is
  // saturated with diffusion starves the WebKit compositor (visible flicker).
  if (!settled) raf = requestAnimationFrame(tick);
  else raf = 0;
}

function kick() {
  if (!raf) raf = requestAnimationFrame(tick);
}

watch(
  () => props.seed,
  () => {
    buildTile();
    current = grainParams(0, "latent");
    kick();
  },
);
watch(target, kick);

let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  buildTile();
  const el = canvas.value;
  if (el) {
    resizeObserver = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      el.width = Math.max(1, Math.round(rect.width));
      el.height = Math.max(1, Math.round(rect.height));
      draw();
    });
    resizeObserver.observe(el);
  }
  kick();
});
onUnmounted(() => {
  cancelAnimationFrame(raf);
  resizeObserver?.disconnect();
});
</script>

<template>
  <canvas ref="canvas" class="ms-develop" aria-hidden="true" />
</template>

<style scoped>
/* Fill the host box exactly as the former Tailwind `h-full w-full` did —
   the canvas stays inline so its box math is unchanged. */
.ms-develop {
  width: 100%;
  height: 100%;
}
</style>
