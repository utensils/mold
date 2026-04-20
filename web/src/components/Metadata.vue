<script setup lang="ts">
import type { GalleryImage } from "../types";
import {
  formatResolution,
  formatFileSize,
  formatScheduler,
  formatAbsoluteTime,
  shortModel,
} from "../util/format";

/*
 * Metadata panel — rendered inside the DetailDrawer aside on desktop
 * and as a bottom-sheet body on mobile. Uses the Studio/Lab prototype
 * styling (spec cards, copyable values, accent-tinted LoRA badge).
 */

type CopyTarget = "prompt" | "seed";

const props = defineProps<{
  item: GalleryImage;
  copied: false | CopyTarget;
  canDelete: boolean;
}>();

const emit = defineEmits<{
  (e: "copy", text: string, kind: CopyTarget): void;
  (e: "rerun", item: GalleryImage): void;
  (e: "delete-clicked"): void;
}>();

function onCopy(text: string, kind: CopyTarget) {
  emit("copy", text, kind);
}

function onRerun() {
  emit("rerun", props.item);
}
</script>

<template>
  <div class="drawer-aside-head">
    <div class="drawer-filename" :title="item.filename">
      {{ item.filename }}
    </div>
    <h2 class="drawer-title">
      <span class="card-dot" />
      {{ shortModel(item.metadata.model) }}
    </h2>
    <div class="drawer-time">{{ formatAbsoluteTime(item.timestamp) }}</div>
  </div>

  <button class="drawer-rerun-btn" @click="onRerun">
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M12 8v4l3 2" />
    </svg>
    <div>
      <div>Re-run with tweaks</div>
      <div class="drawer-rerun-sub">loads this prompt + seed into composer</div>
    </div>
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M5 12h14M13 5l7 7-7 7" />
    </svg>
  </button>

  <section class="drawer-section">
    <header>
      <span>Prompt</span>
      <button
        v-if="item.metadata.prompt"
        class="linkbtn"
        @click="onCopy(item.metadata.prompt, 'prompt')"
      >
        {{ copied === "prompt" ? "copied ✓" : "copy" }}
      </button>
    </header>
    <p v-if="item.metadata.prompt" class="drawer-prompt">
      {{ item.metadata.prompt }}
    </p>
    <p v-else class="drawer-prompt drawer-prompt-muted">
      {{
        item.metadata_synthetic
          ? "No mold metadata was embedded in this file."
          : "No prompt recorded."
      }}
    </p>
  </section>

  <section v-if="item.metadata.original_prompt" class="drawer-section">
    <header>
      <span>Original prompt <em>· pre-expansion</em></span>
    </header>
    <p class="drawer-prompt drawer-prompt-muted">
      {{ item.metadata.original_prompt }}
    </p>
  </section>

  <section v-if="item.metadata.negative_prompt" class="drawer-section">
    <header><span>Negative</span></header>
    <p class="drawer-prompt drawer-prompt-neg">
      {{ item.metadata.negative_prompt }}
    </p>
  </section>

  <section class="drawer-specs">
    <div class="spec">
      <div class="spec-label">Resolution</div>
      <div class="spec-value">
        {{ formatResolution(item.metadata) || "—" }}
      </div>
    </div>
    <div class="spec">
      <div class="spec-label">Seed</div>
      <button
        v-if="item.metadata.seed"
        class="spec-value"
        :title="copied === 'seed' ? 'copied!' : 'click to copy'"
        @click="onCopy(String(item.metadata.seed), 'seed')"
      >
        {{ copied === "seed" ? "copied ✓" : item.metadata.seed }}
      </button>
      <div v-else class="spec-value">—</div>
    </div>
    <div class="spec">
      <div class="spec-label">Steps</div>
      <div class="spec-value">{{ item.metadata.steps || "—" }}</div>
    </div>
    <div class="spec">
      <div class="spec-label">Guidance</div>
      <div class="spec-value">
        {{ item.metadata.guidance ? item.metadata.guidance.toFixed(1) : "—" }}
      </div>
    </div>
    <div
      v-if="
        item.metadata.strength !== undefined && item.metadata.strength !== null
      "
      class="spec"
    >
      <div class="spec-label">img2img</div>
      <div class="spec-value">
        strength {{ item.metadata.strength.toFixed(2) }}
      </div>
    </div>
    <div v-if="item.metadata.scheduler" class="spec">
      <div class="spec-label">Scheduler</div>
      <div class="spec-value">
        {{ formatScheduler(item.metadata.scheduler) }}
      </div>
    </div>
    <div v-if="item.metadata.frames" class="spec">
      <div class="spec-label">Frames</div>
      <div class="spec-value">{{ item.metadata.frames }}</div>
    </div>
    <div v-if="item.metadata.fps" class="spec">
      <div class="spec-label">FPS</div>
      <div class="spec-value">{{ item.metadata.fps }}</div>
    </div>
    <div v-if="item.size_bytes" class="spec">
      <div class="spec-label">Size</div>
      <div class="spec-value">{{ formatFileSize(item.size_bytes) }}</div>
    </div>
    <div v-if="item.format" class="spec">
      <div class="spec-label">Format</div>
      <div class="spec-value" style="text-transform: uppercase">
        {{ item.format }}
      </div>
    </div>
  </section>

  <section v-if="item.metadata.lora" class="drawer-lora">
    <div class="drawer-lora-label">LoRA</div>
    <div class="drawer-lora-name">
      {{ item.metadata.lora }}
      <span v-if="item.metadata.lora_scale">
        × {{ item.metadata.lora_scale }}
      </span>
    </div>
  </section>

  <section v-if="canDelete" style="display: flex; justify-content: flex-end">
    <button class="drawer-delete" @click="emit('delete-clicked')">
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M3 6h18" />
        <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
        <path d="M19 6 18 20a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      </svg>
      Delete
    </button>
  </section>

  <p class="drawer-version">mold {{ item.metadata.version }}</p>
</template>
