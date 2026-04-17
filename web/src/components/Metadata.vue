<script setup lang="ts">
import { imageUrl } from "../api";
import type { GalleryImage } from "../types";
import {
  formatResolution,
  formatFileSize,
  formatScheduler,
  formatAbsoluteTime,
  shortModel,
} from "../util/format";

/*
 * Metadata panel — rendered as the always-visible sidebar on desktop
 * and as a bottom-sheet body on mobile. Kept framework-agnostic so
 * `<DetailDrawer>` can reuse it in both layouts without duplicating
 * the 200-line template.
 */

type CopyTarget = "prompt" | "seed";

defineProps<{
  item: GalleryImage;
  copied: false | CopyTarget;
  canDelete: boolean;
}>();

const emit = defineEmits<{
  (e: "copy", text: string, kind: CopyTarget): void;
  (e: "delete-clicked"): void;
}>();

function onCopy(text: string, kind: CopyTarget) {
  emit("copy", text, kind);
}
</script>

<template>
  <header class="flex items-start gap-3">
    <div class="min-w-0 flex-1">
      <p
        class="truncate font-mono text-[12px] text-ink-400"
        :title="item.filename"
      >
        {{ item.filename }}
      </p>
      <h2
        class="mt-1 flex items-center gap-2 text-base font-semibold text-ink-50"
      >
        <span class="h-1.5 w-1.5 rounded-full bg-brand-400"></span>
        {{ shortModel(item.metadata.model) }}
      </h2>
      <p class="mt-0.5 text-[12px] text-ink-400">
        {{ formatAbsoluteTime(item.timestamp) }}
      </p>
    </div>

    <a
      :href="imageUrl(item.filename)"
      :download="item.filename"
      class="inline-flex h-9 items-center gap-1.5 rounded-full bg-white/10 px-3 text-[12px] font-medium text-white transition hover:bg-white/20"
      aria-label="Download file"
    >
      <svg
        class="h-3.5 w-3.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M12 3v13" />
        <path d="m6 12 6 6 6-6" />
        <path d="M5 21h14" />
      </svg>
      Save
    </a>
  </header>

  <!-- Prompt -->
  <section class="mt-5">
    <div class="mb-2 flex items-center justify-between">
      <span
        class="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-400"
      >
        Prompt
      </span>
      <button
        v-if="item.metadata.prompt"
        class="text-[11px] font-medium text-brand-300 transition hover:text-brand-400"
        @click="onCopy(item.metadata.prompt, 'prompt')"
      >
        {{ copied === "prompt" ? "copied ✓" : "copy" }}
      </button>
    </div>
    <p
      v-if="item.metadata.prompt"
      class="whitespace-pre-wrap rounded-xl bg-white/[0.04] p-3 text-[13.5px] leading-relaxed text-ink-100"
    >
      {{ item.metadata.prompt }}
    </p>
    <p
      v-else
      class="rounded-xl bg-white/[0.04] p-3 text-[13px] italic text-ink-400"
    >
      {{
        item.metadata_synthetic
          ? "No mold metadata was embedded in this file."
          : "No prompt recorded."
      }}
    </p>
  </section>

  <!-- Original prompt (shows only when expansion was used) -->
  <section v-if="item.metadata.original_prompt" class="mt-4">
    <div
      class="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-400"
    >
      Original prompt
    </div>
    <p
      class="whitespace-pre-wrap rounded-xl bg-white/[0.03] p-3 text-[13px] leading-relaxed text-ink-200"
    >
      {{ item.metadata.original_prompt }}
    </p>
  </section>

  <!-- Negative -->
  <section v-if="item.metadata.negative_prompt" class="mt-4">
    <div
      class="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-400"
    >
      Negative
    </div>
    <p
      class="whitespace-pre-wrap rounded-xl bg-rose-500/5 p-3 text-[13px] leading-relaxed text-rose-200"
    >
      {{ item.metadata.negative_prompt }}
    </p>
  </section>

  <!-- Specs grid -->
  <section class="mt-5 grid grid-cols-2 gap-2 text-[12.5px]">
    <div class="rounded-xl bg-white/[0.04] px-3 py-2.5">
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        Resolution
      </div>
      <div class="mt-0.5 font-medium text-ink-50">
        {{ formatResolution(item.metadata) || "—" }}
      </div>
    </div>
    <div class="rounded-xl bg-white/[0.04] px-3 py-2.5">
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        Seed
      </div>
      <button
        class="mt-0.5 block w-full text-left font-mono text-[12px] font-medium text-ink-50 transition hover:text-brand-300"
        :title="copied === 'seed' ? 'copied!' : 'click to copy'"
        @click="onCopy(String(item.metadata.seed), 'seed')"
      >
        {{ item.metadata.seed || "—" }}
      </button>
    </div>
    <div class="rounded-xl bg-white/[0.04] px-3 py-2.5">
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        Steps
      </div>
      <div class="mt-0.5 font-medium text-ink-50">
        {{ item.metadata.steps || "—" }}
      </div>
    </div>
    <div class="rounded-xl bg-white/[0.04] px-3 py-2.5">
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        Guidance
      </div>
      <div class="mt-0.5 font-medium text-ink-50">
        {{ item.metadata.guidance ? item.metadata.guidance.toFixed(1) : "—" }}
      </div>
    </div>

    <div
      v-if="
        item.metadata.strength !== undefined && item.metadata.strength !== null
      "
      class="rounded-xl bg-white/[0.04] px-3 py-2.5"
    >
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        img2img
      </div>
      <div class="mt-0.5 font-medium text-ink-50">
        strength {{ item.metadata.strength.toFixed(2) }}
      </div>
    </div>

    <div
      v-if="item.metadata.scheduler"
      class="rounded-xl bg-white/[0.04] px-3 py-2.5"
    >
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        Scheduler
      </div>
      <div class="mt-0.5 font-medium text-ink-50">
        {{ formatScheduler(item.metadata.scheduler) }}
      </div>
    </div>

    <div
      v-if="item.metadata.frames"
      class="rounded-xl bg-white/[0.04] px-3 py-2.5"
    >
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        Frames
      </div>
      <div class="mt-0.5 font-medium text-ink-50">
        {{ item.metadata.frames }}
      </div>
    </div>
    <div
      v-if="item.metadata.fps"
      class="rounded-xl bg-white/[0.04] px-3 py-2.5"
    >
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">FPS</div>
      <div class="mt-0.5 font-medium text-ink-50">
        {{ item.metadata.fps }}
      </div>
    </div>

    <div v-if="item.size_bytes" class="rounded-xl bg-white/[0.04] px-3 py-2.5">
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        File size
      </div>
      <div class="mt-0.5 font-medium text-ink-50">
        {{ formatFileSize(item.size_bytes) }}
      </div>
    </div>
    <div v-if="item.format" class="rounded-xl bg-white/[0.04] px-3 py-2.5">
      <div class="text-[10.5px] uppercase tracking-wider text-ink-400">
        Format
      </div>
      <div class="mt-0.5 font-medium uppercase text-ink-50">
        {{ item.format }}
      </div>
    </div>
  </section>

  <!-- LoRA -->
  <section
    v-if="item.metadata.lora"
    class="mt-4 rounded-xl bg-brand-500/10 px-3 py-3"
  >
    <div class="text-[10.5px] uppercase tracking-wider text-brand-300">
      LoRA
    </div>
    <div class="mt-0.5 break-all font-mono text-[12px] text-brand-100">
      {{ item.metadata.lora }}
      <span v-if="item.metadata.lora_scale" class="text-brand-300">
        × {{ item.metadata.lora_scale }}
      </span>
    </div>
  </section>

  <!-- Danger zone (opt-in — hidden unless the server advertises
       `gallery.can_delete`). -->
  <section v-if="canDelete" class="mt-6 flex justify-end">
    <button
      class="inline-flex h-9 items-center gap-1.5 rounded-full bg-rose-500/15 px-3 text-[12.5px] font-medium text-rose-200 transition hover:bg-rose-500/25 hover:text-rose-100"
      @click="emit('delete-clicked')"
    >
      <svg
        class="h-3.5 w-3.5"
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

  <p class="mt-6 text-[11px] text-ink-500">mold {{ item.metadata.version }}</p>
</template>
