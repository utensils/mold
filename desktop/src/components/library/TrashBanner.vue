<script setup lang="ts">
/*
 * TrashBanner — the 40px retention strip under the Library header in the
 * Trash scope (V3 "Shelf"). Copy comes from the shared
 * `trashRetentionSummary` (This device sets the sentence, hosts that differ
 * are named) with every number set in the mono face; the right edge carries
 * the count and a "Change retention · Machines" link the parent routes.
 */
import { computed } from "vue";
import { trashRetentionSummary, type RetentionHost } from "@studio/lib/libraryOrganization";
import { formatBytes } from "../../lib/format";

const props = withDefaults(
  defineProps<{
    /** Trash-capable hosts, This device first. */
    hosts: RetentionHost[];
    /** Logical prints in the trash across hosts. */
    count: number;
    /** Bytes across the trash, when known. */
    bytes?: number | null;
    /** Label of the retention link's destination. */
    linkLabel?: string;
  }>(),
  { bytes: null, linkLabel: "Change retention · Machines" },
);

const emit = defineEmits<{ changeRetention: [] }>();

const summary = computed(() => trashRetentionSummary(props.hosts));

const countLabel = computed(() => {
  const noun = props.count === 1 ? "print" : "prints";
  const base = `${props.count} ${noun} in trash`;
  return props.bytes != null && props.bytes > 0 ? `${base} · ${formatBytes(props.bytes)}` : base;
});
</script>

<template>
  <div
    class="border-edge flex h-10 shrink-0 items-center gap-2.5 border-b bg-[color-mix(in_srgb,var(--halide)_10%,var(--bath))] px-6 text-[12.5px] text-ink-2"
    role="status"
    data-test="trash-banner"
  >
    <span class="text-halide" aria-hidden="true">•</span>
    <span v-if="summary.segments.length > 0" data-test="trash-banner-summary">
      <template v-for="(segment, i) in summary.segments" :key="i">
        <b v-if="segment.mono" class="font-utility text-[11.5px] font-semibold text-ink">{{
          segment.text
        }}</b>
        <span v-else>{{ segment.text }}</span>
      </template>
    </span>
    <span v-else data-test="trash-banner-summary">No connected machine keeps a trash.</span>
    <span class="flex-1" />
    <span class="font-utility text-[10.5px] text-ink-3" data-test="trash-banner-count">{{
      countLabel
    }}</span>
    <button
      v-if="hosts.length > 0"
      type="button"
      class="rounded-control px-1 text-[12px] text-safelight underline-offset-2 hover:underline"
      data-test="trash-banner-link"
      @click="emit('changeRetention')"
    >
      {{ linkLabel }}
    </button>
  </div>
</template>
