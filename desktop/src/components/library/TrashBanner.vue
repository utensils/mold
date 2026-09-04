<script setup lang="ts">
import Icon from "@ui/components/Icon.vue";
/*
 * TrashBanner — the 40px retention strip under the Library header in the
 * Trash scope. Copy comes from the shared
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
  const noun = props.count === 1 ? "picture" : "pictures";
  const base = `${props.count} ${noun} in trash`;
  return props.bytes != null && props.bytes > 0 ? `${base} · ${formatBytes(props.bytes)}` : base;
});
</script>

<template>
  <div
    class="border-border flex h-10 shrink-0 items-center gap-2.5 border-b bg-error/10 px-3.5 text-xs text-fg"
    role="status"
    data-test="trash-banner"
  >
    <Icon name="trash" :size="15" class="shrink-0 text-error" aria-hidden="true" />
    <span v-if="summary.segments.length > 0" data-test="trash-banner-summary">
      <template v-for="(segment, i) in summary.segments" :key="i">
        <b v-if="segment.mono" class="font-mono text-xs font-semibold text-fg">{{
          segment.text
        }}</b>
        <span v-else>{{ segment.text }}</span>
      </template>
    </span>
    <span v-else data-test="trash-banner-summary">No connected machine keeps a trash.</span>
    <span class="flex-1" />
    <span class="font-mono text-micro text-fg-dim" data-test="trash-banner-count">{{
      countLabel
    }}</span>
    <button
      v-if="hosts.length > 0"
      type="button"
      class="rounded-control px-1 text-xs text-accent underline-offset-2 hover:underline"
      data-test="trash-banner-link"
      @click="emit('changeRetention')"
    >
      {{ linkLabel }}
    </button>
  </div>
</template>
