<script setup lang="ts">
/*
 * TrashBanner — the 40px retention strip under the Library header in the
 * Trash scope. Copy comes from the shared `trashRetentionSummary` (This device
 * sets the sentence, hosts that differ are named) with every number set in the
 * mono face; the right edge carries the count, the retention as a mono control
 * that routes to where it is actually edited (per host, so it is a link out),
 * and **Empty now**.
 */
import { computed } from "vue";
import Icon from "@ui/components/Icon.vue";
import { trashRetentionSummary, type RetentionHost } from "@studio/lib/libraryOrganization";
import { formatBytes } from "../../lib/format";

/** The banner's rows carry the host key the retention control routes on; the
 *  shared summary only needs the label and the number. */
type BannerRetentionHost = RetentionHost & { key?: string };

const props = withDefaults(
  defineProps<{
    /** Trash-capable hosts. This device leads WHEN it keeps a trash: a host
     *  with none is skipped, so the first row can be a remote. */
    hosts: BannerRetentionHost[];
    /** Logical prints in the trash across hosts. */
    count: number;
    /** Bytes across the trash, when known. */
    bytes?: number | null;
    /** Where the retention is edited ("Settings" / "Machines"). */
    linkLabel?: string;
    busy?: boolean;
  }>(),
  { bytes: null, linkLabel: "Change retention · Machines", busy: false },
);

const emit = defineEmits<{ changeRetention: []; emptyNow: [] }>();

const summary = computed(() => trashRetentionSummary(props.hosts));

const countLabel = computed(() => {
  const noun = props.count === 1 ? "picture" : "pictures";
  const base = `${props.count} ${noun} in trash`;
  return props.bytes != null && props.bytes > 0 ? `${base} · ${formatBytes(props.bytes)}` : base;
});

/** The machine the control edits: This device where it keeps a trash — that
 *  is the one the change is routed to — otherwise the first machine listed.
 *  Machines that differ are named in the sentence beside it. */
const retentionHost = computed(
  () => props.hosts.find((host) => host.key === "local") ?? props.hosts[0] ?? null,
);
const retentionLabel = computed(() => {
  const days = retentionHost.value?.retentionDays ?? 0;
  return days > 0 ? `Keep for ${days} ${days === 1 ? "day" : "days"} ▼` : "Keep forever ▼";
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
      class="rounded-control px-1 font-mono text-micro text-fg-dim hover:text-fg"
      data-test="trash-banner-link"
      :title="linkLabel"
      @click="emit('changeRetention')"
    >
      {{ retentionLabel }}
    </button>
    <button
      type="button"
      class="ms-toolbar-button ms-toolbar-button--danger"
      data-test="empty-trash"
      :disabled="count === 0 || busy"
      @click="emit('emptyNow')"
    >
      Empty now
    </button>
  </div>
</template>
