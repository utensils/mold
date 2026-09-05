<script setup lang="ts">
/*
 * TrashTileActions — what a Trash-scope tile wears on top of its thumbnail
 * in the Trash scope: the "Purges in N d" chip (warning tone on the number only;
 * "Kept" when retention is forever) and, on hover, Restore (default) /
 * Delete forever (secondary). The tile itself stays a plain button — these
 * are absolutely positioned overlays that stop click propagation so the
 * actions never open the print. Pure props/emits.
 */
import { computed } from "vue";
import Icon from "@ui/components/Icon.vue";
import { purgeCountdownFromPurgeAt } from "@studio/lib/libraryOrganization";

const props = withDefaults(
  defineProps<{
    /** Unix seconds the host will purge the print; null = kept forever. */
    purgeAt: number | null | undefined;
    /** Hide the hover buttons (select mode owns the tile). */
    showActions?: boolean;
    busy?: boolean;
    nowMs?: number;
  }>(),
  { showActions: true, busy: false },
);

const emit = defineEmits<{ restore: []; deleteForever: [] }>();

const countdown = computed(() =>
  purgeCountdownFromPurgeAt(props.purgeAt, props.nowMs ?? Date.now()),
);
</script>

<template>
  <span
    class="ms-purge absolute top-2 left-2 rounded-control bg-black/60 px-1.5 py-0.5 font-mono text-micro text-on-media"
    data-test="purge-chip"
    :data-kind="countdown.kind"
    :title="countdown.label"
  >
    <template v-if="countdown.kind === 'purges'">
      Purges in <b class="ms-purge__n">{{ countdown.days }} d</b>
    </template>
    <template v-else-if="countdown.kind === 'today'">
      Purges <b class="ms-purge__n">today</b>
    </template>
    <template v-else>Kept</template>
  </span>
  <span
    v-if="showActions"
    class="absolute right-0 bottom-0 left-0 flex translate-y-full gap-1.5 bg-[linear-gradient(transparent,rgba(0,0,0,0.7))] p-1.5 transition-transform duration-100 group-hover:translate-y-0 group-focus-within:translate-y-0"
    data-test="trash-actions"
    @click.stop
    @dblclick.stop
  >
    <button
      type="button"
      class="ms-ta flex h-6 flex-1 items-center justify-center gap-1 rounded-control border border-white/40 bg-black/45 text-micro text-on-media hover:bg-black/70 disabled:opacity-50"
      data-test="trash-restore"
      :disabled="busy"
      @click.stop="emit('restore')"
    >
      <Icon name="reuse" :size="12" />
      Restore
    </button>
    <button
      type="button"
      class="ms-ta ms-ta--danger flex h-6 flex-1 items-center justify-center rounded-control border bg-black/45 text-micro hover:bg-black/70 disabled:opacity-50"
      data-test="trash-delete-forever"
      :disabled="busy"
      @click.stop="emit('deleteForever')"
    >
      Delete forever
    </button>
  </span>
</template>

<style scoped>
.ms-purge__n {
  color: var(--mold-warning);
  font-weight: 600;
}

.ms-ta--danger {
  border-color: color-mix(in srgb, var(--mold-error) 70%, white);
  color: color-mix(in srgb, var(--mold-error) 35%, white);
}
</style>
