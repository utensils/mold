<script setup lang="ts">
import { ref } from "vue";
import type { FleetActiveWork } from "@studio/api/activity";

const props = withDefaults(
  defineProps<{
    rows: FleetActiveWork[];
    compact?: boolean;
    interactive?: boolean;
    swipeActions?: boolean;
    actionWidth?: number;
    canSwipe?: (row: FleetActiveWork) => boolean;
  }>(),
  {
    compact: false,
    interactive: false,
    swipeActions: false,
    actionWidth: 72,
    canSwipe: () => true,
  },
);

const emit = defineEmits<{
  select: [row: FleetActiveWork];
  contextmenu: [row: FleetActiveWork, event: MouseEvent];
}>();

const openActionKey = ref<string | null>(null);
let swipe: { key: string; x: number; y: number; horizontal: boolean } | null =
  null;

function beginSwipe(row: FleetActiveWork, event: TouchEvent): void {
  if (!props.swipeActions || !props.canSwipe(row) || event.touches.length !== 1)
    return;
  const touch = event.touches[0];
  if (touch)
    swipe = {
      key: row.key,
      x: touch.clientX,
      y: touch.clientY,
      horizontal: false,
    };
}

function moveSwipe(event: TouchEvent): void {
  const touch = event.touches[0];
  if (!swipe || !touch) return;
  const dx = touch.clientX - swipe.x;
  const dy = touch.clientY - swipe.y;
  if (!swipe.horizontal && Math.abs(dx) > 10 && Math.abs(dx) > Math.abs(dy)) {
    swipe.horizontal = true;
  }
  if (swipe.horizontal) event.preventDefault();
}

function finishSwipe(event: TouchEvent): void {
  const gesture = swipe;
  swipe = null;
  const touch = event.changedTouches[0];
  if (!gesture || !gesture.horizontal || !touch) return;
  const dx = touch.clientX - gesture.x;
  if (dx <= -36) openActionKey.value = gesture.key;
  else if (dx >= 36) openActionKey.value = null;
}

function cancelSwipe(): void {
  swipe = null;
}

function select(row: FleetActiveWork): void {
  if (openActionKey.value === row.key) {
    openActionKey.value = null;
    return;
  }
  if (props.interactive) emit("select", row);
}

function title(row: FleetActiveWork): string {
  if (row.kind === "download")
    return row.model ? `Pulling ${row.model}` : "Model download";
  if (row.kind === "sequence")
    return row.model ? `${row.model} sequence` : "Sequence";
  if (row.kind === "generation") return row.model ?? "Generation";
  return row.model ?? row.kind.replaceAll("_", " ");
}

function progress(row: FleetActiveWork): number | null {
  if (!row.total || row.current == null) return null;
  return Math.max(
    0,
    Math.min(100, Math.round((row.current / row.total) * 100)),
  );
}

function phase(row: FleetActiveWork): string {
  if (row.phase === "preparing" && row.preparation_progress?.component) {
    return `Preparing · ${row.preparation_progress.component}`;
  }
  if (row.stage) return row.stage;
  return row.phase.replaceAll("_", " ");
}
</script>

<template>
  <ol
    v-if="rows.length"
    class="live-activity-list"
    :class="{ 'live-activity-list--compact': compact }"
    data-test="shared-live-activity"
  >
    <li
      v-for="row in rows"
      :key="row.key"
      class="live-activity-row"
      :class="{
        'live-activity-row--has-actions': swipeActions && canSwipe(row),
        'live-activity-row--actions-open': openActionKey === row.key,
      }"
      :style="
        swipeActions && canSwipe(row)
          ? { '--live-activity-action-width': `${actionWidth}px` }
          : undefined
      "
      :data-stale="row.stale"
      @touchstart="beginSwipe(row, $event)"
      @touchmove="moveSwipe"
      @touchend="finishSwipe"
      @touchcancel="cancelSwipe"
    >
      <component
        :is="interactive ? 'button' : 'div'"
        :type="interactive ? 'button' : undefined"
        class="live-activity-surface"
        :data-test="interactive ? `live-activity-select-${row.key}` : undefined"
        @click="select(row)"
        @contextmenu="interactive && emit('contextmenu', row, $event)"
      >
        <span
          class="live-activity-dot"
          :data-phase="row.phase"
          aria-hidden="true"
        />
        <span class="live-activity-copy">
          <strong>{{ title(row) }}</strong>
          <span>
            {{ row.hostLabel }} · {{ phase(row) }}
            <template v-if="progress(row) !== null">
              · {{ progress(row) }}%</template
            >
            <template v-if="row.stale">
              · Last seen active ·
              {{ row.hostError ?? "Waiting to reconnect" }}</template
            >
          </span>
        </span>
      </component>
      <div
        v-if="swipeActions && canSwipe(row)"
        class="live-activity-actions"
        @click.stop
      >
        <slot name="actions" :row="row" />
      </div>
    </li>
  </ol>
</template>

<style scoped>
.live-activity-list {
  display: grid;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.live-activity-row {
  position: relative;
  min-width: 0;
}
.live-activity-row--has-actions {
  overflow: hidden;
  touch-action: pan-y;
}
.live-activity-surface {
  position: relative;
  z-index: 1;
  display: flex;
  width: 100%;
  align-items: center;
  text-align: left;
  gap: 9px;
  min-height: 44px;
  padding: 7px 10px;
  border: 1px solid var(--edge);
  border-radius: 9px;
  background: var(--surface);
  color: inherit;
  font: inherit;
  transition: transform 180ms ease;
}
.live-activity-row--actions-open .live-activity-surface {
  transform: translateX(calc(var(--live-activity-action-width) * -1));
}
.live-activity-actions {
  position: absolute;
  z-index: 0;
  top: 0;
  right: 0;
  bottom: 0;
  width: var(--live-activity-action-width);
}
.live-activity-actions :deep(button) {
  width: 100%;
  height: 100%;
  min-height: 44px;
  border: 1px solid var(--edge);
  border-radius: 9px;
  background: color-mix(in srgb, var(--safelight) 18%, var(--surface));
  color: inherit;
  font: inherit;
}
.live-activity-row[data-stale="true"] {
  opacity: 0.72;
  border-style: dashed;
}
.live-activity-surface:is(button) {
  cursor: pointer;
}
.live-activity-surface:is(button):hover {
  border-color: color-mix(in srgb, var(--safelight) 36%, var(--edge));
  background: color-mix(in srgb, var(--safelight) 7%, var(--surface));
}
.live-activity-dot {
  width: 7px;
  height: 7px;
  flex: none;
  border-radius: 50%;
  background: var(--safelight);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--safelight) 16%, transparent);
}
.live-activity-dot[data-phase="queued"],
.live-activity-dot[data-phase="preparing"] {
  background: var(--ink-3);
  box-shadow: none;
}
.live-activity-copy {
  display: grid;
  min-width: 0;
  gap: 2px;
}
.live-activity-copy strong {
  overflow: hidden;
  font-size: 12px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.live-activity-copy span {
  color: var(--ink-3);
  font-size: 11px;
  text-transform: capitalize;
}
.live-activity-list--compact .live-activity-surface {
  min-height: 38px;
  padding: 5px 7px;
  background: transparent;
}
.live-activity-list--compact .live-activity-copy strong {
  font-size: 11.5px;
}
.live-activity-list--compact .live-activity-copy span {
  font-size: 9.5px;
}
</style>
