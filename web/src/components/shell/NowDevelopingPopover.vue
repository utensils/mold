<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import LiveActivityList from "@ui/components/LiveActivityList.vue";
import type { FleetActiveWork } from "@studio/api/activity";
import { queueSection } from "@studio/lib/queueSections";

const props = defineProps<{ rows: FleetActiveWork[] }>();

/** The chip says what the queue says — "Making 1 · 3 waiting" — from the one
 *  shared classifier, so the header and the Queue page cannot disagree. */
const chipLabel = computed(() => {
  let making = 0;
  let waiting = 0;
  for (const row of props.rows) {
    const section = queueSection(row.phase, row.stale);
    if (section === "waiting") waiting += 1;
    else making += 1;
  }
  const parts: string[] = [];
  if (making > 0) parts.push(`Making ${making}`);
  if (waiting > 0) parts.push(`${waiting} waiting`);
  return parts.join(" · ");
});
const emit = defineEmits<{ select: [row: FleetActiveWork] }>();

const open = ref(false);
const root = ref<HTMLElement | null>(null);

function closeOnEscape(event: KeyboardEvent) {
  if (event.key === "Escape") open.value = false;
}

/** Any interaction outside the panel dismisses it — opening a sibling
 *  popover (the notifications bell) must not leave both stacked. */
function closeOnOutsidePointer(event: PointerEvent) {
  if (!open.value) return;
  if (event.target instanceof Node && root.value?.contains(event.target))
    return;
  open.value = false;
}

onMounted(() => {
  window.addEventListener("keydown", closeOnEscape);
  document.addEventListener("pointerdown", closeOnOutsidePointer, true);
});
onBeforeUnmount(() => {
  window.removeEventListener("keydown", closeOnEscape);
  document.removeEventListener("pointerdown", closeOnOutsidePointer, true);
});
</script>

<template>
  <div v-if="rows.length" ref="root" class="now-developing">
    <button
      type="button"
      class="now-developing__trigger"
      data-test="now-developing-trigger"
      :aria-expanded="open"
      aria-controls="now-developing-panel"
      @click="open = !open"
    >
      <span class="now-developing__dot" aria-hidden="true" />
      <span class="now-developing__label">{{ chipLabel }}</span>
    </button>
    <section
      v-if="open"
      id="now-developing-panel"
      class="now-developing__panel"
      data-test="now-developing-panel"
      aria-label="Now developing"
    >
      <div class="now-developing__heading">Now developing</div>
      <LiveActivityList
        :rows="rows"
        interactive
        @select="
          (row) => {
            open = false;
            emit('select', row);
          }
        "
      />
    </section>
  </div>
</template>

<style scoped>
.now-developing {
  position: relative;
  flex: 0 0 auto;
}
.now-developing__trigger {
  display: flex;
  min-height: 36px;
  align-items: center;
  gap: 7px;
  padding: 5px 11px;
  /* The mock's accent-bordered "Making 1 · 3 waiting" chip beside the
     warning-bordered downloads chip: two kinds of work, two chips. */
  border: 1px solid var(--mold-blue);
  border-radius: var(--radius-control);
  background: var(--surface);
  color: var(--mold-text);
  font-family: var(--f-body);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.now-developing__trigger:hover,
.now-developing__trigger[aria-expanded="true"] {
  border-color: color-mix(in srgb, var(--safelight) 36%, var(--edge));
  background: color-mix(in srgb, var(--safelight) 7%, var(--surface));
}
.now-developing__dot {
  width: 7px;
  height: 7px;
  flex: none;
  border-radius: 50%; /* literal: a dot is the one circle allowed */
  background: var(--mold-blue);
}
.now-developing__panel {
  position: absolute;
  z-index: 40;
  top: calc(100% + 10px);
  right: 0;
  width: min(360px, calc(100vw - 28px));
  max-height: min(520px, calc(100svh - 80px));
  overflow-y: auto;
  padding: 12px;
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  background: var(--bench);
  box-shadow: var(--shadow-raised);
}
.now-developing__heading {
  margin: 1px 2px 9px;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
@media (max-width: 799px) {
  .now-developing__label {
    font-size: var(--mold-fs-micro);
  }
}
</style>
