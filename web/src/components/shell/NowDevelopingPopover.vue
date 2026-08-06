<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import LiveActivityList from "@ui/components/LiveActivityList.vue";
import type { FleetActiveWork } from "@studio/api/activity";

defineProps<{ rows: FleetActiveWork[] }>();
const emit = defineEmits<{ select: [row: FleetActiveWork] }>();

const open = ref(false);

function closeOnEscape(event: KeyboardEvent) {
  if (event.key === "Escape") open.value = false;
}

onMounted(() => window.addEventListener("keydown", closeOnEscape));
onBeforeUnmount(() => window.removeEventListener("keydown", closeOnEscape));
</script>

<template>
  <div v-if="rows.length" class="now-developing">
    <button
      type="button"
      class="now-developing__trigger"
      data-test="now-developing-trigger"
      :aria-expanded="open"
      aria-controls="now-developing-panel"
      @click="open = !open"
    >
      <Icon name="create" :size="15" />
      <span class="now-developing__label">Now developing</span>
      <BadgePill tone="accent">{{ rows.length }}</BadgePill>
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
  padding: 5px 9px;
  border: 1px solid var(--edge);
  border-radius: var(--radius-control);
  background: var(--surface);
  color: var(--ink-2);
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
    display: none;
  }
}
</style>
