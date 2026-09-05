<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import Icon from "@ui/components/Icon.vue";
import { shortcutLabel } from "../../lib/platform";

/*
 * ONE rewrite control on the composer's row: the mock's 28px chip — sparkle,
 * the words, the mono chord — with Remix, its source choice and the undo
 * folded behind the chip's caret. Two 26px toolbar buttons plus a Source
 * <select> put ~140px of secondary action on the same baseline as the 28px
 * Style/Shape/Make chips and wrapped the control row, which dropped Generate
 * to a second line and made the composer taller. Remix is a variation on the
 * same verb, so it belongs UNDER it, not beside it.
 *
 * The menu opens upward in place, the way the Style picker's does: the
 * composer is the bottom bar of a workbench that never scrolls, so there is
 * no scrollable ancestor for an absolute panel to extend (HostChip's was in
 * the scrolling inspector, which is why that one is a Popover).
 */
const props = defineProps<{
  prompt: string;
  batchSize: number;
  running: boolean;
  hostLabel: string | null;
  canUndo: boolean;
  blocked?: boolean;
  /**
   * Why the recipe itself refuses a prompt rewrite (`capabilities.prompt.mode:
   * "ignored"` — the family has no text encoder). Both transforms render
   * disabled with this sentence as their tooltip and a visible hint beside
   * them, and the exposed keyboard action stays silent: the view answers the
   * shortcut with the same reason.
   */
  transformBlockedReason?: string | null;
  originalAvailable?: boolean;
  remixSource?: "original" | "current";
}>();

const emit = defineEmits<{
  (e: "expand"): void;
  (e: "remix"): void;
  (e: "update:remixSource", value: "original" | "current"): void;
  (e: "restore"): void;
}>();

const isPreparedBatch = computed(() => props.batchSize > 1);
const actionLabel = computed(() =>
  isPreparedBatch.value ? `Prepare ${props.batchSize} variations` : "Write more for me",
);
const progressLabel = computed(() => {
  const machine = props.hostLabel ?? "the selected machine";
  return isPreparedBatch.value
    ? `Writing ${props.batchSize} versions on ${machine}…`
    : `Writing more on ${machine}…`;
});

/** One refusal rule for the verb and everything folded under it. */
const transformDisabled = computed(
  () => !!props.transformBlockedReason || !!props.blocked || props.running || !props.prompt.trim(),
);
const expandTitle = computed(() => {
  if (props.transformBlockedReason) return props.transformBlockedReason;
  if (props.blocked) return "Refresh or discard the preserved prepared batch first";
  return isPreparedBatch.value
    ? `Prepare ${props.batchSize} prompt variations`
    : "Write more for me";
});
const remixTitle = computed(() => {
  if (props.transformBlockedReason) return props.transformBlockedReason;
  return isPreparedBatch.value
    ? `Prepare ${props.batchSize} subject-preserving prompt remixes`
    : "Remix this prompt in place";
});

const SOURCE_CHOICES = [
  { value: "original" as const, label: "Original idea" },
  { value: "current" as const, label: "Current prompt" },
];
const activeSource = computed(() => props.remixSource ?? "original");

const rootEl = ref<HTMLElement | null>(null);
const menuOpen = ref(false);

function onPointerDown(event: PointerEvent) {
  if (!rootEl.value) return;
  if (!event.composedPath().includes(rootEl.value)) menuOpen.value = false;
}
function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") menuOpen.value = false;
}
watch(menuOpen, (open) => {
  if (open) {
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKeydown, true);
  } else {
    document.removeEventListener("pointerdown", onPointerDown, true);
    document.removeEventListener("keydown", onKeydown, true);
  }
});
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onPointerDown, true);
  document.removeEventListener("keydown", onKeydown, true);
});

function expand() {
  if (props.transformBlockedReason) return;
  if (!props.blocked && !props.running && props.prompt.trim()) emit("expand");
}

function remix() {
  if (transformDisabled.value) return;
  menuOpen.value = false;
  emit("remix");
}

function pickSource(value: "original" | "current") {
  emit("update:remixSource", value);
}

function restore() {
  menuOpen.value = false;
  emit("restore");
}

defineExpose({ expand });
</script>

<template>
  <div ref="rootEl" class="ms-expand">
    <div class="ms-rewrite" :class="{ 'ms-rewrite--disabled': transformDisabled }">
      <button
        type="button"
        data-test="expand-action"
        class="ms-rewrite__verb"
        :disabled="transformDisabled"
        :title="expandTitle"
        @click="expand"
      >
        <Icon name="sparkle" :size="14" />
        {{ actionLabel }}
        <kbd v-if="!running" class="ms-rewrite__chord">{{ shortcutLabel("E") }}</kbd>
      </button>
      <button
        type="button"
        data-test="rewrite-more"
        class="ms-rewrite__caret"
        title="More ways to rewrite"
        aria-label="More ways to rewrite"
        :aria-expanded="menuOpen"
        aria-haspopup="menu"
        @click="menuOpen = !menuOpen"
      >
        <Icon name="chevron-down" :size="12" />
      </button>

      <div
        v-if="menuOpen"
        data-test="rewrite-menu"
        class="ms-rewrite__menu"
        role="menu"
        aria-label="More ways to rewrite"
      >
        <button
          type="button"
          role="menuitem"
          data-test="remix-action"
          class="ms-rewrite__row ms-rewrite__row--stacked"
          :disabled="transformDisabled"
          :title="remixTitle"
          @click="remix"
        >
          <span class="ms-rewrite__row-label">Remix</span>
          <span class="ms-rewrite__row-sub">Keep the subject, change the telling</span>
        </button>

        <template v-if="originalAvailable">
          <div class="ms-rewrite__rule" />
          <div class="ms-rewrite__kicker font-mono">remix from</div>
          <button
            v-for="choice in SOURCE_CHOICES"
            :key="choice.value"
            type="button"
            role="menuitemradio"
            :data-test="`remix-source-${choice.value}`"
            class="ms-rewrite__row"
            :disabled="!!transformBlockedReason"
            :aria-checked="activeSource === choice.value"
            @click="pickSource(choice.value)"
          >
            <span class="ms-rewrite__row-label">{{ choice.label }}</span>
            <span v-if="activeSource === choice.value" class="ms-rewrite__check">✓</span>
          </button>
        </template>

        <template v-if="!isPreparedBatch && canUndo">
          <div class="ms-rewrite__rule" />
          <button
            type="button"
            role="menuitem"
            data-test="restore-original"
            class="ms-rewrite__row"
            title="Restore original prompt"
            aria-label="Restore original prompt"
            @click="restore"
          >
            <span class="ms-rewrite__row-label">↩ Restore original</span>
          </button>
        </template>
      </div>
    </div>

    <span
      v-if="transformBlockedReason"
      data-test="transform-blocked-hint"
      class="ms-expand__note"
      >{{ transformBlockedReason }}</span
    >

    <span
      v-else-if="running"
      role="status"
      aria-live="polite"
      class="ms-expand__note ms-expand__note--live"
    >
      {{ progressLabel }}
    </span>
  </div>
</template>

<style scoped>
.ms-expand {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
}

/* The mock's chip: 28px, on the same baseline as Style, Shape and Make —
   never the 26px toolbar button, which read as a different class of control
   sitting in the middle of the chip row. */
.ms-rewrite {
  position: relative;
  display: inline-flex;
  align-items: stretch;
  flex-shrink: 0;
  height: 28px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  color: var(--mold-text-2);
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-rewrite:hover:not(.ms-rewrite--disabled) {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}
.ms-rewrite--disabled {
  color: var(--mold-text-faint);
}

.ms-rewrite__verb {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  border: 0;
  background: transparent;
  color: inherit;
  font-size: var(--mold-fs-xs);
  font-weight: 500;
  white-space: nowrap;
  cursor: pointer;
}
.ms-rewrite__verb:disabled {
  cursor: default;
}

.ms-rewrite__chord {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.ms-rewrite__caret {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  border: 0;
  border-left: var(--mold-bw) solid var(--mold-border);
  background: transparent;
  color: var(--mold-text-dim);
  cursor: pointer;
}
.ms-rewrite__caret:hover {
  color: var(--mold-text);
}

/* Opens upward over the canvas, like the Style picker's: the composer is the
   bottom bar, so downward is off the window. */
.ms-rewrite__menu {
  position: absolute;
  bottom: calc(100% + 8px);
  left: 0;
  z-index: 30;
  width: 236px;
  padding: 8px;
  background: var(--mold-bg);
  border: var(--mold-bw) solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  box-shadow: 0 18px 50px rgb(0 0 0 / 45%);
}
.ms-rewrite__kicker {
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mold-text-dim);
  padding: 4px 8px 6px;
}
.ms-rewrite__row {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 8px;
  border: 0;
  background: transparent;
  color: var(--mold-text);
  padding: 8px;
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-xs);
  text-align: left;
  cursor: pointer;
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-rewrite__row--stacked {
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
}
.ms-rewrite__row:hover:not(:disabled) {
  background: var(--mold-surface);
}
.ms-rewrite__row:disabled {
  opacity: 0.55;
  cursor: default;
}
.ms-rewrite__row-label {
  flex: 1;
  min-width: 0;
}
.ms-rewrite__row-sub {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  line-height: var(--mold-lh-snug);
}
.ms-rewrite__check {
  color: var(--mold-blue);
}
.ms-rewrite__rule {
  height: var(--mold-bw);
  background: var(--mold-border);
  margin: 4px 8px;
}

.ms-expand__note {
  min-width: 0;
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.ms-expand__note--live {
  color: var(--mold-blue);
}
</style>
