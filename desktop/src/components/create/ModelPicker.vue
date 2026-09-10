<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import type { ModelEntry } from "../../lib/api/types";
import { modelAvailabilityTag } from "../../lib/hosts";
import { modelDisplayName, modelDisplayNameForId } from "../../lib/models";
import { modelSource } from "@studio/lib/modelSource";
import { formatGB } from "../../lib/format";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import StyleMenu from "@studio/components/StyleMenu.vue";
import SourceGlyph from "../generate/SourceGlyph.vue";

/**
 * The Mold Studio installed-model picker — the ONE style picker on Create.
 *
 * This is the POPOVER SHELL: open state, placement, outside-pointerdown and
 * Escape dismissal, the availability-refresh on open, the router push behind
 * Browse more, and the desktop-only source glyphs. The LIST inside it is
 * `@studio/components/StyleMenu.vue`, shared byte-for-byte with web's Create
 * chip and the phone's Style sheet — one menu, three hosts.
 *
 * The trigger is a SLOT: the composer's Style chip opens it in place, above
 * the composer (`placement="up"`), so there is no second selector anywhere.
 * The default trigger stays for any consumer that wants a plain field.
 */
const props = withDefaults(
  defineProps<{
    models: ModelEntry[];
    selected: ModelEntry | null;
    /** Multi-host availability tags; parents suppress them for a sticky host. */
    showAvailability?: boolean;
    /** Non-null marks the entry unpickable and explains why, inline. */
    disabledReason?: ((m: ModelEntry) => string | null) | null;
    browseTarget?: string;
    browseLabel?: string;
    /**
     * A model id the form carries that no machine has installed — a restored
     * print, a template, or a deleted checkpoint. It renders as the selected
     * entry with a "Not on this machine" tag instead of reading "Choose a
     * style", which made the restore look like it had silently dropped the
     * model.
     */
    missingModel?: string | null;
    /**
     * Which way the menu opens. The composer sits on the bottom edge of the
     * canvas, so its chip must open UPWARD or the menu falls off the window.
     */
    placement?: "down" | "up";
    /**
     * The mono kicker naming what this menu holds — the New image view's
     * section ("still picture styles"). Absent on a consumer that offers
     * every style.
     */
    kicker?: string | null;
    /**
     * The sentence for a menu whose whole list is empty, which is a different
     * fact from a filter that matched nothing: the section holds no styles at
     * all, and Browse more below is the way out. Absent falls back to the
     * generic line.
     */
    emptyLabel?: string | null;
  }>(),
  {
    showAvailability: true,
    disabledReason: null,
    browseTarget: "/models",
    browseLabel: "Browse more →",
    missingModel: null,
    placement: "down",
    kicker: null,
    emptyLabel: null,
  },
);

const emit = defineEmits<{ pick: [model: ModelEntry]; "pick-missing": [model: string] }>();

const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const router = useRouter();

const pickerEl = ref<HTMLDivElement | null>(null);
/** The shared menu is a GENERIC SFC, so it has no `InstanceType`: name the
 *  one thing this shell calls on it. */
const menuEl = ref<{ handleKeydown: (event: KeyboardEvent) => void } | null>(null);
const open = ref(false);

/** The phantom entry is only shown when nothing real is selected. */
const phantom = computed(() => (props.selected ? null : (props.missingModel ?? null)));
const phantomLabel = computed(() =>
  phantom.value ? modelDisplayNameForId(phantom.value, props.models) : "",
);

/** The desktop rule, injected: the shared menu never learns what a host is. */
function availabilityTag(m: ModelEntry): string | null {
  if (!hosts.multiHost || !props.showAvailability) return null;
  return modelAvailabilityTag(hostModels.hostsFor(m.name), hosts.all);
}

function toggle() {
  open.value = !open.value;
}
function close() {
  open.value = false;
}

function pick(m: ModelEntry) {
  emit("pick", m);
  close();
}

function pickMissing(name: string) {
  close();
  emit("pick-missing", name);
}

function browse() {
  close();
  void router.push(props.browseTarget);
}

/**
 * Escape is the shell's, ↑/↓/Enter are the menu's.
 *
 * Keys reach here from whatever trigger has focus, which for a list too short
 * to carry a filter field is the chip itself — so they are forwarded into the
 * menu. A key the menu already handled inside its own root bubbles up here
 * too; `defaultPrevented` is how it says so, and stops the row walking twice.
 */
function onKeydown(event: KeyboardEvent) {
  if (!open.value) return;
  if (event.key === "Escape") {
    event.preventDefault();
    close();
    return;
  }
  if (event.defaultPrevented) return;
  menuEl.value?.handleKeydown(event);
}

function onDocumentPointerDown(event: PointerEvent) {
  if (!open.value || !pickerEl.value) return;
  if (!event.composedPath().includes(pickerEl.value)) close();
}
function onDocumentKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") close();
}

// Force-fresh availability when the picker opens — a model pulled on an
// extra host by another client shows up the moment the user looks.
watch(open, (isOpen) => {
  if (isOpen) void hostModels.refresh(true);
});

onMounted(() => {
  document.addEventListener("pointerdown", onDocumentPointerDown);
  document.addEventListener("keydown", onDocumentKeydown);
});
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onDocumentPointerDown);
  document.removeEventListener("keydown", onDocumentKeydown);
});
</script>

<template>
  <div ref="pickerEl" class="ms-model" data-test="model-picker" @keydown="onKeydown">
    <!-- The composer's Style chip fills this; the plain field is the default. -->
    <slot name="trigger" :open="open" :toggle="toggle">
      <button type="button" :aria-expanded="open" class="ms-model__button" @click="toggle">
        <span data-test="selected-model-name" class="min-w-0 break-all text-left">{{
          selected ? modelDisplayName(selected) : phantom ? phantomLabel : "Choose a style"
        }}</span>
        <span v-if="selected?.disk_usage_bytes" class="font-mono text-xs ms-model__size">
          {{ formatGB(selected.disk_usage_bytes) }}
        </span>
        <span
          v-else-if="phantom"
          data-test="selected-model-missing"
          class="font-mono text-micro text-fg-dim whitespace-nowrap shrink-0"
        >
          Not on this machine
        </span>
      </button>
    </slot>
    <StyleMenu
      v-if="open"
      ref="menuEl"
      :class="placement === 'up' ? 'ms-model__menu--up' : 'ms-model__menu--down'"
      :data-placement="placement"
      :models="models"
      :selected="selected"
      :missing-model="missingModel"
      :kicker="kicker"
      :empty-label="emptyLabel"
      :disabled-reason="disabledReason"
      :availability-tag="availabilityTag"
      :browse-label="browseLabel"
      autofocus-filter
      @pick="pick"
      @pick-missing="pickMissing"
      @browse="browse"
    >
      <template #glyph="{ model }">
        <SourceGlyph :source="modelSource(model)" class="mt-0.5 shrink-0 text-fg-dim" />
      </template>
    </StyleMenu>
  </div>
</template>

<style scoped>
.ms-model {
  position: relative;
}
.ms-model__button {
  display: flex;
  cursor: pointer;
  min-height: 40px;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-3);
  background: var(--mold-bg-deep);
  padding: 0 12px;
  font-size: var(--mold-fs-sm);
  color: var(--mold-text);
}
.ms-model__size {
  flex-shrink: 0;
  color: var(--mold-text-dim);
}
/* The surface the shared list sits on. Scoped CSS reaches a child component's
 * ROOT element, which is exactly what the menu is. */
.ms-model__menu--down,
.ms-model__menu--up {
  position: absolute;
  z-index: 30;
  max-height: 22rem;
  width: 100%;
  min-width: 20rem;
  overflow-y: auto;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  background: var(--mold-bg);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
}
.ms-model__menu--down {
  top: 100%;
  margin-top: 4px;
}
/* The composer is on the bottom edge of the canvas: downward would leave the
 * window. Anchored to the trigger's top, growing up. */
.ms-model__menu--up {
  bottom: 100%;
  margin-bottom: 4px;
}
</style>
