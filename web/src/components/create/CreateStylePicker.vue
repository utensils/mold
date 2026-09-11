<script setup lang="ts">
/*
 * Web's Create style control — the chip that names the style and the SHARED
 * `@studio/components/StyleMenu.vue` it opens.
 *
 * It replaced a native `<select>` grouped by family, which could say a family
 * name and an id and nothing else: no size, no on-GPU state, no description,
 * no type-to-filter, and no way to see that the restored style is simply not
 * on any machine yet. Desktop already had all of that, so this hosts the same
 * list rather than growing a second one.
 *
 * `@ui/components/Popover.vue` is the host because web's composer sits in a
 * SCROLLING page: the panel teleports to <body> as `position: fixed`, so
 * merely opening it cannot grow an ancestor's scrollbar. Two of its properties
 * shape this file — it does not forward arrow/Enter keys (StyleMenu owns
 * keydown on its own root, which is why the menu still walks), and it eats
 * Escape in the capture phase (which is why the menu never claimed it).
 */
import { computed, nextTick, ref, watch } from "vue";
import {
  isModelRuntimeUnavailable,
  RUNTIME_UNAVAILABLE_BADGE,
} from "@studio/lib/modelRuntimeAvailability";
import { modelDisplayNameForId } from "@studio/lib/modelDisplay";
import { styleDisplayName } from "@studio/lib/styleLabel";
import StyleMenu from "@studio/components/StyleMenu.vue";
import Popover from "@ui/components/Popover.vue";
import type { ModelInfoExtended } from "../../types";

const props = defineProps<{
  models: ModelInfoExtended[];
  model: string;
  browseTo?: string;
  emptyLabel?: string;
  /**
   * The form's model when NO reachable machine has it (the parent knows the
   * whole fleet's inventory; this list may be narrowed by output kind). The
   * menu keeps it as a phantom row so a restored print's style stays visible
   * with the way to get it, instead of reading as if it had been dropped.
   */
  missingModel?: string | null;
  /**
   * Which machines hold each style, already worded by the page through
   * `@studio/lib/modelAvailability` — the rule needs a fleet and its
   * reachability, and neither belongs in a presentational control. Absent (a
   * single-server browser) renders nothing at all.
   */
  availabilityTag?: ((model: ModelInfoExtended) => string | null) | null;
}>();

const emit = defineEmits<{
  select: [model: ModelInfoExtended];
  browse: [to: string];
}>();

const open = ref(false);
/** The shared menu is a GENERIC SFC, so it has no `InstanceType`: name the two
 *  things this host calls on it. */
const menu = ref<{
  handleKeydown: (event: KeyboardEvent) => void;
  focus: () => void;
  $el?: HTMLElement;
} | null>(null);
const menuId = ref("");

/*
 * The popover teleports its panel to <body> and never moves focus into it, and
 * a list of eight or fewer styles carries no filter field to land on — so the
 * menu is focused explicitly once it exists. The chip ALSO forwards its own
 * keys (`onChipKeydown`), because focus can be back on the chip: the popover
 * returns it there on Escape, and a click on the chip never left it.
 */
watch(open, (isOpen) => {
  if (!isOpen) {
    menuId.value = "";
    return;
  }
  void nextTick(() => {
    menu.value?.focus();
    menuId.value = menu.value?.$el?.id ?? "";
  });
});

/** ↑/↓/Enter belong to the menu; Escape is the popover's, in capture. A key
 *  the menu already handled inside its own root bubbles here too, and
 *  `defaultPrevented` is how it says so — without that check one press would
 *  walk two rows. */
function onChipKeydown(event: KeyboardEvent) {
  if (!open.value || event.defaultPrevented) return;
  menu.value?.handleKeydown(event);
}

const current = computed(
  () => props.models.find((m) => m.name === props.model) ?? null,
);

/** Plain name first. `styleDisplayName` already stands the family's friendly
 *  label in for a bare manifest id, so the id is never said twice. */
const styleName = computed(() => {
  if (current.value) return styleDisplayName(current.value);
  if (!props.model) return "";
  const name = modelDisplayNameForId(props.model, props.models);
  return name === props.model ? "" : name;
});

const styleId = computed(() => current.value?.name ?? props.model ?? "");

const browseTarget = computed(() => props.browseTo ?? "/models");

/** Downloaded but unrunnable here: kept in the list, refused by name. */
function disabledReason(model: { name: string }): string | null {
  const row = props.models.find((m) => m.name === model.name);
  return row && isModelRuntimeUnavailable(row)
    ? RUNTIME_UNAVAILABLE_BADGE
    : null;
}

function pick(model: { name: string }) {
  const row = props.models.find((m) => m.name === model.name);
  if (!row) return;
  open.value = false;
  emit("select", row);
}

function browse() {
  open.value = false;
  emit("browse", browseTarget.value);
}
</script>

<template>
  <Popover
    v-model:open="open"
    label="Style"
    placement="bottom-start"
    class="style-pop"
    data-test="create-style-picker"
  >
    <template #trigger>
      <button
        type="button"
        class="style-chip"
        data-test="style-chip"
        title="Style"
        :aria-expanded="open"
        :aria-controls="menuId || undefined"
        aria-haspopup="listbox"
        @click="open = !open"
        @keydown="onChipKeydown"
      >
        <span data-test="selected-model-name" class="style-chip__label">
          {{ styleName || styleId || "Choose a style" }}
        </span>
        <span
          v-if="styleId && styleId !== styleName"
          data-test="style-chip-id"
          class="style-chip__id"
          >{{ styleId }}</span
        >
        <span class="style-chip__caret" aria-hidden="true">▼</span>
      </button>
    </template>
    <StyleMenu
      ref="menu"
      class="style-menu"
      :models="models"
      :selected="current"
      :missing-model="missingModel ?? null"
      :empty-label="emptyLabel ?? 'No styles ready'"
      :disabled-reason="disabledReason"
      :availability-tag="availabilityTag ?? null"
      autofocus-filter
      @pick="pick"
      @pick-missing="browse"
      @browse="browse"
    />
  </Popover>
</template>

<style scoped>
/* One chip in the composer's row (mock: `Photoreal flux-dev:q4 ▼` at 28px
   beside the shape and count chips). The shared ShapeChip carries the same
   metrics; the popover host stays because the composer sits in a SCROLLING
   page and the panel must teleport rather than grow an ancestor. */
.style-pop {
  display: inline-flex;
  min-width: 0;
  max-width: 100%;
}
.style-pop :deep(.ms-popover__trigger) {
  display: inline-flex;
  min-width: 0;
  max-width: 100%;
}
.style-chip {
  /* literal: the mock's 28px chip row, the one metric between --mold-ctl-sm
   * (24px, dense icon buttons) and --mold-ctl-md (26px, toolbars). */
  --composer-chip-h: 28px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: var(--composer-chip-h);
  max-width: 100%;
  min-width: 0;
  padding: 0 10px;
  white-space: nowrap;
  cursor: pointer;
  background: var(--mold-surface);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  color: var(--mold-text);
  font-size: var(--mold-fs-xs);
  font-weight: 500;
  text-align: left;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.style-chip:hover {
  border-color: var(--mold-border-focus);
}
.style-chip__label {
  min-width: 0;
  /* A catalog display name can run to a sentence ("FLUX.1 Schnell Q8 — fast
   * 4-step, general purpose"); the chip keeps the words that name it and the
   * menu row carries the rest. */
  max-width: 22ch;
  overflow: hidden;
  text-overflow: ellipsis;
}
.style-chip__id {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  font-weight: 400;
  color: var(--mold-text-dim);
}
.style-chip__caret {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
/* The popover panel is already the surface: the menu inside it only needs a
   height bound so a long list scrolls instead of running off the viewport. */
.style-menu {
  max-height: 22rem;
  overflow-y: auto;
  min-width: 18rem;
}
</style>
