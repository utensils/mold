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
import { RouterLink } from "vue-router";
import {
  isModelRuntimeUnavailable,
  RUNTIME_UNAVAILABLE_BADGE,
} from "@studio/lib/modelRuntimeAvailability";
import { modelDisplayNameForId } from "@studio/lib/modelDisplay";
import { styleDisplayName } from "@studio/lib/styleLabel";
import StyleMenu from "@studio/components/StyleMenu.vue";
import Icon from "@ui/components/Icon.vue";
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
  <div class="mp" data-test="create-style-picker">
    <div class="mp__head">
      <span class="mp__kicker">Style</span>
      <RouterLink
        :to="browseTarget"
        class="mp__browse"
        data-test="browse-styles"
      >
        Browse more
        <Icon name="chevron-right" :size="12" />
      </RouterLink>
    </div>
    <Popover
      v-model:open="open"
      label="Style"
      placement="bottom-start"
      class="mp__pop"
    >
      <template #trigger>
        <button
          type="button"
          class="mp__chip"
          data-test="style-chip"
          :aria-expanded="open"
          :aria-controls="menuId || undefined"
          aria-haspopup="listbox"
          @click="open = !open"
          @keydown="onChipKeydown"
        >
          <Icon name="layers" :size="13" />
          <span data-test="selected-model-name" class="mp__chip-label">
            {{ styleName || styleId || "Choose a style" }}
          </span>
          <span
            v-if="styleId && styleId !== styleName"
            data-test="style-chip-id"
            class="mp__chip-id"
            >{{ styleId }}</span
          >
          <span class="mp__chip-caret" aria-hidden="true">▼</span>
        </button>
      </template>
      <StyleMenu
        ref="menu"
        class="mp__menu"
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
  </div>
</template>

<style scoped>
.mp {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 16px 18px;
}
.mp__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.mp__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.mp__browse {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
  text-decoration: none;
}
.mp__browse:hover {
  color: var(--safelight);
}
.mp__pop {
  display: block;
  width: 100%;
}
.mp__pop :deep(.ms-popover__trigger) {
  display: block;
  width: 100%;
}
.mp__chip {
  display: flex;
  width: 100%;
  box-sizing: border-box;
  align-items: center;
  gap: 6px;
  min-height: 40px;
  padding: 0 12px;
  cursor: pointer;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  font-size: 13px;
  text-align: left;
}
.mp__chip:hover {
  border-color: var(--safelight);
}
.mp__chip-label {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mp__chip-id {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}
.mp__chip-caret {
  margin-left: auto;
  font-size: 10px;
  color: var(--ink-3);
}
/* The popover panel is already the surface: the menu inside it only needs a
   height bound so a long list scrolls instead of running off the viewport. */
.mp__menu {
  max-height: 22rem;
  overflow-y: auto;
  min-width: 18rem;
}
</style>
