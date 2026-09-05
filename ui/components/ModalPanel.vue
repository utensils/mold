<script setup lang="ts">
/*
 * Centered task dialog (README §04: 480–560px, header / body / footer,
 * radius-3, a 72% crust scrim) — e.g. connect a machine. Optional stepped
 * progress bars along the top. Renders INSIDE its owning frame: absolute
 * overlay, never Teleport or position:fixed. Backdrop click and Esc close;
 * clicks inside the panel do not.
 *
 * Escape is listened for on the DOCUMENT, not on the root: the panel is only
 * `aria-modal`, so focus can be anywhere in the app when the key arrives, and
 * a dialog that closes only while it happens to hold focus is a trap. Tab is
 * kept inside the panel for the same reason.
 *
 * Only the TOP dialog acts. Every open overlay hears the same document-level
 * Escape, so a confirm opened over a lightbox used to close both — the
 * picture the question was about disappeared with the question. The shared
 * register in `../lib/overlayStack` says which one is on top, and the top one
 * stops the key there so nothing underneath ever sees it.
 */
import { onBeforeUnmount, ref, toRef, useSlots, watch } from "vue";
import { useOverlayStack } from "../lib/overlayStack";
import { useRootFocusOnOpen } from "../lib/useRootFocusOnOpen";

const props = withDefaults(
  defineProps<{
    open: boolean;
    /** Panel width in px. */
    width?: number;
    /** Total number of progress bars; omit to hide the row. */
    steps?: number;
    /** Current 1-based step; this and earlier bars are tinted. */
    step?: number;
    /** Accessible name for the dialog. */
    label?: string | undefined;
    /** `alertdialog` for a decision the user cannot get past. */
    role?: "dialog" | "alertdialog";
    /** Header title; with it the header block renders, bordered below. */
    title?: string | undefined;
    /** One plain sentence under the title; the `description` slot overrides it. */
    description?: string | undefined;
  }>(),
  { width: 480, step: 1, role: "dialog" },
);

const emit = defineEmits<{ close: [] }>();

const slots = useSlots();
const root = ref<HTMLElement | null>(null);
useRootFocusOnOpen(root, () => props.open);
const { isTop } = useOverlayStack(toRef(props, "open"), "modal-panel");

const FOCUSABLE =
  'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';

function onDocumentKeydown(e: KeyboardEvent) {
  if (!isTop()) return;
  if (e.key === "Escape") {
    e.preventDefault();
    e.stopImmediatePropagation();
    emit("close");
    return;
  }
  if (e.key !== "Tab" || !root.value) return;
  const stops = [...root.value.querySelectorAll<HTMLElement>(FOCUSABLE)];
  if (stops.length === 0) return;
  const first = stops[0]!;
  const last = stops[stops.length - 1]!;
  const active = document.activeElement;
  if (e.shiftKey && (active === first || active === root.value)) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && active === last) {
    e.preventDefault();
    first.focus();
  } else if (!root.value.contains(active)) {
    e.preventDefault();
    (e.shiftKey ? last : first).focus();
  }
}

watch(
  () => props.open,
  (open) => {
    if (open) document.addEventListener("keydown", onDocumentKeydown, true);
    else document.removeEventListener("keydown", onDocumentKeydown, true);
  },
  { immediate: true },
);
onBeforeUnmount(() =>
  document.removeEventListener("keydown", onDocumentKeydown, true),
);
</script>

<template>
  <div
    v-if="open"
    ref="root"
    class="ms-modal ms-fade-up"
    :role="role"
    aria-modal="true"
    :aria-label="label ?? title"
    tabindex="-1"
    @click="emit('close')"
  >
    <div class="ms-modal__panel" :style="{ width: `${width}px` }" @click.stop>
      <div v-if="steps" class="ms-modal__steps" aria-hidden="true">
        <span
          v-for="i in steps"
          :key="i"
          class="ms-modal__dot"
          :data-on="i <= step ? 'true' : undefined"
        />
      </div>
      <div v-if="title" class="ms-modal__head">
        <span class="ms-modal__title">{{ title }}</span>
        <span v-if="slots.description || description" class="ms-modal__desc">
          <slot name="description">{{ description }}</slot>
        </span>
      </div>
      <div v-if="slots.default" class="ms-modal__body">
        <slot />
      </div>
      <div v-if="slots.footer" class="ms-modal__footer">
        <slot name="footer" />
      </div>
    </div>
  </div>
</template>

<style scoped>
/* INVARIANT: `absolute inset-0` resolves against the nearest POSITIONED
   ancestor, so a dialog fills whatever box that happens to be. Today that is
   the app root (whole window) or a view root that is deliberately `relative`
   (the library pane). Introducing a new positioned wrapper between a dialog
   and the frame it should cover traps it — hoist the dialog to a sibling of
   that wrapper, as the clip timeline's confirms already do. */
.ms-modal {
  position: absolute;
  inset: 0;
  /* Above every in-view layer (the Create bench resizer, the clip lane's
     seam chip and trim grip), below the command palette and the toasts. The
     fallback keeps the rule true on a host without the desktop tokens. */
  z-index: var(--mold-z-modal, 100);
  background: var(--mold-scrim);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
}

.ms-modal__panel {
  max-width: 92%;
  box-sizing: border-box;
  /* The raised-surface role, so cards inside a dialog still read as cards.
     Without the desktop role, the app background — never the same colour as
     the panel's own contents. */
  background: var(--mold-panel-raised, var(--mold-bg));
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  box-shadow: var(--mold-shadow-md);
  overflow: hidden;
}

.ms-modal__head {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 16px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
}

.ms-modal__title {
  font-size: var(--mold-fs-md);
  font-weight: 600;
  color: var(--mold-text);
}

.ms-modal__desc {
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
}

.ms-modal__steps {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 18px 22px 0;
}

.ms-modal__dot {
  width: 22px;
  height: 3px;
  border-radius: var(--mold-radius-1);
  background: var(--mold-border-control);
  transition: background var(--mold-dur-base) var(--mold-ease-out);
}

.ms-modal__dot[data-on="true"] {
  background: var(--mold-blue);
}

.ms-modal__body {
  padding: 16px;
}

.ms-modal__footer {
  border-top: var(--mold-bw) solid var(--mold-border);
  padding: 14px 16px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}
</style>
