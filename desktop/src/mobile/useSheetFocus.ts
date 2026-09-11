/*
 * Focus for a bottom sheet — the one implementation.
 *
 * `role="dialog"` with `aria-modal="true"` is a promise: while this thing is
 * open, nothing behind it can be reached. The background is not `inert`, so
 * the promise is kept entirely by this block — take focus on open, keep Tab
 * inside, close on Escape, hand focus back on close.
 *
 * Three sheets carried a copy of it and the two added in #1685 carried none,
 * which meant opening Filters or Add a machine left focus on the button behind
 * the scrim, Tab walked the list underneath, and Escape did nothing.
 *
 * Two rules are easy to get wrong and are therefore stated here once:
 *
 * - **Only the TOP sheet acts.** `isTop()` comes from the shared overlay
 *   register. A sheet opening underneath another must not pull focus out of
 *   it, and Escape over a stacked pair belongs to the one on top — which is
 *   why the handler also stops propagation rather than merely returning.
 * - **The panel itself is a tab stop.** Every sheet opens with the panel
 *   focused, so the very first Tab arrives with `activeElement === panel`. A
 *   trap that only wraps at the last control lets that first Tab escape into
 *   browser chrome behind the sheet.
 */
import { nextTick, onBeforeUnmount, watch, type Ref } from "vue";

/** Everything inside the panel a Tab can land on. */
const FOCUSABLE =
  "button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex='0']";

export interface SheetFocusOptions {
  /** The panel element. It must carry `tabindex="-1"` to be focusable. */
  panel: Ref<HTMLElement | null>;
  /** Whether the sheet is open, read at the moment it is asked. */
  open: () => boolean;
  /** Whether this sheet is the one the user is actually looking at. */
  isTop: () => boolean;
  onClose: () => void;
  /**
   * The control to focus instead of the panel — for a sheet that exists to be
   * edited. Return null (the default) to focus the panel, which is what a
   * read-first sheet wants: the keyboard then waits for an explicit field tap
   * rather than covering the rows the sheet was opened to show.
   */
  firstControl?: () => HTMLElement | null;
  /** Run before focus is handed back, e.g. to reset a drag offset. */
  onBeforeClose?: () => void;
}

export interface SheetFocus {
  /** Bind to the sheet root's `keydown`. */
  onKeydown: (event: KeyboardEvent) => void;
}

export function useSheetFocus(options: SheetFocusOptions): SheetFocus {
  let restoreFocus: HTMLElement | null = null;

  watch(
    options.open,
    async (open) => {
      if (open) {
        restoreFocus = document.activeElement as HTMLElement | null;
        // Wait for the panel to render before trying to focus it.
        await nextTick();
        if (!options.open() || !options.isTop()) return;
        const first = options.firstControl?.() ?? null;
        (first ?? options.panel.value)?.focus?.();
      } else {
        options.onBeforeClose?.();
        restoreFocus?.focus?.();
        restoreFocus = null;
      }
    },
    { immediate: true },
  );

  onBeforeUnmount(() => {
    options.onBeforeClose?.();
    // Never focus anything on the way out: a route change over an open sheet
    // would otherwise yank focus to a trigger that is itself gone.
    restoreFocus = null;
  });

  function onKeydown(event: KeyboardEvent): void {
    if (!options.open() || !options.isTop()) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopImmediatePropagation();
      options.onClose();
      return;
    }
    if (event.key !== "Tab") return;
    const panel = options.panel.value;
    const controls = [...(panel?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? [])].filter(
      (node) => !node.closest("[inert]") && node.getClientRects().length > 0,
    );
    const first = controls[0];
    const last = controls.at(-1);
    const atPanel = document.activeElement === panel;
    if (!first || (event.shiftKey && (document.activeElement === first || atPanel))) {
      event.preventDefault();
      (last ?? panel)?.focus();
    } else if (!event.shiftKey && (document.activeElement === last || atPanel)) {
      event.preventDefault();
      first.focus();
    }
  }

  return { onKeydown };
}
