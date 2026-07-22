/*
 * Overlay focus contract (spec §05) — the half the @ui panels do not cover.
 *
 * `ModalPanel` / `SheetPanel` / `DrawerPanel` already give an overlay its
 * `role="dialog"` + `aria-modal` + accessible name, close on Escape and on
 * backdrop click, and move focus onto the dialog root when it opens. What they
 * deliberately do NOT do — because it needs the host component's DOM and the
 * element that opened the overlay — is:
 *
 *   1. trap Tab / Shift+Tab inside the panel while it is open, and
 *   2. return focus to the opener when it closes,
 *   3. lock background scrolling, and
 *   4. keep Escape working if focus is moved outside the panel.
 *
 * This composable supplies both, so an overlay only has to bind `onKeydown`
 * to the element that wraps its panel. Keep it here rather than in `ui/`: the
 * kit is shared with the desktop and iPhone shells, which have their own
 * focus-restoration rules.
 */
import { onBeforeUnmount, watch, type Ref } from "vue";

let bodyScrollLocks = 0;
let previousBodyOverflow = "";
const overlayFocusStack: symbol[] = [];

function lockBodyScroll() {
  if (typeof document === "undefined") return;
  if (bodyScrollLocks === 0) {
    previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
  }
  bodyScrollLocks += 1;
}

function unlockBodyScroll() {
  if (typeof document === "undefined" || bodyScrollLocks === 0) return;
  bodyScrollLocks -= 1;
  if (bodyScrollLocks === 0) {
    document.body.style.overflow = previousBodyOverflow;
    previousBodyOverflow = "";
  }
}

/**
 * Tabbable descendants. `[tabindex="-1"]` is excluded on every branch so the
 * dialog root itself and roving-tabindex members (SegmentedControl's inactive
 * segments, visually-hidden file inputs) never become trap boundaries.
 */
const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button",
  "input",
  "select",
  "textarea",
  "summary",
  "[tabindex]",
]
  .map((base) => `${base}:not([disabled]):not([tabindex="-1"]):not([hidden])`)
  .join(",");

export function focusableWithin(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
}

type OverlayContainer = HTMLElement | { $el?: unknown } | null;

function rootElement(value: OverlayContainer): HTMLElement | null {
  if (value instanceof HTMLElement) return value;
  return value?.$el instanceof HTMLElement ? value.$el : null;
}

/**
 * @param open       reactive open flag for the overlay
 * @param container  the element wrapping the panel; the trap's boundary
 */
export function useOverlayFocus(
  open: Ref<boolean>,
  container: Ref<OverlayContainer>,
  close?: () => void,
) {
  let opener: HTMLElement | null = null;
  let engaged = false;
  const stackToken = Symbol("overlay-focus");

  function restore() {
    const target = opener;
    opener = null;
    // A re-rendered opener is gone from the document; focusing it would throw
    // focus to <body> anyway, so skip rather than guess at a replacement.
    if (target && target.isConnected) target.focus();
  }

  /** Bind to the overlay host: `@keydown="onKeydown"`. */
  function onKeydown(event: KeyboardEvent) {
    if (event.defaultPrevented || event.key !== "Tab") return;
    const root = rootElement(container.value);
    if (!root) return;

    const items = focusableWithin(root);
    if (items.length === 0) {
      // Nothing to move to — keep focus from escaping to the page behind.
      event.preventDefault();
      return;
    }

    const first = items[0] as HTMLElement;
    const last = items[items.length - 1] as HTMLElement;
    const active = document.activeElement;
    const index = active instanceof HTMLElement ? items.indexOf(active) : -1;

    // Focus is on the dialog root (tabindex="-1") or somewhere outside the
    // panel: pull it back to the near edge in the direction of travel.
    if (index === -1) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
      return;
    }
    if (event.shiftKey && active === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function onDocumentKeydown(event: KeyboardEvent) {
    if (
      !open.value ||
      overlayFocusStack[overlayFocusStack.length - 1] !== stackToken
    )
      return;
    if (event.key === "Escape" && close) {
      event.preventDefault();
      event.stopImmediatePropagation();
      close();
      return;
    }
    const root = rootElement(container.value);
    if (event.key === "Tab" && root && !root.contains(event.target as Node)) {
      onKeydown(event);
    }
  }

  function engage() {
    if (engaged || typeof document === "undefined") return;
    engaged = true;
    lockBodyScroll();
    overlayFocusStack.push(stackToken);
    document.addEventListener("keydown", onDocumentKeydown, true);
  }

  function disengage() {
    if (!engaged || typeof document === "undefined") return;
    engaged = false;
    document.removeEventListener("keydown", onDocumentKeydown, true);
    const index = overlayFocusStack.lastIndexOf(stackToken);
    if (index >= 0) overlayFocusStack.splice(index, 1);
    unlockBodyScroll();
  }

  watch(
    open,
    (isOpen, wasOpen) => {
      if (isOpen && !wasOpen) {
        const active = document.activeElement;
        opener = active instanceof HTMLElement ? active : null;
        engage();
      } else if (!isOpen && wasOpen) {
        disengage();
        restore();
      }
    },
    { immediate: true },
  );

  onBeforeUnmount(() => {
    disengage();
    if (open.value) restore();
  });

  return { onKeydown };
}
