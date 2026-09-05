/*
 * One register of the overlays currently on screen, newest last.
 *
 * Every overlay that listens for Escape on the DOCUMENT — a ModalPanel, the
 * web shell's focus trap, the desktop Lightbox — hears the same key. Without
 * a shared register they all act: pressing Escape over a confirm that a
 * lightbox opened closed both, and the picture the user was deciding about
 * vanished with the question. The rule is one line: only the TOP overlay
 * handles Escape and Tab, and it says so by calling `stopImmediatePropagation`
 * so nothing below it hears the key at all.
 *
 * The register is deliberately NOT reactive. It is read inside key handlers,
 * at the moment a key arrives, never rendered.
 */
import { onBeforeUnmount, watch, type Ref } from "vue";

/** One overlay's identity in the register. */
export type OverlayToken = symbol;

const stack: OverlayToken[] = [];

/** A token for one overlay instance. The label is for debugging only. */
export function createOverlayToken(label = "overlay"): OverlayToken {
  return Symbol(label);
}

/** Register an overlay as the topmost. Pushing one already registered is a
 *  no-op, so a re-entrant open never doubles it or reorders the stack. */
export function pushOverlay(token: OverlayToken): void {
  if (stack.includes(token)) return;
  stack.push(token);
}

/** Remove an overlay wherever it sits; unregistered tokens are ignored. */
export function popOverlay(token: OverlayToken): void {
  const index = stack.lastIndexOf(token);
  if (index >= 0) stack.splice(index, 1);
}

/** True while this overlay is the one the user is actually looking at. */
export function isTopOverlay(token: OverlayToken): boolean {
  return stack.length > 0 && stack[stack.length - 1] === token;
}

/** How many overlays are registered — the tests' fence. */
export function overlayDepth(): number {
  return stack.length;
}

/** Drain the register between tests. Never called by app code. */
export function resetOverlayStackForTests(): void {
  stack.length = 0;
}

/**
 * Register this component's overlay for as long as `open` is true, and
 * release it when it closes OR when the component unmounts while still open
 * (a route change over an open dialog is exactly that).
 */
export function useOverlayStack(open: Ref<boolean>, label?: string) {
  const token = createOverlayToken(label);

  watch(
    open,
    (isOpen) => {
      if (isOpen) pushOverlay(token);
      else popOverlay(token);
    },
    { immediate: true },
  );
  onBeforeUnmount(() => popOverlay(token));

  return { token, isTop: () => isTopOverlay(token) };
}
