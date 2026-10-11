import { nextTick, onScopeDispose, ref } from "vue";

/** Owns mobile keyboard geometry, focus repair, and listener/timer lifetime.
 * The platform bridge is injected; this controller never owns composer state. */
export function useMobileKeyboardViewport(options: {
  android: boolean;
  restoreNativeViewport: () => void;
}) {
  let stopped = false;
  let started = false;
  let keyboardViewportRestoreTimer: ReturnType<typeof setTimeout> | null = null;
  const KEYBOARD_VIEWPORT_SETTLE_MS = 400;
  const NON_KEYBOARD_INPUT_TYPES = new Set([
    "button",
    "checkbox",
    "color",
    "file",
    "hidden",
    "image",
    "radio",
    "range",
    "reset",
    "submit",
  ]);
  function usesSoftwareKeyboard(target: EventTarget | null): target is HTMLElement {
    if (!(target instanceof HTMLElement)) return false;
    if (target.matches("textarea, select, [contenteditable='true']")) return true;
    if (!(target instanceof HTMLInputElement)) return false;
    return !NON_KEYBOARD_INPUT_TYPES.has(target.type.toLowerCase());
  }

  const keyboardVisible = ref(false);
  const androidShortLandscape = ref(false);
  let keyboardBaselineWidth = window.innerWidth;
  let keyboardBaselineHeight = window.innerHeight;
  let keyboardRotationOpen = false;
  let lastKeyboardViewportHeight = window.innerHeight;
  function syncVisualViewportOffset(): void {
    const viewport = window.visualViewport;
    // UIKit may resize the WebView itself after the keyboard opens, making
    // innerHeight equal visualViewport.height. Preserve the unobscured height
    // for this orientation instead of mistaking that second resize for Done.
    if (keyboardBaselineWidth !== window.innerWidth) {
      keyboardRotationOpen = keyboardVisible.value && usesSoftwareKeyboard(document.activeElement);
      keyboardBaselineWidth = window.innerWidth;
      keyboardBaselineHeight = window.innerHeight;
    }
    keyboardBaselineHeight = Math.max(
      keyboardBaselineHeight,
      window.innerHeight,
      viewport?.height ?? 0,
    );
    androidShortLandscape.value =
      options.android &&
      keyboardBaselineWidth > keyboardBaselineHeight &&
      keyboardBaselineHeight < 500;
    if (
      !usesSoftwareKeyboard(document.activeElement) ||
      (viewport && viewport.height - lastKeyboardViewportHeight > 120)
    )
      keyboardRotationOpen = false;
    keyboardVisible.value =
      !!viewport && (keyboardRotationOpen || keyboardBaselineHeight - viewport.height > 120);
    lastKeyboardViewportHeight = viewport?.height ?? window.innerHeight;
    const pageTop = viewport?.pageTop ?? window.scrollY;
    document.documentElement.style.setProperty(
      "--mobile-visual-viewport-page-top",
      `${Math.max(0, pageTop)}px`,
    );
    if (viewport && Number.isFinite(viewport.height) && viewport.height > 0) {
      document.documentElement.style.setProperty(
        "--mobile-visual-viewport-height",
        `${viewport.height}px`,
      );
    } else {
      document.documentElement.style.removeProperty("--mobile-visual-viewport-height");
    }
  }

  function restoreNativeViewport(): void {
    if (stopped) return;
    options.restoreNativeViewport();
    syncVisualViewportOffset();
  }

  function revealFocusedEditor(editor: HTMLElement): void {
    // Wait for viewport sizing and keyboard-only chrome to finish updating.
    // Every editor needs this: WebKit's first pan can leave a low numeric,
    // password, or sheet field underneath the resized content edge.
    void nextTick(() => {
      if (stopped || document.activeElement !== editor) return;
      editor.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    });
  }

  function handleKeyboardViewportResize(): void {
    const widthChanged = keyboardBaselineWidth !== window.innerWidth;
    syncVisualViewportOffset();
    if (widthChanged) {
      cancelKeyboardViewportRestore();
      keyboardViewportRestoreTimer = setTimeout(() => {
        keyboardViewportRestoreTimer = null;
        restoreNativeViewport();
        const focused = document.activeElement;
        if (usesSoftwareKeyboard(focused)) revealFocusedEditor(focused);
      }, 400);
    }
    const editor = document.activeElement;
    if (usesSoftwareKeyboard(editor)) revealFocusedEditor(editor);
  }

  function cancelKeyboardViewportRestore(): void {
    if (!keyboardViewportRestoreTimer) return;
    clearTimeout(keyboardViewportRestoreTimer);
    keyboardViewportRestoreTimer = null;
  }

  function handleKeyboardFocusIn(event: FocusEvent): void {
    if (!usesSoftwareKeyboard(event.target)) return;
    const editor = event.target;
    cancelKeyboardViewportRestore();
    // iOS pans WKWebView after focus to reveal an editor. Re-anchor after the
    // focus turn and once more after the keyboard animation; the Create body's
    // own overflow remains the only scrollable app layer.
    queueMicrotask(() => {
      if (stopped || document.activeElement !== editor) return;
      restoreNativeViewport();
      revealFocusedEditor(editor);
      keyboardViewportRestoreTimer = setTimeout(() => {
        keyboardViewportRestoreTimer = null;
        if (document.activeElement !== editor) return;
        restoreNativeViewport();
        revealFocusedEditor(editor);
      }, KEYBOARD_VIEWPORT_SETTLE_MS);
    });
  }

  /**
   * WKWebView can keep the keyboard-reduced presentation frame after an editor
   * blurs. Repair once after focus has moved, then again after UIKit's keyboard
   * dismissal animation so its final layout pass cannot reinstate the gap.
   */
  function handleKeyboardFocusOut(event: FocusEvent): void {
    if (!usesSoftwareKeyboard(event.target)) return;
    queueMicrotask(() => {
      if (stopped || usesSoftwareKeyboard(document.activeElement)) return;
      restoreNativeViewport();
      cancelKeyboardViewportRestore();
      keyboardViewportRestoreTimer = setTimeout(() => {
        keyboardViewportRestoreTimer = null;
        if (usesSoftwareKeyboard(document.activeElement)) return;
        restoreNativeViewport();
      }, KEYBOARD_VIEWPORT_SETTLE_MS);
    });
  }

  function start() {
    if (started || stopped) return;
    started = true;
    document.addEventListener("focusin", handleKeyboardFocusIn, true);
    document.addEventListener("focusout", handleKeyboardFocusOut, true);
    window.addEventListener("scroll", syncVisualViewportOffset, true);
    window.addEventListener("resize", handleKeyboardViewportResize);
    window.visualViewport?.addEventListener("resize", handleKeyboardViewportResize);
    window.visualViewport?.addEventListener("scroll", syncVisualViewportOffset);
    syncVisualViewportOffset();
  }
  onScopeDispose(() => {
    stopped = true;
    document.removeEventListener("focusin", handleKeyboardFocusIn, true);
    document.removeEventListener("focusout", handleKeyboardFocusOut, true);
    window.removeEventListener("scroll", syncVisualViewportOffset, true);
    window.removeEventListener("resize", handleKeyboardViewportResize);
    window.visualViewport?.removeEventListener("resize", handleKeyboardViewportResize);
    window.visualViewport?.removeEventListener("scroll", syncVisualViewportOffset);
    cancelKeyboardViewportRestore();
    document.documentElement.style.removeProperty("--mobile-visual-viewport-page-top");
    document.documentElement.style.removeProperty("--mobile-visual-viewport-height");
  });
  return { keyboardVisible, androidShortLandscape, start };
}
