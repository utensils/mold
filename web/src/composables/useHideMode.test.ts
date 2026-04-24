import { beforeEach, describe, expect, it } from "vitest";
import { __resetHideModeForTests, useHideMode } from "./useHideMode";

beforeEach(() => {
  localStorage.clear();
  __resetHideModeForTests();
});

describe("useHideMode", () => {
  it("starts revealed when storage is empty", () => {
    const h = useHideMode();
    expect(h.hideMode.value).toBe(false);
    expect(h.revealed.value.size).toBe(0);
    expect(h.anyVisible.value).toBe(true);
  });

  it("persists hideMode across singleton rebuilds", () => {
    const h1 = useHideMode();
    h1.toggle();
    expect(h1.hideMode.value).toBe(true);
    expect(localStorage.getItem("mold.gallery.hide")).toBe("true");

    __resetHideModeForTests();
    const h2 = useHideMode();
    expect(h2.hideMode.value).toBe(true);
    expect(h2.revealed.value.size).toBe(0); // revealed is never persisted
  });

  it("toggle from visible hides everything", () => {
    const h = useHideMode();
    h.toggle();
    expect(h.hideMode.value).toBe(true);
    expect(h.anyVisible.value).toBe(false);
  });

  it("toggle from fully-hidden reveals everything", () => {
    const h = useHideMode();
    h.toggle(); // hide
    h.toggle(); // reveal
    expect(h.hideMode.value).toBe(false);
    expect(h.anyVisible.value).toBe(true);
  });

  it("revealOne flips anyVisible without dropping hideMode", () => {
    const h = useHideMode();
    h.toggle(); // hide all
    expect(h.anyVisible.value).toBe(false);
    h.revealOne("a.png");
    expect(h.hideMode.value).toBe(true);
    expect(h.anyVisible.value).toBe(true);
  });

  it("toggle from peeked state re-hides and wipes peeks", () => {
    const h = useHideMode();
    h.toggle(); // hide
    h.revealOne("a.png");
    h.revealOne("b.png");
    expect(h.revealed.value.size).toBe(2);

    h.toggle(); // user clicks "hide everything again"
    expect(h.hideMode.value).toBe(true);
    expect(h.revealed.value.size).toBe(0);
    expect(h.anyVisible.value).toBe(false);
  });

  it("revealOne is idempotent", () => {
    const h = useHideMode();
    h.toggle();
    h.revealOne("a.png");
    const firstRef = h.revealed.value;
    h.revealOne("a.png");
    // No-op re-add should not churn the reactive ref — downstream
    // IntersectionObserver watchers depend on set-identity stability.
    expect(h.revealed.value).toBe(firstRef);
    expect(h.revealed.value.size).toBe(1);
  });

  it("persists revealed state is not written to storage", () => {
    const h = useHideMode();
    h.toggle();
    h.revealOne("a.png");
    expect(localStorage.getItem("mold.gallery.hide")).toBe("true");
    expect(localStorage.getItem("mold.gallery.revealed")).toBeNull();
  });

  it("shares state across multiple useHideMode() calls", () => {
    const h1 = useHideMode();
    const h2 = useHideMode();
    h1.toggle();
    expect(h2.hideMode.value).toBe(true);
    h2.revealOne("x.png");
    expect(h1.revealed.value.has("x.png")).toBe(true);
  });
});
