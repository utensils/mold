import { afterEach, expect, it, vi } from "vitest";
import { setOriginApiKey } from "./originAuth";
import { observeLibraryCopyAuthority } from "./libraryCopyAuthority";
const cleanups: (() => void)[] = [];
afterEach(() => {
  cleanups.splice(0).forEach((stop) => stop());
  sessionStorage.clear();
});
it("invalidates on an origin credential change even when the key is changed back", () => {
  setOriginApiKey("original");
  const invalidate = vi.fn();
  cleanups.push(observeLibraryCopyAuthority(invalidate));
  setOriginApiKey("replacement");
  setOriginApiKey("original");
  expect(invalidate).toHaveBeenCalledTimes(2);
});
it("invalidates primary switches and replaced secure authority without recording credentials", () => {
  const invalidate = vi.fn();
  cleanups.push(observeLibraryCopyAuthority(invalidate));
  window.dispatchEvent(new CustomEvent("mold:studio-primary-changed"));
  window.dispatchEvent(
    new CustomEvent("mold:secure-pairings-changed", {
      detail: { namespace: "fixture", authority: "replaced" },
    }),
  );
  expect(invalidate).toHaveBeenCalledTimes(2);
  window.dispatchEvent(
    new CustomEvent("mold:secure-pairings-changed", {
      detail: { namespace: "fixture", authority: "unchanged" },
    }),
  );
  expect(invalidate).toHaveBeenCalledTimes(2);
});
it("stops listening when the copy finishes", () => {
  const invalidate = vi.fn();
  observeLibraryCopyAuthority(invalidate)();
  setOriginApiKey("new");
  window.dispatchEvent(new CustomEvent("mold:hosts-changed"));
  expect(invalidate).not.toHaveBeenCalled();
});
