import { beforeEach } from "vitest";

/*
 * happy-dom keeps one `localStorage` per test FILE, so anything a store
 * persists in one test is what the next test's fresh Pinia hydrates from —
 * the last-used-styles memory turned a "picks the first installed style"
 * expectation into whatever the previous test had selected. Every test
 * starts from empty storage; a test that wants persistence writes it itself.
 */
beforeEach(() => {
  try {
    globalThis.localStorage?.clear();
  } catch {
    // No storage in this environment.
  }
});
