import { beforeEach, vi } from "vitest";

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

/*
 * No test reaches the network. Several suites drive host URLs that do not
 * exist as fixtures ("studio", "render.tailnet.ts.net") while the code under
 * test sometimes calls `fetch` directly rather than through a mocked API
 * module — `MeshViewer` is the one component that loads its media that way.
 * Those tests were therefore deciding their result on DNS: a slow lookup left
 * a load pending long enough to assert against, and on a network that answers
 * NXDOMAIN with a landing server the reply is not HTTP at all, which surfaces
 * as `HPE_INVALID_CONSTANT` from Node's client. Both also feed the 5s timeouts
 * that make a full run flake under load. A test that wants a response stubs
 * `fetch` itself; this only makes a MISSING stub instant and local instead of
 * slow and dependent on which network the machine is on.
 */
beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.reject(new TypeError("fetch is not stubbed: tests must not use the network")),
    ),
  );
});
