import { beforeEach, vi } from "vitest";

/*
 * No test reaches the network.
 *
 * Several suites drive host URLs that do not exist, as fixtures ("studio",
 * "render.tailnet.ts.net"), while the code under test calls `fetch` directly
 * rather than through a mocked API module — `api/client.ts` on every request,
 * and `components/MeshViewer.vue` for the one component that loads its own
 * media. Those tests were deciding their results on DNS: a slow lookup left a
 * load pending long enough to assert against, and on a network that answers
 * NXDOMAIN with a landing server the reply is not HTTP at all, which surfaces
 * as `HPE_INVALID_CONSTANT` from Node's client. Every lookup is also a real
 * wait feeding the 5s timeouts that make a full run flake under load.
 *
 * A test that wants a response stubs `fetch` itself; this only makes a MISSING
 * stub instant and local instead of slow and dependent on which network the
 * machine is on. Do not call `vi.unstubAllGlobals()` in a file-level
 * `beforeEach` — setup-file hooks run FIRST, so that restores the real global
 * and undoes this for the whole file; `afterEach` is fine.
 *
 * This lives in `studio/` so its two runners agree: `studio/vitest.config.ts`
 * loads it directly, and `desktop/src/test-setup.ts` imports it for the desktop
 * runner, which also collects `../studio/**` and `../ui/**`.
 */
beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.reject(
        new TypeError("fetch is not stubbed: tests must not use the network"),
      ),
    ),
  );
});
