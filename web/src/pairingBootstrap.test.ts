import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
  history.replaceState(null, "", "/");
});

it("removes the one-time invitation before hash routing captures its initial location", async () => {
  vi.resetModules();
  vi.stubEnv("VITE_STANDALONE_STUDIO", "1");
  history.replaceState(
    { owner: "preserved" },
    "",
    "/mold/studio/#invite_seed=synthetic-one-time-secret&v=2",
  );
  const { router } = await import("./router");
  expect(router.options.history.location).toBe("/create");
  expect(location.hash).toBe("#/create");
  expect(history.state.owner).toBe("preserved");
  const { takeInitialPairingLink } = await import("./lib/initialPairing");
  expect(takeInitialPairingLink()).toBe(
    "https://utensils.io/mold/pair/#invite_seed=synthetic-one-time-secret&v=2",
  );
  expect(takeInitialPairingLink()).toBe("");
  router.options.history.destroy();
}, 30000);

it("captures a Regular handoff once and clears its private token from history", async () => {
  vi.resetModules();
  vi.stubEnv("VITE_STANDALONE_STUDIO", "1");
  history.replaceState(
    null,
    "",
    "/mold/studio/#mode=regular&pairing_token=test-owned-token&version=2",
  );
  const { captureInitialPairingLink, takeInitialPairingLink } =
    await import("./lib/initialPairing");
  captureInitialPairingLink();
  expect(location.hash).toBe("#/create");
  expect(takeInitialPairingLink()).toContain(
    "mode=regular&pairing_token=test-owned-token",
  );
  expect(takeInitialPairingLink()).toBe("");
});
