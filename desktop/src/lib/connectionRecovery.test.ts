import { describe, expect, it } from "vitest";
import { shouldRestartEmbeddedEngine } from "./connectionRecovery";

describe("shouldRestartEmbeddedEngine", () => {
  it("restarts only a failed embedded engine after consecutive failures", () => {
    expect(shouldRestartEmbeddedEngine("local", 1)).toBe(false);
    expect(shouldRestartEmbeddedEngine("local", 2)).toBe(true);
    expect(shouldRestartEmbeddedEngine("external", 2)).toBe(false);
    expect(shouldRestartEmbeddedEngine("remote", 3)).toBe(false);
    expect(shouldRestartEmbeddedEngine("off", 3)).toBe(false);
  });
});
