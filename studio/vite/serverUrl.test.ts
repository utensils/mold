import { describe, expect, it } from "vitest";
import { formatDevServerUrl } from "./serverUrl";

describe("formatDevServerUrl", () => {
  it("prints a usable localhost URL for wildcard-bound dev servers", () => {
    expect(formatDevServerUrl("Mold web", "0.0.0.0", 5174)).toBe(
      "Mold web: http://localhost:5174/",
    );
  });

  it("keeps an explicit host and pathname", () => {
    expect(formatDevServerUrl("Mold mobile", "127.0.0.1", 1431, "/create")).toBe(
      "Mold mobile: http://127.0.0.1:1431/create",
    );
  });
});
