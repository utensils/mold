import { expect, it, vi } from "vitest";
import { generationJobTarget } from "./generationJobTarget";
const primary = vi.hoisted(() => ({
  id: "host",
  name: "Host",
  url: "http://192.168.1.4:7680",
  instanceId: "instance",
  connectionMode: "regular" as const,
  apiKey: "regular-key",
}));
vi.mock("./secureStudio", () => ({
  standaloneStudio: () => true,
  studioPrimary: () => primary,
}));
vi.mock("./hostRegistry", () => ({
  ORIGIN_HOST_ID: "origin",
  getHost: () => primary,
}));
it("preserves Secure authority on a legacy frozen origin job after Regular re-pairing", () => {
  expect(
    generationJobTarget({
      hostId: "origin",
      hostInstanceId: "instance",
      target: { baseUrl: primary.url },
    }),
  ).toBeUndefined();
});
it("uses an explicitly frozen Regular origin job on the same instance", () => {
  expect(
    generationJobTarget({
      hostId: "origin",
      hostInstanceId: "instance",
      target: { baseUrl: primary.url, connectionMode: "regular" },
    })?.connectionMode,
  ).toBe("regular");
});
