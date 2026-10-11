import { afterEach, beforeEach, expect, it, vi } from "vitest";
const { invoke, listSecureHosts, forgetSecureHost } = vi.hoisted(() => ({
  invoke: vi.fn(),
  listSecureHosts: vi.fn(),
  forgetSecureHost: vi.fn(),
}));
vi.mock("@studio/api/secureStore", () => ({ listSecureHosts, forgetSecureHost }));
import { ipc } from "./ipc";
const host = { id: "machine", url: "http://192.168.1.2:7680", instanceId: "instance" };
beforeEach(() => {
  Object.defineProperty(window, "__TAURI_INTERNALS__", { value: { invoke }, configurable: true });
  invoke
    .mockReset()
    .mockImplementation(async (command) =>
      command === "app_settings_get" ? { savedHosts: [host] } : [],
    );
  listSecureHosts
    .mockReset()
    .mockResolvedValue([{ namespace: "pin", origins: [host.url], instance_id: host.instanceId }]);
  forgetSecureHost.mockReset().mockResolvedValue(undefined);
});
afterEach(() => {
  delete (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
});
it("forgets canonical encrypted pairing before removing the saved host", async () => {
  await ipc.forgetRemoteHost(host.id);
  expect(forgetSecureHost).toHaveBeenCalledExactlyOnceWith("pin");
  expect(invoke).toHaveBeenCalledWith("forget_remote_host", { id: host.id }, undefined);
  expect(forgetSecureHost.mock.invocationCallOrder[0]).toBeLessThan(
    invoke.mock.invocationCallOrder.at(-1)!,
  );
});
it("keeps the saved host retryable when secure credential removal fails", async () => {
  forgetSecureHost.mockRejectedValueOnce(new Error("key store unavailable"));
  await expect(ipc.forgetRemoteHost(host.id)).rejects.toThrow("key store unavailable");
  expect(invoke.mock.calls.some(([command]) => command === "forget_remote_host")).toBe(false);
});
it("forgets the same pinned installation after its LAN address changed", async () => {
  listSecureHosts.mockResolvedValue([
    { namespace: "pin", origins: ["http://192.168.1.9:7680"], instance_id: host.instanceId },
  ]);
  await ipc.forgetRemoteHost(host.id);
  expect(forgetSecureHost).toHaveBeenCalledExactlyOnceWith("pin");
});
