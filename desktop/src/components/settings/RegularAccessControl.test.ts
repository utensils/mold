import { flushPromises, mount } from "@vue/test-utils";
import { expect, it, vi } from "vitest";
import RegularAccessControl from "./RegularAccessControl.vue";
const api = vi.hoisted(() => ({ get: vi.fn(), set: vi.fn() }));
vi.mock("../../lib/ipc", () => ({ ipc: { regularAccessGet: api.get, regularAccessSet: api.set } }));
it("enables a ready Regular listener and publishes its direct pairing address", async () => {
  api.get.mockResolvedValue({ enabled: false, endpoints: [] });
  api.set.mockResolvedValue({ enabled: true, endpoints: ["http://192.168.1.4:42000"] });
  const view = mount(RegularAccessControl);
  await flushPromises();
  await view.get("input").setValue(true);
  await flushPromises();
  expect(api.set).toHaveBeenCalledWith(true);
  expect(view.emitted("address")?.at(-1)).toEqual(["http://192.168.1.4:42000"]);
  expect(view.text()).toContain("192.168.1.4:42000");
});
it("reports a failed enable without claiming that Regular is ready", async () => {
  api.get.mockResolvedValue({ enabled: false, endpoints: [] });
  api.set.mockRejectedValue(new Error("listener did not become ready"));
  const view = mount(RegularAccessControl);
  await flushPromises();
  await view.get("input").setValue(true);
  await flushPromises();
  expect(view.get('[role="alert"]').text()).toContain("did not become ready");
  expect((view.get("input").element as HTMLInputElement).checked).toBe(false);
});
