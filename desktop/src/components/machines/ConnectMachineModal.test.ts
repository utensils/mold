import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import fixtures from "@studio/api/pairing.fixtures.json";
const mocks = vi.hoisted(() => ({ connect: vi.fn(), claim: vi.fn() }));
vi.mock("../../stores/hosts", () => ({ useHostsStore: () => ({ connect: mocks.connect }) }));
vi.mock("@studio/api/pairing", async (original) => ({
  ...(await original<object>()),
  claimPairingSession: mocks.claim,
}));
import ConnectMachineModal from "./ConnectMachineModal.vue";
const payload = { type: "mold.mobile-pairing", version: 2, ...fixtures.links[0]!.payload };
enableAutoUnmount(afterEach);
beforeEach(() => {
  vi.clearAllMocks();
  mocks.claim.mockResolvedValue({ instance_id: payload.instance_id, api_key: null });
  mocks.connect.mockResolvedValue(undefined);
});
it("connects a newly verified profile without passing a bearer through general host settings", async () => {
  const view = mount(ConnectMachineModal, { props: { open: true } });
  await view.get("textarea").setValue(JSON.stringify(payload));
  await view.get("form").trigger("submit");
  await flushPromises();
  expect(mocks.connect).toHaveBeenCalledWith(payload.base_url, null, payload.name, "secure");
  expect(view.emitted("connected")).toHaveLength(1);
  expect(view.emitted("close")).toHaveLength(1);
});
it("explains identity verification for a discovered nearby host", () => {
  const view = mount(ConnectMachineModal, {
    props: { open: true, initialHost: { name: "Halcyon" } as never },
  });
  expect(view.text()).toContain("Halcyon is nearby");
  expect(view.text()).toContain("verify its identity");
  expect(view.find('input[type="password"]').exists()).toBe(false);
  expect(mocks.connect).not.toHaveBeenCalled();
});
it("shows a connection error after pairing without claiming success", async () => {
  mocks.connect.mockRejectedValue(new Error("Machine is offline. Reconnect from Machines."));
  const view = mount(ConnectMachineModal, { props: { open: true } });
  await view.get("textarea").setValue(JSON.stringify(payload));
  await view.get("form").trigger("submit");
  await flushPromises();
  expect(view.get('[role="alert"]').text()).toContain("offline");
  expect(view.emitted("connected")).toBeUndefined();
});
