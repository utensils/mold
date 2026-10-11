import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import ConnectModal from "./ConnectModal.vue";
import { listStoredHosts } from "../../lib/hostRegistry";
import fixtures from "@studio/api/pairing.fixtures.json";
const claim = vi.hoisted(() => vi.fn());
vi.mock("@studio/api/pairing", async (original) => ({
  ...(await original<object>()),
  claimPairingSession: claim,
}));
const payload = {
  type: "mold.mobile-pairing",
  version: 2,
  ...fixtures.links[0]!.payload,
};
enableAutoUnmount(afterEach);
beforeEach(() => {
  localStorage.clear();
  claim.mockReset();
});
it("stores the verified machine without copying its transport credential into the registry", async () => {
  claim.mockResolvedValue({ instance_id: payload.instance_id, api_key: null });
  const view = mount(ConnectModal, { props: { open: true } });
  await view.get("textarea").setValue(JSON.stringify(payload));
  await view.get("form").trigger("submit");
  await flushPromises();
  expect(claim).toHaveBeenCalledWith(payload, {
    name: "Mold Studio browser",
    kind: "browser",
  });
  expect(listStoredHosts()).toMatchObject([
    {
      url: payload.base_url,
      instanceId: payload.instance_id,
      name: payload.name,
    },
  ]);
  expect(listStoredHosts()[0]?.apiKey).toBeFalsy();
  expect(view.emitted("added")).toHaveLength(1);
  expect(view.emitted("close")).toHaveLength(1);
});
it("offers an actionable error without adding an unverified machine", async () => {
  claim.mockRejectedValue(new Error("Create a fresh code on the host."));
  const view = mount(ConnectModal, { props: { open: true } });
  await view.get("textarea").setValue(JSON.stringify(payload));
  await view.get("form").trigger("submit");
  await flushPromises();
  expect(view.get('[role="alert"]').text()).toContain("fresh code");
  expect(listStoredHosts()).toEqual([]);
  expect(view.emitted("added")).toBeUndefined();
});
it("does not add a machine from a late claim after the dialog closes", async () => {
  let finish!: (value: unknown) => void;
  claim.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const view = mount(ConnectModal, { props: { open: true } });
  await view.get("textarea").setValue(JSON.stringify(payload));
  await view.get("form").trigger("submit");
  await view.setProps({ open: false });
  finish({ instance_id: payload.instance_id, api_key: null });
  await flushPromises();
  expect(listStoredHosts()).toEqual([]);
  expect(view.emitted("added")).toBeUndefined();
});
