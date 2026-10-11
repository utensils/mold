import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import SecurePairingForm from "./SecurePairingForm.vue";
import fixtures from "../api/pairing.fixtures.json";
const claim = vi.hoisted(() => vi.fn());
vi.mock("../api/pairing", async (original) => ({
  ...(await original<object>()),
  claimPairingSession: claim,
}));
beforeEach(() => claim.mockReset());
it("pairs with a pinned invitation and clears its secret after success", async () => {
  const payload = {
    type: "mold.mobile-pairing",
    version: 2,
    ...fixtures.links[0]!.payload,
  };
  claim.mockResolvedValue({ instance_id: payload.instance_id, api_key: null });
  const view = mount(SecurePairingForm, {
    props: { clientName: "Browser", clientKind: "browser" },
  });
  await view.get("textarea").setValue(JSON.stringify(payload));
  await view.get("form").trigger("submit");
  await flushPromises();
  expect(claim).toHaveBeenCalledWith(payload, {
    name: "Browser",
    kind: "browser",
  });
  expect(view.emitted("paired")).toHaveLength(1);
  expect((view.get("textarea").element as HTMLTextAreaElement).value).toBe("");
  view.unmount();
});
it("rejects retired links locally and exposes an actionable retry error", async () => {
  const view = mount(SecurePairingForm, {
    props: { clientName: "Desktop", clientKind: "desktop" },
  });
  await view
    .get("textarea")
    .setValue('{"type":"mold.mobile-pairing","version":1,"token":"old"}');
  await view.get("form").trigger("submit");
  await flushPromises();
  expect(claim).not.toHaveBeenCalled();
  expect(view.get('[role="alert"]').text()).toMatch(/pairing|code/i);
  expect(view.emitted("paired")).toBeUndefined();
  view.unmount();
});
