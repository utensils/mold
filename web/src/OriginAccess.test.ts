import { mount, flushPromises } from "@vue/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import OriginAccess from "./OriginAccess.vue";
import { originApiKey } from "./lib/originAuth";
vi.mock("./App.vue", () => ({
  default: { template: '<div data-test="studio">Studio</div>' },
}));
beforeEach(() => sessionStorage.clear());
afterEach(() => vi.unstubAllGlobals());
it("mounts Studio only after a validated key and never saves a refused key", async () => {
  const fetchMock = vi
    .fn()
    .mockResolvedValueOnce(new Response("", { status: 401 }))
    .mockResolvedValueOnce(new Response("", { status: 401 }))
    .mockResolvedValueOnce(new Response("{}", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const wrapper = mount(OriginAccess);
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  await wrapper.get("input").setValue("wrong");
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(originApiKey()).toBeNull();
  await wrapper.get("input").setValue("right");
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(originApiKey()).toBe("right");
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(true);
  expect(new Headers(fetchMock.mock.calls[2][1].headers).get("x-api-key")).toBe(
    "right",
  );
  await wrapper.get("button").trigger("click");
  expect(originApiKey()).toBeNull();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  wrapper.unmount();
});
it("opens an auth-disabled local machine without asking for a key", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}")));
  const wrapper = mount(OriginAccess);
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(true);
  wrapper.unmount();
});
