import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import OriginAccess from "./OriginAccess.vue";
import {
  addHost,
  listStoredHosts,
  removeHost,
  updateHost,
} from "./lib/hostRegistry";
import { setStudioPrimary, studioPrimary } from "./lib/secureStudio";

const mocks = vi.hoisted(() => ({
  fetch: vi.fn(),
  forget: vi.fn(),
  clear: vi.fn(),
  secure: vi.fn(),
  configure: vi.fn(),
  claim: vi.fn(),
  parse: vi.fn(),
}));
vi.mock("@studio/api/relayTransport", async (original) => ({
  ...(await original<typeof import("@studio/api/relayTransport")>()),
  relayFetch: mocks.fetch,
  clearRelayInfoCache: mocks.clear,
  configureSecureOrigin: mocks.configure,
}));
vi.mock("@studio/api/secureStore", () => ({
  secureHostForOrigin: mocks.secure,
  forgetSecureHostForMachine: mocks.forget,
}));
vi.mock("@studio/api/pairing", () => ({
  parseMobilePairingPayload: mocks.parse,
  claimPairingSession: mocks.claim,
}));
vi.mock("./App.vue", async () => {
  const { inject } = await import("vue");
  const { ORIGIN_ACCESS_CHANGE_KEY } = await import("./lib/originAuth");
  return {
    default: {
      setup: () => ({ change: inject(ORIGIN_ACCESS_CHANGE_KEY) }),
      template:
        '<div data-test="studio"><button @click="change">Switch machine</button></div>',
    },
  };
});
enableAutoUnmount(afterEach);
beforeEach(() => {
  vi.stubEnv("VITE_STANDALONE_STUDIO", "1");
  localStorage.clear();
  sessionStorage.clear();
  setStudioPrimary(null);
  vi.clearAllMocks();
  mocks.fetch.mockResolvedValue(new Response("{}"));
  mocks.forget.mockResolvedValue(undefined);
  mocks.secure.mockResolvedValue({ namespace: "fixture" });
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.useRealTimers();
  vi.restoreAllMocks();
});
function hosts() {
  const first = addHost({
    url: "https://first.invalid",
    name: "First",
    instanceId: "a",
  });
  const second = addHost({
    url: "https://second.invalid",
    name: "Second",
    instanceId: "b",
  });
  setStudioPrimary(first);
  return { first, second };
}
it("switches selected hosts through the gate without forgetting either pairing", async () => {
  const { second } = hosts();
  const wrapper = mount(OriginAccess);
  await flushPromises();
  await wrapper.get('[data-test="studio"] button').trigger("click");
  expect(studioPrimary()).toBeNull();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(wrapper.findAll("option")).toHaveLength(2);
  await wrapper.get("select").setValue(second.id);
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(studioPrimary()?.id).toBe(second.id);
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(true);
  expect(mocks.forget).not.toHaveBeenCalled();
  expect(listStoredHosts()).toHaveLength(2);
});
it("returns to the gate after successful selected Forget and can choose another host", async () => {
  const { second } = hosts();
  const wrapper = mount(OriginAccess);
  await flushPromises();
  await removeHost("origin");
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(studioPrimary()).toBeNull();
  expect(wrapper.get("select").element.value).toBe(second.id);
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(studioPrimary()?.id).toBe(second.id);
});
it("keeps retry and switch available when Forget storage fails", async () => {
  const { first } = hosts();
  const wrapper = mount(OriginAccess);
  await flushPromises();
  mocks.forget.mockRejectedValueOnce(new Error("remove failed"));
  await expect(removeHost("origin")).rejects.toThrow("remove failed");
  expect(studioPrimary()?.id).toBe(first.id);
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(true);
  await wrapper.get('[data-test="studio"] button').trigger("click");
  expect(wrapper.find("form").exists()).toBe(true);
});
it("suspends the current tab before accepting another tab's changed selection", async () => {
  const { first, second } = hosts();
  const wrapper = mount(OriginAccess);
  await flushPromises();
  localStorage.setItem("mold.web.primary.v3", JSON.stringify(second));
  expect(studioPrimary()?.url).toBe(first.url);
  window.dispatchEvent(
    new StorageEvent("storage", { key: "mold.web.primary.v3" }),
  );
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(studioPrimary()).toBeNull();
  expect(mocks.clear).toHaveBeenCalled();
});

it("does not remount or restore a forgotten primary after a late status response", async () => {
  const { first } = hosts();
  let resolve!: (response: Response) => void;
  mocks.fetch.mockImplementationOnce(
    () =>
      new Promise<Response>((done) => {
        resolve = done;
      }),
  );
  const wrapper = mount(OriginAccess);
  await flushPromises();
  await removeHost("origin");
  resolve(new Response("{}"));
  await flushPromises();
  expect(studioPrimary()).toBeNull();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(listStoredHosts().some((host) => host.id === first.id)).toBe(false);
});
it("can pair a fresh invitation after forgetting the last selected machine", async () => {
  const first = addHost({
    url: "https://first.invalid",
    name: "First",
    instanceId: "a",
  });
  setStudioPrimary(first);
  const wrapper = mount(OriginAccess);
  await flushPromises();
  await removeHost("origin");
  await flushPromises();
  mocks.parse.mockReturnValue({ base_url: "https://new.invalid", name: "New" });
  mocks.claim.mockResolvedValue({ instance_id: "new-id" });
  await wrapper.get("textarea").setValue("fresh invitation");
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(studioPrimary()?.url).toBe("https://new.invalid");
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(true);
  expect(wrapper.find("textarea").exists()).toBe(false);
});
it("keeps verified same-machine route learning connected but suspends explicit route edits", async () => {
  const { first } = hosts();
  const wrapper = mount(OriginAccess);
  await flushPromises();
  updateHost(first.id, { url: "http://192.168.1.25:7680" }, true);
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(true);
  expect(studioPrimary()?.url).toBe(first.url);
  updateHost(first.id, { url: "https://replacement.invalid" });
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(studioPrimary()).toBeNull();
});

it("allows pairing to finish after the status timeout interval and starts that timeout at status", async () => {
  vi.useFakeTimers();
  vi.spyOn(AbortSignal, "timeout").mockImplementation((milliseconds) => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), milliseconds);
    return controller.signal;
  });
  let resolve!: (claim: { instance_id: string }) => void;
  mocks.parse.mockReturnValue({ base_url: "https://new.invalid", name: "New" });
  mocks.claim.mockImplementationOnce(
    () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  const wrapper = mount(OriginAccess);
  await flushPromises();
  await wrapper.get("textarea").setValue("fresh invitation");
  await wrapper.get("form").trigger("submit");
  await vi.advanceTimersByTimeAsync(12000);
  resolve({ instance_id: "new-id" });
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(true);
  vi.useRealTimers();
});
it("never selects a late invitation claim after another tab changes selection", async () => {
  let resolve!: (claim: { instance_id: string }) => void;
  mocks.parse.mockReturnValue({ base_url: "https://new.invalid", name: "New" });
  mocks.claim.mockImplementationOnce(
    () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  const wrapper = mount(OriginAccess);
  await flushPromises();
  await wrapper.get("textarea").setValue("fresh invitation");
  await wrapper.get("form").trigger("submit");
  window.dispatchEvent(
    new StorageEvent("storage", { key: "mold.web.primary.v3" }),
  );
  resolve({ instance_id: "new-id" });
  await flushPromises();
  expect(studioPrimary()).toBeNull();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(mocks.fetch).not.toHaveBeenCalled();
});
