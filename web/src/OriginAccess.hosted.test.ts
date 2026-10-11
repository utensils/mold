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
  reload: vi.fn(),
}));
vi.mock("./lib/secureStudio", async (original) => ({
  ...(await original<typeof import("./lib/secureStudio")>()),
  reloadStudioDocument: mocks.reload,
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
it("switches through a fresh gate without forgetting either pairing", async () => {
  const { second } = hosts();
  const wrapper = mount(OriginAccess);
  await flushPromises();
  await wrapper.get('[data-test="studio"] button').trigger("click");
  expect(studioPrimary()).toBeNull();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(wrapper.findAll("option")).toHaveLength(2);
  expect(mocks.reload).toHaveBeenCalledTimes(1);
  wrapper.unmount();
  const nextDocument = mount(OriginAccess);
  await flushPromises();
  expect(nextDocument.find('[data-test="studio"]').exists()).toBe(false);
  await nextDocument.get("select").setValue(second.id);
  await nextDocument.get("form").trigger("submit");
  await flushPromises();
  expect(studioPrimary()?.id).toBe(second.id);
  expect(nextDocument.find('[data-test="studio"]').exists()).toBe(true);
  expect(mocks.reload).toHaveBeenCalledTimes(1);
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
  expect(mocks.reload).toHaveBeenCalledTimes(1);
  wrapper.unmount();
  const nextDocument = mount(OriginAccess);
  await flushPromises();
  await nextDocument.get("form").trigger("submit");
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
  expect(mocks.reload).toHaveBeenCalledTimes(1);
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
  expect(mocks.reload).toHaveBeenCalledTimes(1);
  wrapper.unmount();
  const nextDocument = mount(OriginAccess);
  await flushPromises();
  mocks.parse.mockReturnValue({ base_url: "https://new.invalid", name: "New" });
  mocks.claim.mockResolvedValue({ instance_id: "new-id" });
  await nextDocument.get("textarea").setValue("fresh invitation");
  await nextDocument.get("form").trigger("submit");
  await flushPromises();
  expect(studioPrimary()?.url).toBe("https://new.invalid");
  expect(nextDocument.find('[data-test="studio"]').exists()).toBe(true);
  expect(nextDocument.find("textarea").exists()).toBe(false);
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

it("requires a fresh document after switching so old model and download actions cannot mount on B", async () => {
  const { second } = hosts();
  const wrapper = mount(OriginAccess);
  await flushPromises();
  expect(mocks.reload).not.toHaveBeenCalled();
  await wrapper.get('[data-test="studio"] button').trigger("click");
  expect(mocks.reload).toHaveBeenCalledTimes(1);
  expect(studioPrimary()).toBeNull();
  const before = mocks.fetch.mock.calls.length;
  await wrapper.get("select").setValue(second.id);
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(studioPrimary()).toBeNull();
  expect(mocks.fetch).toHaveBeenCalledTimes(before);
  window.dispatchEvent(
    new StorageEvent("storage", { key: "mold.web.primary.v3" }),
  );
  expect(mocks.reload).toHaveBeenCalledTimes(1);
});

it("reloads only after a selected Forget succeeds", async () => {
  hosts();
  const wrapper = mount(OriginAccess);
  await flushPromises();
  mocks.forget.mockRejectedValueOnce(new Error("storage refused"));
  await expect(removeHost("origin")).rejects.toThrow("storage refused");
  expect(mocks.reload).not.toHaveBeenCalled();
  await removeHost("origin");
  await flushPromises();
  expect(mocks.reload).toHaveBeenCalledTimes(1);
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
});

it("keeps the fresh hosted document at the gate when only remembered machines exist", async () => {
  hosts();
  setStudioPrimary(null);
  const wrapper = mount(OriginAccess);
  await flushPromises();
  expect(wrapper.find('[data-test="studio"]').exists()).toBe(false);
  expect(mocks.fetch).not.toHaveBeenCalled();
  expect(mocks.reload).not.toHaveBeenCalled();
  expect(wrapper.findAll("option")).toHaveLength(2);
});
