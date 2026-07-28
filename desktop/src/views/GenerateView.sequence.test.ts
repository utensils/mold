import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";

const { routerPush, routerReplace, routeQuery } = vi.hoisted(() => ({
  routerPush: vi.fn(),
  routerReplace: vi.fn(),
  routeQuery: { value: {} as Record<string, unknown> },
}));
vi.mock("vue-router", () => ({
  useRouter: () => ({ push: routerPush, replace: routerReplace }),
  useRoute: () => ({ query: routeQuery.value }),
}));
const apiJson = vi.fn();
const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  ApiError: class ApiError extends Error {
    status = 0;
  },
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
  },
}));
vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn().mockResolvedValue(undefined) }));

import GenerateView from "./GenerateView.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useGenerateFormStore } from "../stores/generateForm";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import type { ModelEntry } from "../lib/api/types";

enableAutoUnmount(afterEach);

const videoModel: ModelEntry = {
  name: "ltx-video",
  family: "ltx-video",
  downloaded: true,
  default_width: 1024,
  default_height: 576,
  default_steps: 25,
  default_guidance: 3,
} as ModelEntry;

function readyLocal() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
}

function mountView() {
  return mount(GenerateView, {
    shallow: true,
    attachTo: document.body,
    global: { stubs: { SequenceComposer: true, ComposerCard: true } },
  });
}

let installedPayload: ModelEntry[] = [];

beforeEach(() => {
  setActivePinia(createPinia());
  routerPush.mockClear();
  routerReplace.mockClear();
  routeQuery.value = {};
  installedPayload = [];
  apiJson.mockReset();
  apiJson.mockImplementation((path: unknown) =>
    Promise.resolve(path === "/api/models" ? installedPayload : []),
  );
  apiJsonTo.mockReset();
  apiJsonTo.mockImplementation((_target: unknown, path: unknown) => {
    if (path === "/api/chain-jobs") return Promise.resolve({ jobs: [] });
    if (path === "/api/models") return Promise.resolve(installedPayload);
    return Promise.resolve({});
  });
  window.localStorage?.clear?.();
});
afterEach(() => (document.body.innerHTML = ""));

describe("GenerateView — sequence output", () => {
  it("consumes ?output=sequence once, then strips the query", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.prompt = "a storm rolls in";
    routeQuery.value = { output: "sequence" };
    mountView();
    await flushPromises();

    const draft = useSequenceDraftStore();
    expect(draft.output).toBe("sequence");
    expect(draft.clips.length).toBeGreaterThanOrEqual(2);
    expect(draft.clips[0]!.prompt).toBe("a storm rolls in");
    expect(routerReplace).toHaveBeenCalledWith({ path: "/create" });
  });

  it("renders the sequence bench instead of the single composer", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    useGenerateFormStore().form.model = "ltx-video";
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(25);
    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.find("sequence-composer-stub").exists()).toBe(true);
    expect(wrapper.find("composer-card-stub").exists()).toBe(false);
  });

  it("keeps the single composer for one-shot output", async () => {
    readyLocal();
    installedPayload = [videoModel];
    useModelStore().all = [videoModel];
    const wrapper = mountView();
    await flushPromises();
    expect(wrapper.find("composer-card-stub").exists()).toBe(true);
    expect(wrapper.find("sequence-composer-stub").exists()).toBe(false);
  });

  it("guides to Discover when no chain-capable video model is installed", async () => {
    readyLocal();
    installedPayload = [];
    apiJsonTo.mockRejectedValue(new Error("offline"));
    useModelStore().all = [];
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.find("[data-test='sequence-empty']").exists()).toBe(true);
    expect(wrapper.find("sequence-composer-stub").exists()).toBe(false);
  });
});
