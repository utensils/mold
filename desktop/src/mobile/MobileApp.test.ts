import { flushPromises, mount, type DOMWrapper, type VueWrapper } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import type { GalleryImage, ModelEntry, ServerStatus } from "../lib/api/types";

const { invoke, apiFetchTo, apiJsonTo, sseStream, streamableMediaUrl, evictMedia } = vi.hoisted(
  () => ({
    invoke: vi.fn(),
    apiFetchTo: vi.fn(),
    apiJsonTo: vi.fn(),
    sseStream: vi.fn(),
    streamableMediaUrl: vi.fn(),
    evictMedia: vi.fn(),
  }),
);

vi.mock("@tauri-apps/api/core", () => ({ invoke }));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiFetchTo,
  apiJsonTo,
}));
vi.mock("../lib/api/sse", () => ({ sseStream }));
vi.mock("../lib/gallery/media", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/gallery/media")>()),
  streamableMediaUrl,
  evictMedia,
}));

import MobileApp from "./MobileApp.vue";

installMemoryLocalStorage();

const target = { baseUrl: "http://studio.tailnet.ts.net:7680", apiKey: "secret" };
const status: ServerStatus = {
  version: "0.18.0",
  models_loaded: [],
  uptime_secs: 60,
  hostname: "studio",
  instance_id: "studio-id",
};
const model: ModelEntry = {
  name: "ltx2:q8",
  family: "ltx2",
  size_gb: 20,
  is_loaded: false,
  hf_repo: "example/ltx2",
  default_steps: 30,
  default_guidance: 3,
  default_width: 768,
  default_height: 512,
  description: "Video model",
  downloaded: true,
};
const print: GalleryImage = {
  filename: "storm clip.mp4",
  timestamp: 1_700_000_000,
  format: "mp4",
  metadata: {
    prompt: "a ship crossing violet lightning",
    negative_prompt: "calm water",
    model: model.name,
    seed: 77,
    steps: 28,
    guidance: 4.25,
    width: 1536,
    height: 1024,
    generation_width: 768,
    generation_height: 512,
    output_format: "mp4",
    scheduler: "ddim",
    frames: 121,
    fps: 30,
  },
};

let wrapper: VueWrapper | null = null;
let objectUrlSequence = 0;

function fieldControl(label: string): DOMWrapper<Element> {
  const field = wrapper
    ?.findAll("label.field")
    .find((candidate) => candidate.find("span").text() === label);
  if (!field) throw new Error(`Missing ${label} field`);
  return field.find("input, textarea, select");
}

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem(
    "mold.mobile.hosts.v1",
    JSON.stringify([
      {
        id: "studio-id",
        name: "Studio",
        baseUrl: target.baseUrl,
        hostname: "studio",
        version: "0.18.0",
        online: false,
      },
    ]),
  );
  invoke
    .mockReset()
    .mockImplementation((command: string) =>
      Promise.resolve(command === "keychain_get_api_key" ? target.apiKey : null),
    );
  apiJsonTo.mockReset().mockImplementation((_target: unknown, path: string) => {
    if (path === "/api/status") return Promise.resolve(status);
    if (path === "/api/models") return Promise.resolve([model]);
    if (path === "/api/gallery") return Promise.resolve([print]);
    return Promise.reject(new Error(`Unexpected API path: ${path}`));
  });
  apiFetchTo.mockReset().mockResolvedValue({
    blob: () => Promise.resolve(new Blob(["thumbnail"])),
  } as Response);
  sseStream.mockReset();
  streamableMediaUrl.mockReset().mockResolvedValue("https://studio/media/full-video");
  evictMedia.mockReset();
  objectUrlSequence = 0;
  URL.createObjectURL = vi.fn(() => `blob:thumbnail-${++objectUrlSequence}`);
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  document.body.innerHTML = "";
});

describe("MobileApp gallery", () => {
  it("opens media first, then explicitly reuses the prompt and visible settings", async () => {
    wrapper = mount(MobileApp, { attachTo: document.body });
    await flushPromises();

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));

    const tile = wrapper.get("[data-test='gallery-item']");
    expect(tile.attributes("aria-label")).toBe("Open storm clip.mp4 from Studio");
    await tile.trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-selected")).toBe(
      "true",
    );
    expect(wrapper.find("[data-test='gallery-viewer-video']").exists()).toBe(true);
    expect(streamableMediaUrl).toHaveBeenCalledWith("/api/gallery/image/storm%20clip.mp4", {
      target,
      cacheKey: "studio-id",
      allowLegacyBlob: false,
    });

    await wrapper.get("[data-test='gallery-viewer-close']").trigger("click");
    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-selected")).toBe(
      "true",
    );

    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-tab-generate']").attributes("aria-selected")).toBe(
      "true",
    );
    expect(wrapper.get("#mobile-prompt").element).toHaveProperty("value", print.metadata.prompt);
    expect(fieldControl("Negative prompt").element).toHaveProperty("value", "calm water");
    expect(fieldControl("Width").element).toHaveProperty("value", "768");
    expect(fieldControl("Height").element).toHaveProperty("value", "512");
    expect(fieldControl("Format").element).toHaveProperty("value", "mp4");
    expect(fieldControl("Frames").element).toHaveProperty("value", "121");
    expect(fieldControl("FPS").element).toHaveProperty("value", "30");
  });

  it("keeps the viewer open when the print host's models cannot be loaded", async () => {
    const remoteTarget = { baseUrl: "http://remote.tailnet.ts.net:7680", apiKey: "secret" };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          version: "0.18.0",
          online: false,
        },
        {
          id: "remote-id",
          name: "Remote",
          baseUrl: remoteTarget.baseUrl,
          hostname: "remote",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/gallery") {
        return Promise.resolve(baseUrl === remoteTarget.baseUrl ? [print] : []);
      }
      if (baseUrl === remoteTarget.baseUrl) return Promise.reject(new Error("offline"));
      if (path === "/api/status") return Promise.resolve(status);
      if (path === "/api/models") return Promise.resolve([model]);
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mount(MobileApp, { attachTo: document.body });
    await flushPromises();
    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='gallery-viewer']").exists()).toBe(true);
    expect(wrapper.get("[role='alert']").text()).toContain("Couldn’t load models from Remote");
    expect(wrapper.get("[data-test='mobile-tab-gallery']").attributes("aria-selected")).toBe(
      "true",
    );
  });

  it("reloads model ownership after removing the active host before reuse", async () => {
    const remoteTarget = { baseUrl: "http://remote.tailnet.ts.net:7680", apiKey: "secret" };
    const studioModel: ModelEntry = {
      ...model,
      name: "flux:studio-only",
      family: "flux",
      default_width: 1024,
      default_height: 1024,
    };
    localStorage.setItem(
      "mold.mobile.hosts.v1",
      JSON.stringify([
        {
          id: "studio-id",
          name: "Studio",
          baseUrl: target.baseUrl,
          hostname: "studio",
          version: "0.18.0",
          online: false,
        },
        {
          id: "remote-id",
          name: "Remote",
          baseUrl: remoteTarget.baseUrl,
          hostname: "remote",
          version: "0.18.0",
          online: false,
        },
      ]),
    );
    apiJsonTo.mockImplementation((requestTarget: unknown, path: string) => {
      const baseUrl = (requestTarget as { baseUrl: string }).baseUrl;
      if (path === "/api/status") {
        return Promise.resolve({
          ...status,
          hostname: baseUrl === remoteTarget.baseUrl ? "remote" : "studio",
        });
      }
      if (path === "/api/models") {
        return Promise.resolve(baseUrl === remoteTarget.baseUrl ? [model] : [studioModel]);
      }
      if (path === "/api/gallery") {
        return Promise.resolve(baseUrl === remoteTarget.baseUrl ? [print] : []);
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mount(MobileApp, { attachTo: document.body });
    await vi.waitFor(() =>
      expect(fieldControl("Model").element).toHaveProperty("value", studioModel.name),
    );

    const hostsTab = wrapper
      .findAll("button.mobile-tab")
      .find((button) => button.text() === "Hosts");
    if (!hostsTab) throw new Error("Missing Hosts tab");
    await hostsTab.trigger("click");
    const studioRow = wrapper
      .findAll(".host-row")
      .find((row) => row.find(".host-name").text() === "Studio");
    if (!studioRow) throw new Error("Missing Studio host row");
    await studioRow.get("button.danger-button").trigger("click");
    await vi.waitFor(() => expect(apiJsonTo).toHaveBeenCalledWith(remoteTarget, "/api/models"));

    await wrapper.get("[data-test='mobile-tab-gallery']").trigger("click");
    await vi.waitFor(() => expect(wrapper?.find("[data-test='gallery-item']").exists()).toBe(true));
    await wrapper.get("[data-test='gallery-item']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='gallery-viewer-reuse']").trigger("click");
    await flushPromises();

    expect(fieldControl("Model").element).toHaveProperty("value", model.name);
    expect(wrapper.get(".status-line").text()).toBe("Prompt settings restored");
    const developButton = wrapper
      .findAll("button")
      .find((button) => button.text() === "Develop print");
    expect(developButton?.attributes("disabled")).toBeUndefined();
  });
});
