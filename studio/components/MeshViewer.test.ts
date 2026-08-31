import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { triangleGlb } from "../lib/glbFixture";
import MeshViewer from "./MeshViewer.vue";

/*
 * The GPU half of this component is covered by the browser, not here: happy-dom
 * has no WebGL, so every mount in this file lands on the degraded path. That is
 * exactly the path worth pinning — a lightbox must never show a black rectangle,
 * and the poster it falls back to is the only thing a viewer without WebGL,
 * without the file, or with a corrupt file ever sees.
 */

interface FetchCall {
  url: string;
  signal: AbortSignal | undefined;
}

const calls: FetchCall[] = [];

function stubFetch(respond: () => Promise<unknown>): void {
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    calls.push({ url, signal: init?.signal ?? undefined });
    return respond();
  });
}

function ok(body: ArrayBuffer): Promise<unknown> {
  return Promise.resolve({
    ok: true,
    status: 200,
    arrayBuffer: () => Promise.resolve(body),
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  calls.length = 0;
});

describe("MeshViewer", () => {
  it("shows the poster and a status line while the mesh loads", () => {
    stubFetch(() => new Promise(() => {}));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", poster: "/media/mesh.png" },
    });
    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("loading");
    expect(
      wrapper.get("[data-test=mesh-viewer-poster]").attributes("src"),
    ).toBe("/media/mesh.png");
    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toContain(
      "Loading",
    );
    wrapper.unmount();
  });

  it("keeps the poster and says so when the file cannot be read", async () => {
    stubFetch(() => ok(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]).buffer));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", poster: "/media/mesh.png" },
    });
    await flushPromises();

    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("failed");
    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toBe(
      "This mesh file couldn't be read.",
    );
    expect(wrapper.find("[data-test=mesh-viewer-poster]").exists()).toBe(true);
    expect(wrapper.emitted("fail")).toHaveLength(1);
    wrapper.unmount();
  });

  it("falls back to the poster when the browser has no WebGL", async () => {
    // happy-dom hands back null already; pinning it keeps the assertion about
    // the component rather than about the DOM stand-in.
    const getContext = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(null);
    stubFetch(() => ok(triangleGlb()));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", poster: "/media/mesh.png" },
    });
    await flushPromises();

    expect(
      wrapper.get("[data-test=mesh-viewer]").attributes("data-status"),
    ).toBe("failed");
    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toContain(
      "can't display 3-D previews",
    );
    expect(wrapper.find("[data-test=mesh-viewer-poster]").exists()).toBe(true);
    expect(wrapper.emitted("fail")).toHaveLength(1);
    getContext.mockRestore();
    wrapper.unmount();
  });

  it("falls back without blaming the file when the host refuses it", async () => {
    stubFetch(() =>
      Promise.resolve({
        ok: false,
        status: 404,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
      }),
    );
    const wrapper = mount(MeshViewer, { props: { src: "/media/gone.glb" } });
    await flushPromises();

    expect(wrapper.get("[data-test=mesh-viewer-note]").text()).toContain(
      "couldn't start",
    );
    expect(wrapper.emitted("fail")).toHaveLength(1);
    wrapper.unmount();
  });

  it("aborts the in-flight fetch on unmount", async () => {
    stubFetch(() => new Promise(() => {}));
    const wrapper = mount(MeshViewer, { props: { src: "/media/mesh.glb" } });
    expect(calls).toHaveLength(1);
    expect(calls[0]?.signal?.aborted).toBe(false);

    wrapper.unmount();
    await flushPromises();
    expect(calls[0]?.signal?.aborted).toBe(true);
    // An aborted load is not a failure the viewer should announce.
    expect(wrapper.emitted("fail")).toBeUndefined();
  });

  it("refetches when the src changes", async () => {
    stubFetch(() => new Promise(() => {}));
    const wrapper = mount(MeshViewer, { props: { src: "/media/one.glb" } });
    await wrapper.setProps({ src: "/media/two.glb" });
    await flushPromises();

    expect(calls.map((call) => call.url)).toEqual([
      "/media/one.glb",
      "/media/two.glb",
    ]);
    expect(calls[0]?.signal?.aborted).toBe(true);
    wrapper.unmount();
  });

  it("labels the canvas for assistive technology", () => {
    stubFetch(() => new Promise(() => {}));
    const wrapper = mount(MeshViewer, {
      props: { src: "/media/mesh.glb", alt: "A ceramic owl" },
    });
    const canvas = wrapper.get("[data-test=mesh-viewer-canvas]");
    expect(canvas.attributes("role")).toBe("img");
    expect(canvas.attributes("tabindex")).toBe("0");
    expect(canvas.attributes("aria-label")).toContain("A ceramic owl");
    wrapper.unmount();
  });
});
