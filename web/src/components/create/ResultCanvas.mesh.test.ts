import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ResultCanvas from "./ResultCanvas.vue";

/**
 * A finished mesh is an interactive viewer, not a still: the poster is the
 * fallback the viewer itself falls back to, and the mesh arm is probed before
 * the video one because a mesh carries neither frames nor samples.
 */

const MeshViewerStub = {
  name: "MeshViewer",
  props: {
    src: { type: String, default: "" },
    poster: { type: String, default: "" },
    alt: { type: String, default: "" },
    autoRotate: { type: Boolean, default: false },
    expandable: { type: Boolean, default: false },
  },
  template: "<div data-test='mesh-viewer-stub' />",
};

function mountMesh(props: Record<string, unknown> = {}) {
  return mount(ResultCanvas, {
    props: {
      mode: "result",
      resultSrc: "data:image/png;base64,POSTER",
      resultMeshSrc: "blob:mesh-1",
      resultCaption: "hunyuan3d · 49,152 tris · 24,576 verts",
      ...props,
    },
    global: { stubs: { MeshViewer: MeshViewerStub } },
  });
}

describe("ResultCanvas 3-D mesh", () => {
  it("mounts the shared viewer with the poster as its fallback", () => {
    const wrapper = mountMesh();
    const viewer = wrapper.findComponent(MeshViewerStub);
    expect(viewer.exists()).toBe(true);
    expect(viewer.props("src")).toBe("blob:mesh-1");
    expect(viewer.props("poster")).toBe("data:image/png;base64,POSTER");
    expect(viewer.props("autoRotate")).toBe(true);
    expect(viewer.props("expandable")).toBe(true);
    expect(wrapper.find("img.canvas__img").exists()).toBe(false);
    expect(wrapper.find("[data-test='canvas-video']").exists()).toBe(false);
  });

  it("keeps the provenance caption under the viewer", () => {
    const wrapper = mountMesh();
    expect(wrapper.get("[data-test='canvas-caption']").text()).toContain(
      "49,152 tris",
    );
  });

  // A stitched clip's poster and a mesh poster are both PNGs in `resultSrc`;
  // only the GLB source decides which arm draws.
  it("still renders the video arm when no mesh source is present", () => {
    const wrapper = mount(ResultCanvas, {
      props: {
        mode: "result",
        resultSrc: "data:image/png;base64,POSTER",
        resultVideoSrc: "data:video/mp4;base64,CLIP",
      },
      global: { stubs: { MeshViewer: MeshViewerStub } },
    });
    expect(wrapper.find("[data-test='mesh-viewer-stub']").exists()).toBe(false);
    expect(wrapper.find("[data-test='canvas-video']").exists()).toBe(true);
  });
});
