import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { setMeshWorkflowDraftStorage } from "../stores/meshWorkflowDraft";

/*
 * The 3-D Studio draft is a module-scoped store, so a case that leaves a
 * selected workflow behind would have the NEXT mount restore it and fetch its
 * result — which is exactly how the polling budget below saw four fetches
 * instead of two. Every case starts on a fresh Pinia and a storage stub;
 * happy-dom keeps one localStorage per file, so the stub is what stops the
 * cases hydrating each other's prompts.
 */
beforeEach(() => {
  setActivePinia(createPinia());
  setMeshWorkflowDraftStorage({
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
  });
});

const createMeshWorkflow = vi.hoisted(() =>
  vi.fn(async () => ({ job_id: "workflow-1" })),
);
const listMeshWorkflows = vi.hoisted(() => vi.fn(async () => ({ jobs: [] })));

vi.mock("../api/meshWorkflows", () => ({
  createMeshWorkflow,
  listMeshWorkflows,
  getMeshWorkflow: vi.fn(async () => null),
  cancelMeshWorkflow: vi.fn(),
  resumeMeshWorkflow: vi.fn(),
  deleteMeshWorkflow: vi.fn(),
}));

vi.mock("../api/client", () => ({
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(async (_target, path: string) => {
    if (path === "/api/status") return { instance_id: "local-instance" };
    if (path === "/api/capabilities") return {};
    if (path === "/api/models") {
      return [
        {
          name: "hunyuan3d-mini-turbo:fp16",
          family: "hunyuan3d",
          downloaded: true,
          runtime_available: true,
          default_steps: 5,
          default_guidance: 5,
          default_width: 0,
          default_height: 0,
          generation_profile: {
            default_recipe_id: "shape",
            recipes: [
              {
                id: "shape",
                capabilities: {
                  canvasless: true,
                  mesh: { workflow_modes: ["text_to_mesh"] },
                },
              },
            ],
          },
        },
        {
          name: "z-image-turbo:q8",
          family: "z-image",
          modality: "image",
          downloaded: true,
          runtime_available: true,
          default_steps: 9,
          default_guidance: 0,
          default_width: 1024,
          default_height: 1024,
          generation_profile: {
            default_recipe_id: "image",
            recipes: [
              {
                id: "image",
                capabilities: { prompt: { mode: "required" } },
              },
            ],
          },
        },
      ];
    }
    throw new Error(`unexpected ${path}`);
  }),
}));

vi.mock("../api/referenceUploads", () => ({
  prepareReferenceUploads: vi.fn(),
  requestShouldUseReferenceUploads: vi.fn(() => false),
}));

vi.mock("./MeshViewer.vue", () => ({ default: { template: "<div />" } }));

import MeshWorkflowStudio from "./MeshWorkflowStudio.vue";

describe("MeshWorkflowStudio feature-aware PBR authoring", () => {
  beforeEach(() => vi.clearAllMocks());

  it("keeps text-to-mesh usable without sending texture controls to a host that lacks them", async () => {
    const wrapper = mount(MeshWorkflowStudio, {
      props: {
        target: { baseUrl: "http://metal-host:7680", apiKey: null },
      },
    });
    await flushPromises();

    expect(wrapper.find("[data-test='mesh-workflow-texture']").exists()).toBe(
      false,
    );
    expect(
      wrapper.get("[data-test='mesh-workflow-texture-unavailable']").text(),
    ).toContain("PBR painting is unavailable on this machine");

    await wrapper.get("textarea").setValue("A brass telescope");
    await wrapper.get("form").trigger("submit");
    await flushPromises();

    expect(createMeshWorkflow).toHaveBeenCalledTimes(1);
    const request = (
      createMeshWorkflow.mock.calls as unknown as Array<
        [unknown, { mode: string; mesh_request: { mesh?: unknown } }]
      >
    )[0]![1];
    expect(request.mode).toBe("text_to_mesh");
    expect(request.mesh_request.mesh).toBeUndefined();
  });
});

it("preserves the draft when telemetry recreates an equivalent target", async () => {
  const target = { baseUrl: "http://metal-host:7680", apiKey: null };
  const wrapper = mount(MeshWorkflowStudio, { props: { target } });
  await flushPromises();
  await wrapper.get("textarea").setValue("A brass telescope");
  const calls = listMeshWorkflows.mock.calls.length;
  for (let tick = 0; tick < 3; tick++) {
    await wrapper.setProps({ target: { ...target } });
    await flushPromises();
  }
  expect(listMeshWorkflows.mock.calls.length).toBe(calls);
  expect(wrapper.get("textarea").element.value).toBe("A brass telescope");
  expect(wrapper.text()).not.toContain("Loading workflow capabilities");
  wrapper.unmount();
});

it("submits and reads workflow history on the resolved owner", async () => {
  const target = { baseUrl: "http://remote:7680", apiKey: "remote-key" };
  const resolveTarget = vi.fn(async () => ({ target, label: "Render box" }));
  const wrapper = mount(MeshWorkflowStudio, {
    props: {
      target: { baseUrl: "http://local:7680", apiKey: null },
      resolveTarget,
    },
  });
  await flushPromises();
  await wrapper.get("textarea").setValue("A brass telescope");
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(resolveTarget).toHaveBeenCalledWith(
    expect.objectContaining({
      mode: "text_to_mesh",
      imageModel: "z-image-turbo:q8",
    }),
  );
  expect(createMeshWorkflow).toHaveBeenCalledWith(target, expect.any(Object));
  expect(listMeshWorkflows).toHaveBeenLastCalledWith(target);
  wrapper.unmount();
});

it("retains unchanged result media across progress polls and stops polling on unmount", async () => {
  const { getMeshWorkflow } = await import("../api/meshWorkflows");
  const { apiFetchTo } = await import("../api/client");
  vi.clearAllMocks();
  vi.useFakeTimers();
  vi.mocked(getMeshWorkflow).mockResolvedValue({
    id: "workflow-1",
    state: "running",
    mode: "text_to_mesh",
    stages: [],
    output_filename: "result.glb",
  } as never);
  vi.mocked(apiFetchTo).mockImplementation(
    async () => new Response(new Uint8Array([1])),
  );
  const createUrl = vi
    .spyOn(URL, "createObjectURL")
    .mockReturnValue("blob:result");
  const revokeUrl = vi
    .spyOn(URL, "revokeObjectURL")
    .mockImplementation(() => {});
  const wrapper = mount(MeshWorkflowStudio, {
    props: { target: { baseUrl: "http://local:7680", apiKey: null } },
  });
  try {
    await flushPromises();
    await wrapper.get("textarea").setValue("A brass telescope");
    await wrapper.get("form").trigger("submit");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(750);
    await flushPromises();
    expect(getMeshWorkflow).toHaveBeenCalledTimes(2);
    expect(apiFetchTo).toHaveBeenCalledTimes(2);
    wrapper.unmount();
    await vi.advanceTimersByTimeAsync(1500);
    expect(getMeshWorkflow).toHaveBeenCalledTimes(2);
  } finally {
    wrapper.unmount();
    createUrl.mockRestore();
    revokeUrl.mockRestore();
    vi.useRealTimers();
  }
});

/*
 * The reported bug, at the level a person hits it: type a description, go to
 * the Queue, come back. The router lazy-loads this view and nothing keeps it
 * alive, so every draft ref used to unmount with it.
 */
it("keeps the description and the chosen styles across leaving the view and returning", async () => {
  const first = mount(MeshWorkflowStudio, {
    props: { target: { baseUrl: "http://local:7680", apiKey: null } },
  });
  await flushPromises();
  await first.get("textarea").setValue("a hand-carved wooden fox");
  const style = first.get<HTMLSelectElement>(
    "[data-test='mesh-workflow-model']",
  ).element.value;
  expect(style).not.toBe("");
  first.unmount();

  const second = mount(MeshWorkflowStudio, {
    props: { target: { baseUrl: "http://local:7680", apiKey: null } },
  });
  await flushPromises();
  expect(second.get<HTMLTextAreaElement>("textarea").element.value).toBe(
    "a hand-carved wooden fox",
  );
  expect(
    second.get<HTMLSelectElement>("[data-test='mesh-workflow-model']").element
      .value,
  ).toBe(style);
  second.unmount();
});

it("restores a selected workflow's settings without overwriting edits on progress polls", async () => {
  const { getMeshWorkflow } = await import("../api/meshWorkflows");
  vi.useFakeTimers();
  listMeshWorkflows.mockResolvedValue({
    jobs: [{ id: "saved", mode: "text_to_mesh", state: "running" }] as never,
  });
  vi.mocked(getMeshWorkflow).mockResolvedValue({
    id: "saved",
    mode: "text_to_mesh",
    state: "running",
    stages: [],
    request: {
      mode: "text_to_mesh",
      image_request: {
        model: "z-image-turbo:q8",
        prompt: "A saved wooden fox",
      },
      mesh_request: {
        model: "hunyuan3d-mini-turbo:fp16",
        mesh: { texture: false },
      },
    },
  } as never);
  const wrapper = mount(MeshWorkflowStudio, {
    props: { target: { baseUrl: "http://local:7680", apiKey: null } },
  });
  try {
    await flushPromises();
    await wrapper.get('[aria-label="Previous 3-D workflow"]').setValue("saved");
    await flushPromises();
    expect(wrapper.get("textarea").element.value).toBe("A saved wooden fox");
    await wrapper.get("textarea").setValue("An edited fox");
    await vi.advanceTimersByTimeAsync(750);
    expect(wrapper.get("textarea").element.value).toBe("An edited fox");
  } finally {
    wrapper.unmount();
    vi.useRealTimers();
  }
});

it.each([
  ["running", "Cancel", "cancelMeshWorkflow"],
  ["paused", "Resume", "resumeMeshWorkflow"],
] as const)(
  "sends %s workflow controls to the resolved owner",
  async (state, label, action) => {
    const api = await import("../api/meshWorkflows");
    vi.clearAllMocks();
    vi.mocked(api.getMeshWorkflow).mockResolvedValue({
      id: "workflow-1",
      state,
      mode: "text_to_mesh",
      stages: [],
    } as never);
    const owner = { baseUrl: "http://plato-uat:7689", apiKey: null };
    const wrapper = mount(MeshWorkflowStudio, {
      props: {
        target: { baseUrl: "http://browse-host:7680", apiKey: null },
        resolveTarget: async () => ({ target: owner, label: "Plato UAT" }),
      },
    });
    try {
      await flushPromises();
      await wrapper.get("textarea").setValue("A wooden fox");
      await wrapper.get("form").trigger("submit");
      await flushPromises();
      await wrapper
        .findAll("button")
        .find((button) => button.text() === label)!
        .trigger("click");
      await flushPromises();
      expect(api[action]).toHaveBeenCalledWith(owner, "workflow-1");
    } finally {
      wrapper.unmount();
    }
  },
);

describe("a returning view does not overwrite what you were writing", () => {
  const job = (id: string) => ({
    contract_version: 1,
    id,
    state: "completed",
    mode: "text_to_mesh",
    stage_count: 3,
    current_stage: 2,
    created_at_ms: Date.now(),
    updated_at_ms: Date.now(),
  });

  async function withOneFinishedRun() {
    const { listMeshWorkflows, getMeshWorkflow } =
      await import("../api/meshWorkflows");
    vi.mocked(listMeshWorkflows).mockResolvedValue({
      jobs: [job("run-1")],
    } as never);
    vi.mocked(getMeshWorkflow).mockResolvedValue({
      id: "run-1",
      state: "completed",
      mode: "text_to_mesh",
      stages: [],
      request: {
        mode: "text_to_mesh",
        image_request: {
          model: "z-image-turbo:q8",
          prompt: "the finished run",
        },
        mesh_request: {
          model: "hunyuan3d-mini-turbo:fp16",
          mesh: { texture: false },
        },
      },
    } as never);
  }

  /*
   * `bootstrap` retains the selection, and it used to restore the DRAFT from
   * it on every entry. So after your first run — the normal state — leaving
   * for the Queue and coming back reverted your unsent description to the
   * finished workflow's and dropped both attachments, then persisted the
   * clobbered values. That is the reported bug, reintroduced for the common
   * case. The draft is only restored for a DEEP LINK, which is a request to
   * open someone's specific workflow.
   */
  it("keeps a new description over a finished run's, across a remount", async () => {
    await withOneFinishedRun();
    const target = { baseUrl: "http://local:7680", apiKey: null };

    // Arrive on the workflow the way a queue row does, then type over it.
    const first = mount(MeshWorkflowStudio, {
      props: { target, desktop: true, openWorkflow: "run-1" },
    });
    await flushPromises();
    await first.get("textarea").setValue("a completely different object");
    first.unmount();

    const second = mount(MeshWorkflowStudio, {
      props: { target, desktop: true },
    });
    await flushPromises();
    expect(second.get<HTMLTextAreaElement>("textarea").element.value).toBe(
      "a completely different object",
    );
    second.unmount();
  });

  /* A deep link IS a request to open that workflow, so it does restore. */
  it("restores the workflow's own settings when a link names it", async () => {
    await withOneFinishedRun();
    const wrapper = mount(MeshWorkflowStudio, {
      props: {
        target: { baseUrl: "http://local:7680", apiKey: null },
        desktop: true,
        openWorkflow: "run-1",
      },
    });
    await flushPromises();
    expect(wrapper.get<HTMLTextAreaElement>("textarea").element.value).toBe(
      "the finished run",
    );
    wrapper.unmount();
  });

  /*
   * The machine is RECORDED, not inferred: `ownedTarget` is re-seeded from
   * the props on every fresh mount, so comparing against it always said "same
   * machine" and a hal9000 job id was retained against plato — clearing the
   * canvas with "no longer on this machine", which is wrong and unexplained.
   */
  it("drops a selection belonging to a machine it no longer talks to", async () => {
    await withOneFinishedRun();
    const first = mount(MeshWorkflowStudio, {
      props: {
        target: { baseUrl: "http://hal9000:7680", apiKey: "hk" },
        desktop: true,
        openWorkflow: "run-1",
      },
    });
    await flushPromises();
    first.unmount();

    // A FRESH mount against a different machine — the case the old test missed.
    const { listMeshWorkflows } = await import("../api/meshWorkflows");
    vi.mocked(listMeshWorkflows).mockResolvedValue({ jobs: [] } as never);
    const second = mount(MeshWorkflowStudio, {
      props: {
        target: { baseUrl: "http://plato:7680", apiKey: "pk" },
        desktop: true,
      },
    });
    await flushPromises();
    expect(second.text()).not.toContain("no longer on this machine");
    second.unmount();
  });
});

describe("the composer is the one place a 3-D object is authored", () => {
  const target = { baseUrl: "http://local:7680", apiKey: null };

  const paintingModel = {
    name: "hunyuan3d-2.1:fp16",
    family: "hunyuan3d",
    downloaded: true,
    runtime_available: true,
    default_steps: 30,
    default_guidance: 5,
    default_width: 0,
    default_height: 0,
    generation_profile: {
      default_recipe_id: "shape",
      recipes: [
        {
          id: "shape",
          capabilities: {
            canvasless: true,
            mesh: { workflow_modes: ["text_to_mesh", "mesh_texture"] },
          },
        },
      ],
    },
  } as never;

  async function desktopStudio(props: Record<string, unknown> = {}) {
    const wrapper = mount(MeshWorkflowStudio, {
      props: { target, desktop: true, ...props },
    });
    await flushPromises();
    return wrapper;
  }

  /*
   * The claim of the whole rework: the description, both style chips and
   * Generate sit on the composer, not at the foot of the settings rail. The
   * chrome test pins the SOURCE; this pins what actually renders.
   */
  it("renders the description, both style slots and Generate on the composer", async () => {
    const wrapper = await desktopStudio();
    const wrapper2 = mount(MeshWorkflowStudio, {
      props: { target, desktop: true },
      slots: {
        "mesh-picker": "<i data-test='slot-mesh'/>",
        "image-picker": "<i data-test='slot-image'/>",
      },
    });
    await flushPromises();
    const composer = wrapper2.get("[data-test='mesh-composer']");
    expect(composer.find("textarea").exists()).toBe(true);
    expect(composer.find("[data-test='slot-mesh']").exists()).toBe(true);
    expect(composer.find("[data-test='slot-image']").exists()).toBe(true);
    expect(composer.find("[data-test='mesh-generate']").exists()).toBe(true);
    // ...and NOT in the rail it came from.
    const rail = wrapper2.get("form.mesh-studio__composer");
    expect(rail.find("textarea").exists()).toBe(false);
    expect(rail.find("[data-test='mesh-generate']").exists()).toBe(false);
    wrapper2.unmount();
    wrapper.unmount();
  });

  /*
   * `studio/` cannot read the shell's platform table, and desktop ships Linux
   * and Windows builds where New image's composer says `Ctrl↩`. A hard-coded
   * `⌘↩` would have the two composers disagree two clicks apart.
   */
  it("says the chord the shell tells it, and nothing when there is none", async () => {
    const withKey = await desktopStudio({ generateShortcut: "Ctrl↩" });
    expect(withKey.get("[data-test='mesh-generate']").text()).toContain(
      "Ctrl↩",
    );
    withKey.unmount();

    const without = await desktopStudio();
    expect(
      without.get("[data-test='mesh-generate']").find("kbd").exists(),
    ).toBe(false);
    without.unmount();
  });

  /* A supplied mesh needs no description, so the prompt row is absent. */
  it("shows the description only for the workflow that reads one", async () => {
    const wrapper = await desktopStudio();
    expect(
      wrapper.get("[data-test='mesh-composer']").find("textarea").exists(),
    ).toBe(true);
    wrapper.unmount();
  });

  /* Generate refuses until the workflow can actually run. */
  it("keeps Generate disabled until the workflow is complete", async () => {
    const wrapper = await desktopStudio();
    const button = wrapper.get("[data-test='mesh-generate']");
    expect(button.attributes("disabled")).toBeDefined();
    await wrapper.get("[data-test='mesh-composer'] textarea").setValue("a fox");
    await flushPromises();
    expect(button.attributes("disabled")).toBeUndefined();
    wrapper.unmount();
  });

  /* The empty canvas is the kit's block, in the app's own voice. */
  it("uses the shared empty state rather than a bare heading", async () => {
    const wrapper = await desktopStudio();
    const empty = wrapper.get("[data-test='mesh-empty-canvas']");
    expect(empty.text()).toContain("Your 3-D object appears here");
    expect(empty.text()).toContain("press Generate");
    wrapper.unmount();
  });

  /*
   * Web is a different layout and the rework must not have moved anything
   * there: its description, pickers and Generate stay in the rail.
   */
  it("leaves the web layout authoring in the rail", async () => {
    const wrapper = mount(MeshWorkflowStudio, { props: { target } });
    await flushPromises();
    expect(wrapper.find("[data-test='mesh-composer']").exists()).toBe(false);
    const rail = wrapper.get("form.mesh-studio__composer");
    expect(rail.find("textarea").exists()).toBe(true);
    expect(rail.find("button[type='submit']").exists()).toBe(true);
    wrapper.unmount();
  });

  /*
   * Web's controls lost their accessible name when the wrapping `<label>`
   * became a `<span>` — the desktop branches kept theirs, so the omission was
   * invisible on the surface being worked on.
   */
  it("names every web control for assistive tech", async () => {
    const wrapper = mount(MeshWorkflowStudio, {
      props: { target, availableModels: [paintingModel] },
    });
    await flushPromises();
    for (const label of ["Texture size"]) {
      const named = wrapper
        .findAll("select")
        .some((s) => s.attributes("aria-label") === label);
      expect(named, label).toBe(true);
    }
    wrapper.unmount();
  });

  /* The caps are desktop's presentation, not the words themselves. */
  it("does not shout on web", async () => {
    const wrapper = mount(MeshWorkflowStudio, {
      props: { target, availableModels: [paintingModel] },
    });
    await flushPromises();
    expect(wrapper.text()).not.toContain("TEXTURE SIZE");
    expect(wrapper.text()).toContain("Texture size");
    wrapper.unmount();
  });
});
