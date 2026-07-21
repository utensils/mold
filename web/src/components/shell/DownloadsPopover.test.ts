import { describe, it, expect } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import DownloadsPopover from "./DownloadsPopover.vue";

const makeJob = (over: Record<string, unknown> = {}) => ({
  id: "a",
  model: "flux-dev:q4",
  status: "active",
  catalog_id: null,
  files_done: 1,
  files_total: 5,
  bytes_done: 500_000,
  bytes_total: 1_000_000,
  current_file: "transformer.gguf",
  started_at: Date.now(),
  completed_at: null,
  error: null,
  ...over,
});

function mountPopover(props: Record<string, unknown>) {
  return mount(DownloadsPopover, {
    props: {
      open: true,
      active: [],
      queued: [],
      history: [],
      etaByJob: {},
      rateByJob: {},
      ...props,
    } as never,
  });
}

describe("DownloadsPopover", () => {
  it("renders nothing when closed", () => {
    const wrapper = mountPopover({ open: false, active: [makeJob() as never] });
    expect(wrapper.find('[data-test="downloads-popover"]').exists()).toBe(
      false,
    );
  });

  it("renders an active download with progress, file, rate and eta", () => {
    const wrapper = mountPopover({
      active: [makeJob() as never],
      etaByJob: { a: 5 },
      rateByJob: { a: 2_000_000 },
    });
    expect(wrapper.text()).toContain("flux-dev:q4");
    expect(wrapper.text()).toContain("transformer.gguf");
    expect(wrapper.text()).toMatch(/1\s*\/\s*5/);
    expect(wrapper.text()).toContain("50%");
    expect(wrapper.text()).toContain("2.0 MB/s");
    expect(wrapper.text()).toContain("eta 5s");
  });

  it("shows each concurrent download its own eta", () => {
    const wrapper = mountPopover({
      active: [
        makeJob({ id: "a", model: "model-a" }) as never,
        makeJob({ id: "b", model: "model-b" }) as never,
      ],
      etaByJob: { a: 5, b: 42 },
      rateByJob: {},
    });
    expect(wrapper.text()).toContain("eta 5s");
    expect(wrapper.text()).toContain("eta 42s");
  });

  it("cancels an active download", async () => {
    const wrapper = mountPopover({ active: [makeJob() as never] });
    await wrapper.get('[data-test="dl-cancel-a"]').trigger("click");
    expect(wrapper.emitted("cancel")?.[0]).toEqual(["a"]);
  });

  it("renders queued rows with their position and cancels them", async () => {
    const wrapper = mountPopover({
      queued: [
        makeJob({ id: "q1", model: "sd1.5:fp16", status: "queued" }) as never,
        makeJob({
          id: "q2",
          model: "flux-schnell:q4",
          status: "queued",
        }) as never,
      ],
    });
    expect(wrapper.text()).toContain("sd1.5:fp16");
    expect(wrapper.text()).toContain("flux-schnell:q4");
    expect(wrapper.text()).toMatch(/#1/);
    expect(wrapper.text()).toMatch(/#2/);
    await wrapper.get('[data-test="dl-cancel-q1"]').trigger("click");
    expect(wrapper.emitted("cancel")?.[0]).toEqual(["q1"]);
  });

  it("lists recently installed models with size and a retry on failure", async () => {
    const wrapper = mountPopover({
      history: [
        makeJob({
          id: "h1",
          model: "z-image:turbo",
          status: "completed",
          bytes_total: 3_650_722_201,
        }) as never,
        makeJob({
          id: "h2",
          model: "flux-dev:q4",
          status: "failed",
          error: "network blip",
        }) as never,
      ],
    });
    expect(wrapper.text()).toContain("Recently installed");
    expect(wrapper.text()).toContain("z-image:turbo");
    expect(wrapper.text()).toContain("3.4 GB");
    const btn = wrapper.get("[data-test=retry-h2]");
    await btn.trigger("click");
    expect(wrapper.emitted("retry")?.[0]).toEqual(["flux-dev:q4"]);
  });

  it("closes from the header button", async () => {
    const wrapper = mountPopover({ history: [makeJob() as never] });
    await wrapper.get('[aria-label="Close downloads"]').trigger("click");
    expect(wrapper.emitted("close")).toBeTruthy();
  });

  it("shows an empty message when there is nothing to show", () => {
    const wrapper = mountPopover({});
    expect(wrapper.text()).toContain("No downloads yet");
  });
});

function baseProps() {
  return {
    open: true,
    active: [],
    queued: [],
    history: [],
    etaByJob: {},
    rateByJob: {},
  };
}

describe("DownloadsPopover dismissal", () => {
  it("closes on Escape from anywhere, not just when focus is inside it", async () => {
    // The handler used to sit on the <aside>, which never receives key events
    // because opening the popover doesn't focus it — so Escape did nothing.
    const w = mount(DownloadsPopover, { props: baseProps() });
    await flushPromises();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(w.emitted("close")).toBeTruthy();
    w.unmount();
  });

  it("closes when clicking outside the panel", async () => {
    const w = mount(DownloadsPopover, {
      props: baseProps(),
      attachTo: document.body,
    });
    await flushPromises();
    document.body.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    await flushPromises();
    expect(w.emitted("close")).toBeTruthy();
    w.unmount();
  });

  it("stays open when clicking inside the panel", async () => {
    const w = mount(DownloadsPopover, {
      props: baseProps(),
      attachTo: document.body,
    });
    await flushPromises();
    const panel = w.get("[data-test='downloads-popover']");
    (panel.element as HTMLElement).dispatchEvent(
      new MouseEvent("mousedown", { bubbles: true }),
    );
    await flushPromises();
    expect(w.emitted("close")).toBeFalsy();
    w.unmount();
  });
});
