import { beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import NowDevelopingPopover from "./NowDevelopingPopover.vue";
import type { FleetActiveWork } from "@studio/api/activity";

beforeEach(() => {
  document.body.innerHTML = "";
});

function row(): FleetActiveWork {
  return {
    id: "job-1",
    kind: "print",
    phase: "denoise",
    model: "flux-dev:q8",
    created_at_unix_ms: 1,
    updated_at_unix_ms: 1,
    can_cancel: false,
    key: "origin/job-1",
    hostId: "origin",
    hostLabel: "this server",
    routeUrl: "http://origin:7680",
    instanceId: "origin-instance",
    stale: false,
    hostError: null,
  };
}

describe("NowDevelopingPopover", () => {
  it("dismisses on an outside pointerdown so sibling popovers never stack", async () => {
    const wrapper = mount(NowDevelopingPopover, {
      props: { rows: [row()] },
      attachTo: document.body,
    });

    await wrapper.get("[data-test='now-developing-trigger']").trigger("click");
    expect(wrapper.find("[data-test='now-developing-panel']").exists()).toBe(
      true,
    );

    document.body.dispatchEvent(
      new PointerEvent("pointerdown", { bubbles: true }),
    );
    await wrapper.vm.$nextTick();
    expect(wrapper.find("[data-test='now-developing-panel']").exists()).toBe(
      false,
    );
    wrapper.unmount();
  });

  it("stays open for pointerdowns inside its own panel", async () => {
    const wrapper = mount(NowDevelopingPopover, {
      props: { rows: [row()] },
      attachTo: document.body,
    });

    await wrapper.get("[data-test='now-developing-trigger']").trigger("click");
    await wrapper
      .get("[data-test='now-developing-panel']")
      .trigger("pointerdown");
    expect(wrapper.find("[data-test='now-developing-panel']").exists()).toBe(
      true,
    );
    wrapper.unmount();
  });
});

describe("the live-work chip's words", () => {
  // The mock's header chip: "Making 1 · 3 waiting" — the queue's own words
  // (Being made / Waiting), never a count with a noun nobody says.
  it("says how many are being made and how many wait", () => {
    const making = row();
    const waiting = {
      ...row(),
      id: "job-2",
      key: "origin/job-2",
      phase: "queued",
    };
    const alsoWaiting = {
      ...row(),
      id: "job-3",
      key: "origin/job-3",
      phase: "queued",
    };
    const wrapper = mount(NowDevelopingPopover, {
      props: { rows: [making, waiting, alsoWaiting] },
      attachTo: document.body,
    });
    expect(wrapper.get("[data-test='now-developing-trigger']").text()).toBe(
      "Making 1 · 2 waiting",
    );
    wrapper.unmount();
  });

  it("drops the half that is zero", () => {
    const wrapper = mount(NowDevelopingPopover, {
      props: { rows: [row()] },
      attachTo: document.body,
    });
    expect(wrapper.get("[data-test='now-developing-trigger']").text()).toBe(
      "Making 1",
    );
    wrapper.unmount();
  });
});
