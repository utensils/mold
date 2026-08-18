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
