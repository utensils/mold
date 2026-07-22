import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AboutSection from "./AboutSection.vue";

const { openExternalMock } = vi.hoisted(() => ({ openExternalMock: vi.fn() }));
vi.mock("../../lib/openExternal", () => ({ openExternal: openExternalMock }));

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn().mockRejectedValue(new Error("offline")),
}));

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { openLogsDir: vi.fn() },
}));

describe("AboutSection", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    openExternalMock.mockClear();
  });

  it("credits both core contributors", () => {
    const wrapper = mount(AboutSection);

    expect(wrapper.text()).toContain("Core contributors");
    expect(wrapper.text()).toContain("James Brink");
    expect(wrapper.text()).toContain("Jeffrey Dilley");
    expect(wrapper.text()).not.toMatch(/equal (project )?owners/i);
  });

  it("opens the public privacy policy", async () => {
    const wrapper = mount(AboutSection);

    await wrapper.get("[data-test='desktop-privacy-policy']").trigger("click");

    expect(openExternalMock).toHaveBeenCalledOnce();
    expect(openExternalMock).toHaveBeenCalledWith("https://utensils.io/mold/privacy");
  });
});
