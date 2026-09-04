import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AboutSection from "./AboutSection.vue";
import { useConnectionStore } from "../../stores/connection";

const { openExternalMock } = vi.hoisted(() => ({ openExternalMock: vi.fn() }));
vi.mock("../../lib/openExternal", () => ({ openExternal: openExternalMock }));

const { apiJsonMock } = vi.hoisted(() => ({
  apiJsonMock: vi.fn().mockRejectedValue(new Error("offline")),
}));
vi.mock("../../lib/api/client", () => ({ apiJson: apiJsonMock }));

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { openLogsDir: vi.fn() },
}));

describe("AboutSection", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    openExternalMock.mockClear();
    apiJsonMock.mockReset();
    apiJsonMock.mockRejectedValue(new Error("offline"));
  });

  it("asks the engine again when the connection comes up", async () => {
    // Launching straight into Settings (restoreLastRoute) mounts this while
    // the engine is still starting. A mount-only read left Engine reading
    // "offline" for the rest of the session.
    const conn = useConnectionStore();
    const wrapper = mount(AboutSection);
    await flushPromises();
    expect(wrapper.text()).toContain("offline");

    apiJsonMock.mockResolvedValue({ version: "0.18.0", models_loaded: [] });
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: null };
    conn.status = "ready";
    await flushPromises();

    expect(wrapper.text()).toContain("mold 0.18.0");
  });

  it("credits both core contributors", () => {
    const wrapper = mount(AboutSection);

    expect(wrapper.get("[data-test='about-section-content']").classes()).toContain("w-full");
    expect(wrapper.get("[data-test='about-section-content']").classes()).not.toContain("max-w-md");
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
