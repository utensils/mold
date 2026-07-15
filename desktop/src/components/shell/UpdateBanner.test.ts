import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { mount } from "@vue/test-utils";

const install = vi.fn();

vi.mock("../../stores/updater", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../stores/updater")>();
  return {
    ...actual,
    useUpdaterStore: () => ({
      candidate: {
        id: "candidate-1",
        version: "0.18.0",
        publishedAt: null,
        notes: null,
      },
      phase: "available",
      shouldNotify: true,
      isBusy: false,
      install,
      dismissCandidate: vi.fn(),
    }),
  };
});

import UpdateBanner from "./UpdateBanner.vue";

describe("UpdateBanner", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    install.mockReset();
  });

  it("announces an automatically discovered update and installs only on click", async () => {
    const wrapper = mount(UpdateBanner);

    expect(wrapper.text()).toContain("Mold 0.18.0 is available");
    expect(install).not.toHaveBeenCalled();

    await wrapper.get('[data-test="banner-install-update"]').trigger("click");
    expect(install).toHaveBeenCalledOnce();
  });
});
