import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import TopBar from "./TopBar.vue";

let routeName: "gallery" | "generate" = "gallery";

vi.mock("vue-router", () => ({
  useRoute: () => ({ name: routeName }),
}));

vi.mock("../composables/useDownloads", () => ({
  useDownloads: () => ({
    active: { value: null },
    queued: { value: [] },
  }),
}));

function mountTopBar() {
  return mount(TopBar, {
    props: {
      filter: "all",
      search: "",
      view: "grid",
      muted: true,
      counts: { total: 1, images: 1, video: 0, filtered: 1 },
      loading: false,
    },
    global: {
      stubs: {
        RouterLink: { template: "<a><slot /></a>" },
        ResourceStrip: { template: "<div />" },
      },
    },
  });
}

describe("TopBar visibility controls", () => {
  it("does not render gallery hide or unhide buttons", () => {
    routeName = "gallery";
    const wrapper = mountTopBar();

    expect(wrapper.find('[aria-label="Hide gallery"]').exists()).toBe(false);
    expect(wrapper.find('[aria-label="Unhide gallery"]').exists()).toBe(false);
  });

  it("does not render generate preview hide or reveal buttons", () => {
    routeName = "generate";
    const wrapper = mountTopBar();

    expect(wrapper.find('[aria-label="Hide previews"]').exists()).toBe(false);
    expect(wrapper.find('[aria-label="Reveal previews"]').exists()).toBe(false);
  });
});
