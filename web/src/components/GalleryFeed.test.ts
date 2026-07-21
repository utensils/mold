import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import GalleryFeed from "./GalleryFeed.vue";
import type { GalleryImage } from "../types";

class FakeIntersectionObserver {
  observe() {}
  disconnect() {}
}

const entries: GalleryImage[] = [
  {
    filename: "recent.png",
    timestamp: 1_700_000_000,
    format: "png",
    metadata: {
      prompt: "recent image",
      model: "flux-dev:fp16",
      seed: 7,
      steps: 20,
      guidance: 3.5,
      width: 1024,
      height: 1024,
      version: "test",
    },
  },
];

describe("GalleryFeed", () => {
  beforeEach(() => {
    (globalThis as any).IntersectionObserver = FakeIntersectionObserver;
  });

  afterEach(() => {
    delete (globalThis as Partial<typeof globalThis>).IntersectionObserver;
  });

  it("renders current entries", () => {
    const wrapper = mount(GalleryFeed, {
      props: {
        entries,
        loading: false,
        view: "grid",
        muted: true,
      },
    });

    expect(wrapper.text()).toContain("recent image");
  });
});
