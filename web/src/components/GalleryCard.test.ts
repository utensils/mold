import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import GalleryCard from "./GalleryCard.vue";
import type { GalleryImage } from "../types";

class FakeIntersectionObserver {
  observe() {}
  disconnect() {}
}

const item: GalleryImage = {
  filename: "visible.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "visible image",
    model: "flux-dev:fp16",
    seed: 42,
    steps: 20,
    guidance: 3.5,
    width: 1024,
    height: 1024,
    version: "test",
  },
};

describe("GalleryCard visibility", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IntersectionObserver: typeof FakeIntersectionObserver;
      }
    ).IntersectionObserver = FakeIntersectionObserver;
  });

  afterEach(() => {
    delete (globalThis as Partial<typeof globalThis>).IntersectionObserver;
  });

  it("does not render a reveal overlay when legacy hide props are present", () => {
    const wrapper = mount(GalleryCard, {
      props: { item, hideMode: true, revealed: false },
    });

    expect(wrapper.text()).not.toContain("Reveal");
  });
});
