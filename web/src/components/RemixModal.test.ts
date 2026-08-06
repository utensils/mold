import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import RemixModal from "./RemixModal.vue";

const remixPromptMock = vi.fn();
vi.mock("../api", () => ({
  remixPrompt: (...args: unknown[]) => remixPromptMock(...args),
}));

describe("RemixModal", () => {
  beforeEach(() => {
    remixPromptMock.mockReset();
    remixPromptMock.mockResolvedValue({
      source_prompt: "root idea",
      root_prompt: "root idea",
      source_kind: "original",
      variants: ["one", "two", "three"].map((prompt) => ({
        prompt,
        dimensions: ["camera"],
      })),
    });
  });

  it("defaults to the original, requests exactly three, and preserves the selected host", async () => {
    const target = { baseUrl: "http://studio:7680", apiKey: "secret" };
    const wrapper = mount(RemixModal, {
      props: {
        open: true,
        prompt: "expanded idea",
        originalPrompt: "root idea",
        family: "flux",
        task: "text-to-image",
        target,
      },
    });
    await wrapper.get('[data-test="remix-generate"]').trigger("click");
    await flushPromises();
    expect(remixPromptMock).toHaveBeenCalledWith(
      expect.objectContaining({
        source_prompt: "root idea",
        root_prompt: "root idea",
        source_kind: "original",
        variations: 3,
      }),
      undefined,
      target,
    );
    expect(wrapper.findAll('[data-test^="apply-remix-"]')).toHaveLength(3);

    await wrapper.get('[data-test="select-remix-2"]').setValue(false);
    await wrapper.get('[data-test="prepare-remix-batch"]').trigger("click");
    expect(wrapper.emitted("prepare")?.[0]?.[0]).toMatchObject({
      variants: [{ prompt: "one" }, { prompt: "two" }],
    });
  });

  it("offers only backend-safe dimensions for conditioned video and locked styles", async () => {
    const wrapper = mount(RemixModal, {
      props: {
        open: true,
        prompt: "move the lighthouse beam",
        family: "ltx2",
        task: "image-to-video",
        style: "Cinematic look",
      },
    });
    expect(
      wrapper.find('[data-test="remix-dimension-movement"]').exists(),
    ).toBe(true);
    expect(wrapper.find('[data-test="remix-dimension-camera"]').exists()).toBe(
      false,
    );
    expect(wrapper.find('[data-test="remix-dimension-style"]').exists()).toBe(
      false,
    );
    await wrapper.get('[data-test="remix-generate"]').trigger("click");
    await flushPromises();
    expect(remixPromptMock).toHaveBeenCalledWith(
      expect.objectContaining({ dimensions: ["movement"] }),
      undefined,
      undefined,
    );
  });
});
