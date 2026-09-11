import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobileGenerationQueueCard from "./MobileGenerationQueueCard.vue";

describe("MobileGenerationQueueCard", () => {
  it("gives a prompt-free mesh job a visible and accessible title without an empty text row", () => {
    const view = mount(MobileGenerationQueueCard, {
      props: { title: "", subtitle: "Hunyuan3D · plato", status: "Decoding volume", ariaLabel: "" },
    });
    expect(view.get(".mobile-generation-job-copy p").text()).toBe("Hunyuan3D · plato");
    expect(view.find(".mobile-generation-job-copy span").exists()).toBe(false);
    expect(view.attributes("aria-label")).toBe("Hunyuan3D · plato. Decoding volume");
  });

  it("gives held-error summaries the full row width while retaining their text", () => {
    const detail = "GPU ran out of memory. Reduce the size before retrying.";
    const view = mount(MobileGenerationQueueCard, {
      props: { title: "Clip", subtitle: "LTX · plato", status: "HELD", detail },
    });
    expect(view.classes()).toContain("mobile-generation-job--detailed-status");
    expect(view.get("[data-test='mobile-generation-held-error']").text()).toBe(detail);
    expect(view.attributes("aria-label")).toContain(`HELD. ${detail}`);
  });

  it("activates with Enter and Space", async () => {
    const view = mount(MobileGenerationQueueCard, {
      props: {
        title: "Recovered print",
        subtitle: "MiniMax H3 FL2VA · plato",
        status: "STREAMING MINIMAX H3 TRANSFORMER BLOCKS · 17/20",
      },
    });
    const card = view.get("[data-test='mobile-generation-queue-card']");

    await card.trigger("keydown", { key: "Enter" });
    await card.trigger("keydown", { key: " " });

    expect(view.emitted("activate")).toHaveLength(2);
  });

  it("shows the picture being made, how far along, and which one of the batch", () => {
    const view = mount(MobileGenerationQueueCard, {
      props: {
        title: "Neon arcade, long exposure",
        subtitle: "FLUX · studio-rack",
        status: "Adding detail",
        thumbnailUrl: "blob:preview",
        progress: 64,
        meta: "image 2 of 4 · studio-rack",
      },
    });

    // The live latent preview already exists on Make; the queue had no idea.
    expect(view.get("[data-test='mobile-generation-job-thumb'] img").attributes("src")).toBe(
      "blob:preview",
    );
    expect(view.get("[role='progressbar']").attributes("aria-valuenow")).toBe("64");
    expect(view.get("[data-test='mobile-generation-job-meta']").text()).toBe(
      "image 2 of 4 · studio-rack",
    );
    expect(view.classes()).toContain("mobile-generation-job--active");
  });

  it("leaves out every part it was given nothing for", () => {
    const view = mount(MobileGenerationQueueCard, {
      props: { title: "Waiting print", subtitle: "FLUX · plato", status: "QUEUED" },
    });

    expect(view.find("[data-test='mobile-generation-job-thumb']").exists()).toBe(false);
    expect(view.find("[role='progressbar']").exists()).toBe(false);
    expect(view.find("[data-test='mobile-generation-job-meta']").exists()).toBe(false);
    expect(view.classes()).not.toContain("mobile-generation-job--active");
  });

  it("stands a waiting print's place in line where the picture would be", () => {
    const view = mount(MobileGenerationQueueCard, {
      props: { title: "Second print", subtitle: "FLUX · plato", status: "QUEUED", position: "2" },
    });

    // A queued row has no pixels yet, so the glyph square says its place.
    const glyph = view.get("[data-test='mobile-generation-job-position']");
    expect(glyph.text()).toBe("2");
    expect(view.find("[data-test='mobile-generation-job-thumb']").exists()).toBe(false);
  });

  it("marks a held print with the tone it was given, without losing the shared shape", () => {
    const view = mount(MobileGenerationQueueCard, {
      props: {
        title: "Held print",
        subtitle: "LTX · plato",
        status: "HELD",
        position: "↓",
        tone: "warning",
      },
    });

    expect(view.classes()).toContain("mobile-generation-job--warning");
    expect(view.get("[data-test='mobile-generation-status']").text()).toBe("HELD");
  });
});
