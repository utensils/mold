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
    expect(view.attributes("aria-label")).toBe("Hunyuan3D · plato");
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
});
