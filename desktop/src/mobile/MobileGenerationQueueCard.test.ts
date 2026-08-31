import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobileGenerationQueueCard from "./MobileGenerationQueueCard.vue";

describe("MobileGenerationQueueCard", () => {
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
