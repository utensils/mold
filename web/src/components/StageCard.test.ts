import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import StageCard from "./StageCard.vue";
import type { ChainStageToml } from "../lib/chainToml";

/** Minimum props for the card; tests override `stage` per case. */
function props(stage: ChainStageToml, isFirst = false) {
  return {
    index: 0,
    isFirst,
    stage,
    framesPerClipCap: 97,
    fadeFramesMax: 32,
  };
}

describe("StageCard source image affordance", () => {
  it("shows the attach button when no image is set", () => {
    const card = mount(StageCard, {
      props: props({ prompt: "", frames: 97 }),
    });
    // Both the 12×12 empty-state slot and the inline "Attach image" link
    // should point at the pick-image event.
    const attach = card.find('[aria-label="Attach source image"]');
    expect(attach.exists()).toBe(true);
    expect(card.text()).toContain("Attach image");
  });

  it("emits pick-image when the attach button is clicked", async () => {
    const card = mount(StageCard, {
      props: props({ prompt: "", frames: 97 }),
    });
    await card.find('[aria-label="Attach source image"]').trigger("click");
    expect(card.emitted("pick-image")).toHaveLength(1);
  });

  it("renders the thumbnail and clear button when an image is attached", async () => {
    const b64 = "iVBORw0KGgo"; // truncated — we only check the src contains it
    const card = mount(StageCard, {
      props: props({
        prompt: "scene",
        frames: 97,
        source_image: "hero.png",
        source_image_b64: b64,
      }),
    });
    const img = card.find("img");
    expect(img.exists()).toBe(true);
    expect(img.attributes("src")).toContain(b64);

    const clear = card.find('[aria-label="Remove source image"]');
    expect(clear.exists()).toBe(true);
    await clear.trigger("click");
    expect(card.emitted("clear-image")).toHaveLength(1);
  });

  it("labels continuation stages with the identity-anchor hint on smooth", () => {
    // Smooth continuations primarily use the motion tail — the attached
    // image still anchors character/scene identity past the tail window.
    // The hint label communicates that so users don't think the image is
    // a no-op.
    const card = mount(StageCard, {
      props: props(
        {
          prompt: "scene",
          frames: 97,
          transition: "smooth",
          source_image: "hero.png",
          source_image_b64: "x",
        },
        false,
      ),
    });
    expect(card.text()).toContain("identity anchor");
  });
});
