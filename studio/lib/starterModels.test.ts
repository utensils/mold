import { describe, expect, it } from "vitest";
import { MODEL_PRESENTATIONS } from "./generated/modelPresentationV1";
import { modelDisplayName } from "./modelDisplay";
import { STARTER_MODELS } from "./starterModels";

describe("STARTER_MODELS", () => {
  it("names every starter through the presentation authority", () => {
    for (const starter of STARTER_MODELS) {
      expect(MODEL_PRESENTATIONS[starter.model], starter.model).toBeDefined();
      expect(starter).not.toHaveProperty("displayName");
      expect(modelDisplayName({ name: starter.model })).not.toBe(starter.model);
    }
  });
});
