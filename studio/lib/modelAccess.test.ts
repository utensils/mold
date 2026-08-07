import { describe, expect, it } from "vitest";
import {
  filterRestrictedModels,
  isModelAccessRestricted,
  modelAccessRestrictionFor,
  type ModelAccessCapabilityRecord,
} from "./modelAccess";

const capabilities: ModelAccessCapabilityRecord = {
  model_access: {
    restrictions: [
      {
        code: "MINIMAX_H3_AUTHORIZATION_REQUIRED",
        family: "minimax-h3",
        message: "MiniMax-H3 requires authorization.",
        license_url:
          "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
        authorization_url: "https://github.com/utensils/mold/issues/831",
      },
    ],
  },
};

describe("model access policy", () => {
  it.each([
    { model: "minimax-h3:bf16" },
    { model: "hf:MiniMaxAI/MiniMax-H3" },
    { model: "https://huggingface.co/MiniMaxAI/MiniMax_H3" },
    { model: "MiniMaxH3Scheduler" },
    { model: "MiniMaxH3Transformer3DModel" },
    { model: "AutoencoderKLMiniMaxH3" },
    { model: "cv:123", family: "MiniMax H3" },
  ])("matches restricted model and family aliases", (identity) => {
    expect(modelAccessRestrictionFor(capabilities, identity)?.code).toBe(
      "MINIMAX_H3_AUTHORIZATION_REQUIRED",
    );
  });

  it.each([
    { model: "h3" },
    { model: "minimax-h30" },
    { model: "minimaxh30" },
    { model: "notminimax-h3" },
    { model: "notminimaxh3" },
    { model: "flux-dev:q8", family: "flux" },
  ])("does not overmatch unrelated identities", (identity) => {
    expect(isModelAccessRestricted(capabilities, identity)).toBe(false);
  });

  it("keeps older-server rows and removes restricted current-server rows", () => {
    const models = [
      { name: "flux-dev:q8", family: "flux" },
      { name: "hf:opaque", family: "minimax-h3" },
    ];
    expect(filterRestrictedModels(models, undefined)).toEqual(models);
    expect(filterRestrictedModels(models, capabilities)).toEqual([models[0]]);
  });
});
