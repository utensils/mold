import { describe, expect, it } from "vitest";
import {
  filterRestrictedModels,
  isModelAccessRestricted,
  modelAccessRestrictionFor,
  type ModelAccessCapabilityRecord,
} from "./modelAccess";
import {
  AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256,
  authenticatedMiniMaxH3Capabilities,
} from "./minimaxH3Inventory.testFixtures";

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

  it("hides models whose runtime contract is explicitly unavailable", () => {
    const models = [
      { name: "flux-dev:q8", family: "flux" },
      {
        name: "minimax-h3-fl2va:official-bf16",
        family: "minimax-h3",
        runtime_available: false,
      },
    ];
    expect(filterRestrictedModels(models, undefined)).toEqual([models[0]]);
  });

  it("opens only the exact fully authenticated private FL2VA row", () => {
    const exact = {
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
      generation_profile: {
        profile_hash: AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256,
      },
    };
    const unrelated = { name: "hf:opaque", family: "minimax-h3" };
    expect(
      filterRestrictedModels(
        [exact, unrelated],
        authenticatedMiniMaxH3Capabilities(),
      ),
    ).toEqual([exact]);
  });

  it("keeps the family closed when request or component authority is incomplete", () => {
    const model = {
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
      generation_profile: {
        profile_hash: AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256,
      },
    };
    const missingRequest = authenticatedMiniMaxH3Capabilities();
    delete missingRequest.minimax_h3?.partitions[0]!.request;
    expect(filterRestrictedModels([model], missingRequest)).toEqual([]);

    const missingComponent = authenticatedMiniMaxH3Capabilities();
    missingComponent.minimax_h3!.components[0]!.state = "missing";
    expect(filterRestrictedModels([model], missingComponent)).toEqual([]);

    const widenedEnvelope = authenticatedMiniMaxH3Capabilities();
    widenedEnvelope.minimax_h3!.partitions[0]!.request!.frames = 141;
    expect(filterRestrictedModels([model], widenedEnvelope)).toEqual([]);

    const mismatchedProfile = authenticatedMiniMaxH3Capabilities();
    expect(
      filterRestrictedModels(
        [{ ...model, generation_profile: { profile_hash: "b".repeat(64) } }],
        mismatchedProfile,
      ),
    ).toEqual([]);
  });
});
