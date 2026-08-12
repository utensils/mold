import { describe, expect, it, vi } from "vitest";

const apiJson = vi.hoisted(() => vi.fn());
const apiJsonTo = vi.hoisted(() => vi.fn());

vi.mock("./client", () => ({
  apiJson,
  apiJsonTo,
}));

import { fetchChainLimits } from "./chains";

describe("chain limits API", () => {
  it("requests the cap for the active fps", async () => {
    apiJson.mockResolvedValueOnce({ frames_per_clip_cap: 241 });

    await fetchChainLimits("ltx-2-19b-distilled:fp8", null, 12);

    expect(apiJson).toHaveBeenCalledWith(
      "/api/capabilities/chain-limits?model=ltx-2-19b-distilled%3Afp8&fps=12",
    );
  });
});
