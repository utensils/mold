import { describe, expect, it } from "vitest";

import { HeldPullOffers } from "./heldPullOffers";

describe("HeldPullOffers", () => {
  it("offers a held print exactly once while the same hold stands", () => {
    const offers = new HeldPullOffers();
    expect(offers.claim("job-a")).toBe(true);
    expect(offers.claim("job-a")).toBe(false);
    expect(offers.claim("job-b")).toBe(true);
  });

  it("forgets a print that is no longer held", () => {
    const offers = new HeldPullOffers();
    offers.claim("job-a");
    offers.claim("job-b");
    // `job-a` settled or resumed; `job-b` is still parked.
    offers.retain(["job-b"]);
    expect(offers.claim("job-b")).toBe(false);
    // A resumed print held again for the same missing model is a new offer:
    // the pull it was offered before plainly did not fix it.
    expect(offers.claim("job-a")).toBe(true);
  });

  it("forgets everything once nothing is held", () => {
    const offers = new HeldPullOffers();
    offers.claim("job-a");
    offers.retain([]);
    expect(offers.claim("job-a")).toBe(true);
  });
});
