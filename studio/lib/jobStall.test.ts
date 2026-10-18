import { describe, expect, it } from "vitest";
import {
  parseJobStall,
  jobStallLabel,
  parseQueueCancellationResponse,
} from "./jobStall";
const diagnostic = {
  stage: "VAE decode",
  last_completed_work_unix_ms: 1700000000000,
  elapsed_stall_ms: 90000,
  cancel_requested: false,
  device_still_held: true,
};
describe("held worker diagnostics", () => {
  it("retains additive stall evidence without inventing a terminal job state", () => {
    expect(parseJobStall({ ...diagnostic, future_field: 1 })).toEqual(
      diagnostic,
    );
    expect(jobStallLabel(diagnostic)).toBe("Stalled · VAE decode");
    expect(jobStallLabel({ ...diagnostic, cancel_requested: true })).toBe(
      "Cancel requested; device still held",
    );
  });
  it.each([
    null,
    {},
    { ...diagnostic, cancel_requested: "false" },
    { ...diagnostic, elapsed_stall_ms: -1 },
    { ...diagnostic, last_completed_work_unix_ms: Infinity },
  ])("does not infer a stall from malformed or absent evidence", (value) => {
    expect(parseJobStall(value)).toBeNull();
  });
  it("keeps an accepted cancellation live until the server confirms a terminal outcome", async () => {
    expect(
      await parseQueueCancellationResponse(
        Response.json(
          {
            cancel_requested: true,
            device_still_held: true,
            stall: { ...diagnostic, cancel_requested: true },
          },
          { status: 202 },
        ),
      ),
    ).toEqual({
      terminal: false,
      deviceStillHeld: true,
      stall: { ...diagnostic, cancel_requested: true },
    });
    expect(
      await parseQueueCancellationResponse(new Response(null, { status: 202 })),
    ).toEqual({ terminal: false, deviceStillHeld: false, stall: null });
    expect(
      await parseQueueCancellationResponse(new Response(null, { status: 204 })),
    ).toEqual({ terminal: true, deviceStillHeld: false, stall: null });
  });
});
