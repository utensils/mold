import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  generationHostSubmissionPolicy,
  type GenerationSubmissionHost,
} from "./generationSubmissionPolicy";

function canonicalHost(
  hostId: string,
  overrides: Partial<GenerationSubmissionHost> = {},
): GenerationSubmissionHost {
  return {
    hostId,
    queue: { heterogeneous_batch_max_outputs: 64 },
    durableMedia: {
      protocol_version: 2,
      encrypted_at_rest: true,
      generate_request_media: true,
      identity: true,
      private_h3: true,
    },
    ...overrides,
  };
}

describe("generation submission policy", () => {
  it("sends every pinned print directly to canonical durable admission", () => {
    expect(
      generationHostSubmissionPolicy(
        { kind: "pinned", hostId: "hal" },
        canonicalHost("hal"),
      ),
    ).toEqual({
      routing: "none",
      admission: "canonical_durable",
      refusal: null,
    });
  });

  it("fans automatic routing through cache-only probes", () => {
    for (const target of [{ kind: "auto" }, { kind: "capable" }] as const) {
      expect(
        generationHostSubmissionPolicy(target, canonicalHost("hal")),
      ).toMatchObject({
        routing: "telemetry_only",
        admission: "canonical_durable",
      });
    }
  });

  /**
   * The durable protocol carries every request trait, so a client-side
   * per-trait fence could only ever refuse work the server would have taken.
   * The server's typed admission refusal is the single authority.
   */
  it("never inspects the request — the decision is the machine's contract", () => {
    // `import.meta.url` is not a file URL in every environment these tests
    // run in, and the vitest root differs between the studio, web, and
    // desktop configs, so the module's own location is the anchor.
    const relative = "studio/lib/generationSubmissionPolicy.ts";
    let directory = process.cwd();
    let modulePath = resolve(directory, relative);
    while (!existsSync(modulePath)) {
      const parent = dirname(directory);
      if (parent === directory) throw new Error(`could not find ${relative}`);
      directory = parent;
      modulePath = resolve(directory, relative);
    }
    const source = readFileSync(modulePath, "utf8");
    // The policy may not reach for the request's shape at all.
    expect(source).not.toContain('from "./generationMedia"');
    expect(source).not.toContain('from "./minimaxH3Identity"');
    expect(source).not.toMatch(/^\s*request[?:]/m);
  });

  it("refuses a machine that does not speak the contract, by name", () => {
    const cases: Array<[Partial<GenerationSubmissionHost>, string]> = [
      [
        { queue: null },
        "this machine does not advertise the durable generation queue",
      ],
      [
        { queue: {} },
        "this machine does not advertise the durable generation queue",
      ],
      [
        { queue: { heterogeneous_batch_max_outputs: 0 } },
        "this machine does not advertise the durable generation queue",
      ],
    ];
    for (const [overrides, refusal] of cases) {
      expect(
        generationHostSubmissionPolicy(
          { kind: "pinned", hostId: "hal" },
          canonicalHost("hal", overrides),
        ),
      ).toEqual({ routing: "none", admission: "refused", refusal });
    }
  });

  it("keeps sequences on the chain-job route with its placement preview", () => {
    expect(
      generationHostSubmissionPolicy(
        { kind: "pinned", hostId: "hal" },
        canonicalHost("hal"),
        "sequence",
      ),
    ).toMatchObject({
      routing: "placement_preview",
      admission: "refused",
    });
  });
});
