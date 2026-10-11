import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, expectTypeOf, it } from "vitest";
import type {
  OutputMetadata,
  ModelComponentStatus,
  ServerStatus,
} from "./serverTypes";

describe("shared server wire contracts", () => {
  it("retains additive preprocessing and prefix-cache provenance together", () => {
    const metadata: OutputMetadata = {
      prompt: "fixture",
      model: "fixture",
      seed: 1,
      steps: 1,
      guidance: 1,
      width: 512,
      height: 512,
      version: "0.1.0",
      source_preprocessing: {
        profile: { generation: "v2_3", image_crf: 33 },
        codec: "jpeg",
        fit_policy: "contain",
      },
      prefix_cache: "retained",
    };
    expect(metadata.source_preprocessing?.profile.generation).toBe("v2_3");
    expect(metadata.prefix_cache).toBe("retained");
    expectTypeOf<OutputMetadata["version"]>().toEqualTypeOf<string>();
    // Synthesized desktop aliases belong to its adapter, never server JSON.
    expectTypeOf<"video_frames">().not.toMatchTypeOf<keyof OutputMetadata>();
  });

  it("keeps older optional telemetry and repair choices readable", () => {
    const status: ServerStatus = {
      version: "old",
      models_loaded: [],
      uptime_secs: 1,
    };
    const component: ModelComponentStatus = {
      kind: "vae",
      name: "vae",
      present: false,
    };
    expect(status.host_memory).toBeUndefined();
    expect(component.options).toBeUndefined();
  });

  it("ties the fields that drifted to their Rust wire authority", () => {
    let root = process.cwd();
    while (!existsSync(resolve(root, "crates/mold-core/src/types.rs"))) {
      const parent = dirname(root);
      if (parent === root) throw new Error("repository not found");
      root = parent;
    }
    const rust = readFileSync(
      resolve(root, "crates/mold-core/src/types.rs"),
      "utf8",
    );
    const metadata = rust.slice(
      rust.indexOf("pub struct OutputMetadata {"),
      rust.indexOf("impl OutputMetadata"),
    );
    for (const name of [
      "source_preprocessing",
      "prefix_cache",
      "chain_job_id",
      "output_mode",
      "version",
    ]) {
      expect(metadata).toContain(`pub ${name}:`);
    }
    expect(metadata).toContain("pub version: String");
    expect(metadata).not.toContain("pub video_frames:");
  });
});
