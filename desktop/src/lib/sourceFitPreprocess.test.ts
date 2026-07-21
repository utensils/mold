import { describe, expect, it, vi } from "vitest";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import type { Rect, SourceFitPolicy, SourceFitTransform } from "@studio/lib/sourceFit";
import {
  applySourceFitPreprocess,
  drawableFitPolicy,
  type SourceFitCanvasOps,
  type SourceFitInput,
} from "./sourceFitPreprocess";

/** Canvas ops fake: pure string transforms so assertions stay exact. */
function fakeOps(size: { width: number; height: number }): SourceFitCanvasOps & {
  fitCalls: Array<{ base64: string; transform: SourceFitTransform }>;
  maskCalls: Array<{ existing: string | null; padRects: Rect[] }>;
} {
  const fitCalls: Array<{ base64: string; transform: SourceFitTransform }> = [];
  const maskCalls: Array<{ existing: string | null; padRects: Rect[] }> = [];
  return {
    fitCalls,
    maskCalls,
    imageSize: () => Promise.resolve(size),
    fitImage: (base64, transform) => {
      fitCalls.push({ base64, transform });
      return Promise.resolve(`fit(${base64})`);
    },
    buildMask: (existing, _transform, padRects) => {
      maskCalls.push({ existing, padRects });
      return Promise.resolve(`mask(${existing ?? "none"})`);
    },
  };
}

const TARGET = { width: 1024, height: 1024 };

describe("drawableFitPolicy", () => {
  it("falls back to pad-repaint when unset and honors upscale-then-fit's inner fit", () => {
    expect(drawableFitPolicy(undefined)).toEqual({ mode: "pad-repaint" });
    // The inner fit drives the draw — required so maskless (video) img2img can
    // upscale-then-fit without pad bands it has no way to repaint. The UI's
    // default inner fit is still pad-repaint, so image flows are unchanged.
    expect(
      drawableFitPolicy({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x2plus:fp16",
        fit: { mode: "crop-fill" },
      }),
    ).toEqual({ mode: "crop-fill" });
    expect(drawableFitPolicy({ mode: "crop-fill" })).toEqual({ mode: "crop-fill" });
  });
});

describe("applySourceFitPreprocess", () => {
  it("is a no-op when there is no source image", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const result = await applySourceFitPreprocess(
      { source: null, mask: null, policy: { mode: "pad-repaint" }, target: TARGET },
      { ops },
    );
    expect(result).toEqual({ source: null, mask: null, changed: false });
    expect(ops.fitCalls).toHaveLength(0);
  });

  it("is a no-op when the source already matches the target dimensions", async () => {
    const ops = fakeOps({ width: 1024, height: 1024 });
    const result = await applySourceFitPreprocess(
      { source: "SRC", mask: "MASK", policy: { mode: "pad-repaint" }, target: TARGET },
      { ops },
    );
    expect(result).toEqual({ source: "SRC", mask: "MASK", changed: false });
    expect(ops.fitCalls).toHaveLength(0);
    expect(ops.maskCalls).toHaveLength(0);
  });

  it("pad-repaint fits the source and attaches a generated pad mask", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const result = await applySourceFitPreprocess(
      { source: "SRC", mask: null, policy: { mode: "pad-repaint" }, target: TARGET },
      { ops },
    );

    expect(result.changed).toBe(true);
    expect(result.source).toBe("fit(SRC)");
    expect(result.mask).toBe("mask(none)");
    expect(ops.fitCalls[0]).toMatchObject({
      base64: "SRC",
      transform: { drawWidth: 1024, drawHeight: 768, offsetY: 128, maskPaddedPixels: true },
    });
    // Top and bottom letterbox bands get repainted (white in the mask).
    expect(ops.maskCalls[0]!.padRects).toEqual([
      { x: 0, y: 0, width: 1024, height: 128 },
      { x: 0, y: 896, width: 1024, height: 128 },
    ]);
  });

  it("crop-fill leaves a missing mask absent but refits an existing mask", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const noMask = await applySourceFitPreprocess(
      { source: "SRC", mask: null, policy: { mode: "crop-fill" }, target: TARGET },
      { ops },
    );
    expect(noMask.mask).toBeNull();
    expect(ops.maskCalls).toHaveLength(0);

    const withMask = await applySourceFitPreprocess(
      { source: "SRC", mask: "MASK", policy: { mode: "crop-fill" }, target: TARGET },
      { ops },
    );
    // The user's painted mask must track the fitted source — no pad rects.
    expect(withMask.mask).toBe("mask(MASK)");
    expect(ops.maskCalls[0]).toMatchObject({ existing: "MASK", padRects: [] });
  });

  it("upscale-then-fit rewrites the source through the upscaler before fitting", async () => {
    const sizes = new Map([
      ["SRC", { width: 640, height: 480 }],
      ["up(SRC)", { width: 1280, height: 960 }],
    ]);
    const ops = fakeOps({ width: 0, height: 0 });
    ops.imageSize = (b64: string) => Promise.resolve(sizes.get(b64)!);
    const upscale = vi.fn((image: string, _model: string) => Promise.resolve(`up(${image})`));

    const result = await applySourceFitPreprocess(
      {
        source: "SRC",
        mask: null,
        policy: {
          mode: "upscale-then-fit",
          upscalerModel: "real-esrgan-x2plus:fp16",
          fit: { mode: "pad-repaint" },
        },
        target: TARGET,
      },
      { ops, upscale },
    );

    expect(upscale).toHaveBeenCalledWith("SRC", "real-esrgan-x2plus:fp16");
    // The FIT step consumed the upscaled bytes, not the original.
    expect(ops.fitCalls[0]!.base64).toBe("up(SRC)");
    expect(result.source).toBe("fit(up(SRC))");
    expect(result.changed).toBe(true);
    // drawableFitPolicy maps upscale-then-fit to pad-repaint (web parity),
    // so the pad mask is generated.
    expect(result.mask).toBe("mask(none)");
  });

  it("reuses cached upscale and fit results for unchanged inputs", async () => {
    const sizes = new Map([
      ["SRC", { width: 640, height: 480 }],
      ["up(SRC)", { width: 1280, height: 960 }],
    ]);
    const ops = fakeOps({ width: 0, height: 0 });
    ops.imageSize = (b64: string) => Promise.resolve(sizes.get(b64)!);
    const upscale = vi.fn((image: string) => Promise.resolve(`up(${image})`));
    const cache = new SourceFitPreprocessCache();
    const input: SourceFitInput = {
      source: "SRC",
      mask: null,
      policy: {
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x2plus:fp16",
        fit: { mode: "pad-repaint" },
      },
      target: TARGET,
    };

    const first = await applySourceFitPreprocess(input, { ops, upscale, cache });
    const second = await applySourceFitPreprocess(input, { ops, upscale, cache });

    expect(second).toEqual(first);
    expect(upscale).toHaveBeenCalledTimes(1);
    expect(ops.fitCalls).toHaveLength(1);
    expect(ops.maskCalls).toHaveLength(1);
  });

  it("keeps the upscale cache across target changes but invalidates the fit", async () => {
    const sizes = new Map([
      ["SRC", { width: 640, height: 480 }],
      ["up(SRC)", { width: 1280, height: 960 }],
    ]);
    const ops = fakeOps({ width: 0, height: 0 });
    ops.imageSize = (b64: string) => Promise.resolve(sizes.get(b64)!);
    const upscale = vi.fn((image: string) => Promise.resolve(`up(${image})`));
    const cache = new SourceFitPreprocessCache();
    const policy: SourceFitPolicy = {
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x2plus:fp16",
      fit: { mode: "crop-fill" },
    };

    await applySourceFitPreprocess(
      { source: "SRC", mask: null, policy, target: TARGET },
      { ops, upscale, cache },
    );
    await applySourceFitPreprocess(
      { source: "SRC", mask: null, policy, target: { width: 768, height: 768 } },
      { ops, upscale, cache },
    );

    expect(upscale).toHaveBeenCalledTimes(1);
    expect(ops.fitCalls).toHaveLength(2);
  });

  it("invalidates cache layers for source, model, fit policy, and mask changes", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const upscale = vi.fn((image: string, model: string) =>
      Promise.resolve(`up(${model}:${image})`),
    );
    ops.imageSize = () => Promise.resolve({ width: 640, height: 480 });
    const cache = new SourceFitPreprocessCache();
    const run = (
      source: string,
      model: string,
      fit: Exclude<SourceFitPolicy, { mode: "upscale-then-fit" }>,
      mask: string | null,
    ) =>
      applySourceFitPreprocess(
        {
          source,
          mask,
          policy: { mode: "upscale-then-fit", upscalerModel: model, fit },
          target: TARGET,
        },
        { ops, upscale, cache },
      );

    await run("SRC", "up:a", { mode: "crop-fill" }, null);
    await run("SRC", "up:a", { mode: "pad-fit" }, null);
    expect(upscale).toHaveBeenCalledTimes(1);
    expect(ops.fitCalls).toHaveLength(2);

    await run("SRC", "up:a", { mode: "pad-fit" }, "MASK");
    expect(upscale).toHaveBeenCalledTimes(1);
    expect(ops.fitCalls).toHaveLength(3);

    await run("SRC", "up:b", { mode: "pad-fit" }, "MASK");
    await run("OTHER", "up:b", { mode: "pad-fit" }, "MASK");
    expect(upscale).toHaveBeenCalledTimes(3);
  });

  it("shares an in-flight cache entry between concurrent batch siblings", async () => {
    const ops = fakeOps({ width: 1024, height: 1024 });
    let finish!: (value: string) => void;
    const upscale = vi.fn(() => new Promise<string>((resolve) => (finish = resolve)));
    const cache = new SourceFitPreprocessCache();
    const input: SourceFitInput = {
      source: "SRC",
      mask: null,
      policy: {
        mode: "upscale-then-fit",
        upscalerModel: "up:model",
        fit: { mode: "crop-fill" },
      },
      target: TARGET,
    };

    const first = applySourceFitPreprocess(input, { ops, upscale, cache });
    const second = applySourceFitPreprocess(input, { ops, upscale, cache });
    await vi.waitFor(() => expect(upscale).toHaveBeenCalledTimes(1));
    finish("UPSCALED");
    await Promise.all([first, second]);

    expect(upscale).toHaveBeenCalledTimes(1);
  });

  it("upscale-then-fit without an upscaler model skips the upscale but still fits", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const upscale = vi.fn();
    const result = await applySourceFitPreprocess(
      {
        source: "SRC",
        mask: null,
        policy: { mode: "upscale-then-fit", upscalerModel: "", fit: { mode: "pad-repaint" } },
        target: TARGET,
      },
      { ops, upscale },
    );
    expect(upscale).not.toHaveBeenCalled();
    expect(result.source).toBe("fit(SRC)");
  });

  it("propagates upscaler failures without touching the form fields", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const upscale = vi.fn(() => Promise.reject(new Error("no such model")));
    await expect(
      applySourceFitPreprocess(
        {
          source: "SRC",
          mask: null,
          policy: {
            mode: "upscale-then-fit",
            upscalerModel: "real-esrgan-x2plus:fp16",
            fit: { mode: "pad-repaint" },
          },
          target: TARGET,
        },
        { ops, upscale },
      ),
    ).rejects.toThrow("no such model");
    expect(ops.fitCalls).toHaveLength(0);
  });

  it("reports progress for the upscale stage", async () => {
    const ops = fakeOps({ width: 1024, height: 1024 });
    const statuses: string[] = [];
    await applySourceFitPreprocess(
      {
        source: "SRC",
        mask: null,
        policy: {
          mode: "upscale-then-fit",
          upscalerModel: "real-esrgan-x2plus:fp16",
          fit: { mode: "pad-repaint" },
        },
        target: TARGET,
      },
      {
        ops,
        upscale: (image) => Promise.resolve(image),
        onStatus: (msg) => statuses.push(msg),
      },
    );
    expect(statuses.some((s) => s.includes("real-esrgan-x2plus:fp16"))).toBe(true);
  });
});
