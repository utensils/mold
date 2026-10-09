/** Old-server fallback. Wording is pinned against Rust and Swift fixtures. */
export function readableBytes(value: number): string {
  for (const [scale, unit] of [
    [1e12, "TB"],
    [1e9, "GB"],
    [1e6, "MB"],
    [1e3, "KB"],
  ] as const) {
    if (value >= scale) return `${Number((value / scale).toFixed(2))} ${unit}`;
  }
  return `${value} B`;
}

export function userFacingError(diagnostic: string): string {
  const raw = diagnostic.trim();
  const lower = raw.toLowerCase();
  if (
    lower.includes("cuda") &&
    (lower.includes("restart") || lower.includes("quarantined"))
  )
    return "The graphics device needs to recover. Restart Mold on that machine before trying again.";
  const legacyMemory = legacyMemorySummary(lower);
  if (legacyMemory) return legacyMemory;
  if (lower.includes("cuda") && lower.includes("cooldown"))
    return "The machine is recovering from a memory failure. Wait for the cooldown or try a smaller output size.";
  if (lower.includes("requires cuda compute capability"))
    return "This graphics device cannot run that model. Choose another machine or model.";
  if (lower.includes("lora adapter is no longer readable"))
    return "A LoRA file is missing or unreadable. Restore it on the machine, then retry the job.";
  if (
    lower.includes("ffprobe is required") ||
    lower.includes("ffmpeg is required")
  )
    return "Video processing tools are missing on the machine. Install ffmpeg and ffprobe, then try again.";
  if (
    lower.includes("admission sample") &&
    /(?:device|host|unified-memory) bytes/.test(lower)
  ) {
    const required = Number(/needs at least ([0-9]+)\b/.exec(lower)?.[1]);
    const available = Number(/exceeding the ([0-9]+)\b/.exec(lower)?.[1]);
    if (
      Number.isSafeInteger(required) &&
      Number.isSafeInteger(available) &&
      required > available
    ) {
      const resource = lower.includes("device bytes")
        ? "graphics"
        : lower.includes("unified-memory bytes")
          ? "shared"
          : "system";
      return `Not enough ${resource} memory. Estimated need: ${readableBytes(required)}; available budget: ${readableBytes(available)} (${readableBytes(required - available)} short). Close apps on that machine or try a smaller model or output size.`;
    }
  }
  if (
    [
      "out of memory",
      "out_of_memory",
      "insufficient memory",
      "allocation failed",
      "memory allocation",
      "cuda_error_oom",
    ].some((s) => lower.includes(s))
  ) {
    return "The machine ran out of memory. Try a smaller model, output size or batch.";
  }
  if (lower.includes("no space left on device"))
    return "The machine is out of storage. Free some disk space and try again.";
  if (lower.includes("no such file or directory"))
    return "A required file is missing on the machine. Restore or download the model again.";
  if (lower.includes("permission denied"))
    return "The machine cannot access a required file. Check its file permissions.";
  if (lower.includes("no device could produce an execution plan"))
    return "This machine cannot run those settings. Try another machine, model or output size.";
  if (
    [
      "tensor shape",
      "tensor mismatch",
      "tensor(",
      "dtype",
      "shape mismatch",
    ].some((s) => lower.includes(s))
  )
    return "The model’s data did not match what the renderer expected. Try another model or report this job’s failure details.";
  if (lower.includes("safetensors error"))
    return "The model file could not be read. Download that model again on the machine, then retry.";
  if (
    lower.includes(
      "failed to authenticate the reviewed minimax h3 turbo adapter",
    )
  )
    return "The H3 Turbo model file could not be verified. Check its installation on that machine, then retry or move the job.";
  if (lower.includes("minimax h3 preparation evidence was rejected"))
    return "The model could not be prepared on this machine. Check its installation or move the job to another machine.";
  if (lower.includes("cuda") || lower.includes("metal error"))
    return "The graphics device could not finish the render. Retry the job or move it to another machine.";
  if (lower.includes("backtrace") || lower.includes("panic"))
    return "Mold encountered an internal error while handling the request.";
  if (!raw || raw.includes("\n") || [...raw].length > 240)
    return "Mold encountered an unexpected error and could not complete the request.";
  return raw.replace(
    /(?<![\w.])([0-9]+) bytes?\b/g,
    (token, digits: string) => {
      const value = Number(digits);
      return Number.isSafeInteger(value) ? readableBytes(value) : token;
    },
  );
}

function scaledPrefix(raw: string): number | null {
  const match = /^\s*~?([0-9]+(?:\.[0-9]+)?)\s+(tb|gb|mb|kb|bytes?)\b/.exec(
    raw,
  );
  if (!match) return null;
  const scales: Record<string, number> = {
    tb: 1e12,
    gb: 1e9,
    mb: 1e6,
    kb: 1e3,
    bytes: 1,
    byte: 1,
  };
  const value = Math.round(Number(match[1]) * scales[match[2]!]!);
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function scaledAfter(raw: string, marker: string): number | null {
  const index = raw.indexOf(marker);
  return index < 0 ? null : scaledPrefix(raw.slice(index + marker.length));
}

function legacyMemorySummary(raw: string): string | null {
  if (
    ![
      "needs more host memory",
      "needs more device memory",
      "effective vram capacity",
    ].some((s) => raw.includes(s))
  )
    return null;
  const required = scaledAfter(raw, "requires ") ?? scaledAfter(raw, "needs ~");
  let available =
    scaledAfter(raw, "over this request's ~") ?? scaledAfter(raw, "over the ");
  if (available === null && raw.includes("requires ")) {
    const comma = raw.indexOf(",", raw.indexOf("requires "));
    if (comma >= 0) available = scaledPrefix(raw.slice(comma + 1));
  }
  if (required === null || available === null || required <= available)
    return null;
  const resource = raw.includes("host memory")
    ? "system"
    : raw.includes("metal:") || raw.includes("unified-memory")
      ? "shared"
      : "graphics";
  const advice = raw.includes("cooldown")
    ? "Wait for the cooldown or try a smaller output size."
    : "Close apps on that machine or try a smaller model or output size.";
  return `Not enough ${resource} memory. Estimated need: ${readableBytes(required)}; available budget: ${readableBytes(available)} (${readableBytes(required - available)} short). ${advice}`;
}
