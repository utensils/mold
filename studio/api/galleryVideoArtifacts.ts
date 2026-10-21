import { apiFetchTo } from "./client";
import { bounded, MAX_JSON } from "./galleryCopyWire";
import type { CopyMachine } from "./galleryCopyTypes";

const CHUNK = 4 * 1024 * 1024;
const DIGEST = /^[a-f0-9]{64}$/i;
const U64 = 18446744073709551615n;
interface ArtifactFile {
  sha256: string;
  size_bytes: string;
}
export interface GalleryVideoArtifacts {
  version: number;
  bundle_id: string;
  provenance: ArtifactFile;
  pcm?: {
    file: ArtifactFile;
    sample_rate: number;
    channels: number;
    frames: string;
  };
}
type Role = "provenance" | "pcm";
function decimal(value: unknown, positive = false): bigint {
  if (typeof value !== "string" || !/^(0|[1-9][0-9]*)$/.test(value))
    throw new Error("The machine returned an invalid saved video size.");
  const n = BigInt(value);
  if (n > U64 || (positive && n === 0n))
    throw new Error("The machine returned an invalid saved video size.");
  return n;
}
/** A declared invalid descriptor is a refusal, never an output-only fallback. */
export function declaredVideoArtifacts(
  metadata: object,
): GalleryVideoArtifacts | undefined {
  const job = (metadata as { video_job?: { artifacts?: unknown } }).video_job;
  if (!job || !Object.hasOwn(job, "artifacts")) return;
  const a = job.artifacts as GalleryVideoArtifacts;
  const file = (f: ArtifactFile) => {
    if (!f || !DIGEST.test(f.sha256))
      throw new Error("The machine returned an invalid saved video checksum.");
    return decimal(f.size_bytes, true);
  };
  if (!a || a.version !== 1 || !DIGEST.test(a.bundle_id))
    throw new Error("This video's saved data could not be verified.");
  file(a.provenance);
  if (a.pcm !== undefined) {
    const pcm = a.pcm;
    if (
      !pcm ||
      !Number.isInteger(pcm.sample_rate) ||
      pcm.sample_rate < 1 ||
      pcm.sample_rate > 192000 ||
      ![1, 2].includes(pcm.channels) ||
      file(pcm.file) !== decimal(pcm.frames, true) * BigInt(pcm.channels) * 4n
    )
      throw new Error("This video's saved audio data could not be verified.");
  }
  return a;
}
export async function requireVideoArtifactTransfer(
  source: CopyMachine,
  destination: CopyMachine,
  check: () => void,
  signal?: AbortSignal,
): Promise<void> {
  for (const machine of [source, destination]) {
    check();
    const response = await apiFetchTo(machine.target, "/api/capabilities", {
      signal: signal ?? null,
    });
    const value = JSON.parse(
      new TextDecoder().decode(await bounded(response, MAX_JSON, signal)),
    );
    check();
    if (value.retained_media_transfer?.video_artifacts !== true)
      throw new Error(
        `Update ${machine.label} before copying this video's saved audio and clip history. The copy would be incomplete.`,
      );
  }
}
interface TransferOptions {
  source: CopyMachine;
  destination: CopyMachine;
  sourceFilename: string;
  destinationFilename: string;
  sourceArchive: string;
  destinationArchive: string;
  artifacts: GalleryVideoArtifacts;
  check(): void;
  signal?: AbortSignal;
}
/** Server-owned resumable staging keeps memory bounded independently of movie duration. */
export async function transferGalleryVideoArtifacts(
  o: TransferOptions,
): Promise<void> {
  const roles: [Role, ArtifactFile][] = [
    ["provenance", o.artifacts.provenance],
  ];
  if (o.artifacts.pcm) roles.push(["pcm", o.artifacts.pcm.file]);
  const base = `/api/gallery/video-artifacts/${encodeURIComponent(o.destinationFilename)}/transfer`;
  const headers = {
    "x-mold-destination-instance": o.destination.instanceId,
    "x-mold-gallery-archive-identity": o.destinationArchive,
  };
  let handle: string | undefined;
  const request = async (path: string, method = "GET", body?: BodyInit) => {
    o.check();
    const response = await apiFetchTo(o.destination.target, path, {
      method,
      headers: {
        ...headers,
        ...(method === "POST" && body
          ? { "Content-Type": "application/json" }
          : method === "PUT"
            ? { "Content-Type": "application/octet-stream" }
            : {}),
      },
      ...(body ? { body } : {}),
      signal: o.signal ?? null,
    });
    const status = JSON.parse(
      new TextDecoder().decode(await bounded(response, MAX_JSON, o.signal)),
    );
    o.check();
    if (
      !/^[a-f0-9]{64}$/.test(status.handle) ||
      (handle && handle !== status.handle) ||
      status.archive_identity_sha256 !== o.destinationArchive
    )
      throw new Error(
        "The destination copy changed. Refresh the Library before copying again.",
      );
    if (
      typeof status.complete !== "boolean" ||
      !status.offsets ||
      typeof status.offsets !== "object"
    )
      throw new Error(
        "The destination could not verify progress for this video's saved data.",
      );
    if (
      Object.keys(status.offsets).some(
        (role) => !roles.some(([expected]) => expected === role),
      )
    )
      throw new Error("The destination returned unexpected saved video data.");
    for (const [role, file] of roles) {
      const offset = decimal(status.offsets[role]);
      if (
        offset > decimal(file.size_bytes) ||
        (status.complete && offset !== decimal(file.size_bytes))
      )
        throw new Error("The destination's saved video data is incomplete.");
    }
    handle = status.handle as string;
    return status as {
      handle: string;
      complete: boolean;
      offsets: Record<Role, string>;
    };
  };
  let status = await request(
    base,
    "POST",
    JSON.stringify({
      archive_identity_sha256: o.destinationArchive,
      artifacts: o.artifacts,
    }),
  );
  status = await request(`${base}/${handle}`);
  if (status.complete) return;
  for (const [role, file] of roles) {
    const size = decimal(file.size_bytes);
    let offset = decimal(status.offsets[role]);
    while (offset < size) {
      o.check();
      const response = await apiFetchTo(
        o.source.target,
        `/api/gallery/video-artifacts/${encodeURIComponent(o.sourceFilename)}/${role}?offset_bytes=${offset}`,
        {
          headers: {
            "x-mold-source-instance": o.source.instanceId,
            "x-mold-gallery-archive-identity": o.sourceArchive,
          },
          signal: o.signal ?? null,
        },
      );
      const length = Number(
        size - offset > BigInt(CHUNK) ? BigInt(CHUNK) : size - offset,
      );
      let chunk: Uint8Array<ArrayBuffer>;
      try {
        chunk = await bounded(response, length, o.signal);
      } catch (failure) {
        if (
          failure instanceof Error &&
          failure.message.startsWith("Copy is limited")
        )
          throw new Error(
            "The original machine sent more saved video data than expected.",
          );
        throw failure;
      }
      o.check();
      if (chunk.length !== length)
        throw new Error(
          "The original video's saved data changed during copying.",
        );
      const next = offset + BigInt(chunk.length);
      const previous = status.offsets;
      status = await request(
        `${base}/${handle}/${role}?offset_bytes=${offset}`,
        "PUT",
        chunk,
      );
      if (
        roles.some(
          ([other]) =>
            other !== role && status.offsets[other] !== previous[other],
        )
      )
        throw new Error(
          "Saved video progress changed unexpectedly. Copy again to resume.",
        );
      if (decimal(status.offsets[role]) !== next)
        throw new Error(
          "Saved video data progress was not confirmed. Copy again to resume.",
        );
      offset = next;
    }
  }
  status = await request(`${base}/${handle}/complete`, "POST");
  if (!status.complete)
    throw new Error(
      "Saving all video data was not confirmed. Copy again to repair this video.",
    );
}
