import { watch } from "vue";
import { apiFetchTo, type ApiTarget } from "../api/client";
import {
  relayRetainedSourceMedia,
  retainedSourceMediaDisclosable,
  retainedSourceMediaDisclosure,
  type RetainedSourceMediaMetadataLike,
  type RetainedSourceMediaInventory,
  type RetainedSourceMediaMember,
} from "../api/gallerySourceMedia";
import { imageDimensionsFromBase64 } from "./imageDimensions";

interface Image {
  base64: string;
  filename: string;
}
interface Frame {
  frame: number;
  image: string;
  name?: string | null;
}
interface RestoredWire {
  source_image?: string;
  mask_image?: string;
  control_image?: string;
  edit_images?: string[];
  id_image?: string;
  id_images?: string[];
  keyframes?: Frame[];
  audio_file?: string;
  source_video?: string;
  extend_video?: string;
}
export interface RetainedDraftLayout {
  web: boolean;
  boundary: boolean;
  sourceMode: string;
  h3: boolean;
}

const legacyRoles = new Set([
  "source_image",
  "stage_source:0",
  "mask_image",
  "control_image",
  "edit_images",
  "identity_image",
  "identity_images",
  "keyframes",
  "audio_file",
  "audio_file_path",
  "source_video",
  "source_video_path",
  "extend_video",
  "extend_video_path",
]);
const twoWells = (mode: string) =>
  mode === "single-or-references" || mode === "single-and-references";

/** Pure UI projection: frame numbers and list order are wire authority. */
export function projectRetainedDraftMedia(
  wire: RestoredWire,
  members: readonly RetainedSourceMediaMember[],
  layout: RetainedDraftLayout,
): Record<string, unknown> {
  const patch: Record<string, unknown> = {};
  const name = (role: string) =>
    members.find((m) => m.role === role)?.display_name ?? role;
  const picked = (base64: string, filename: string) => ({
    ...(layout.web ? { kind: "upload" } : {}),
    base64,
    filename,
    ...imageDimensionsFromBase64(base64),
  });
  const source = (image: Image) => {
    if (layout.web)
      patch.imageAttachments = [picked(image.base64, image.filename)];
    else {
      patch.sourceImage = image.base64;
      patch.sourceImageName = image.filename;
      const pixels = imageDimensionsFromBase64(image.base64);
      patch.sourceImageWidth = pixels?.width ?? null;
      patch.sourceImageHeight = pixels?.height ?? null;
    }
  };
  if (wire.source_image)
    source({ base64: wire.source_image, filename: name("source_image") });
  if (wire.edit_images)
    patch[
      layout.web && twoWells(layout.sourceMode)
        ? "referenceImages"
        : "imageAttachments"
    ] = wire.edit_images.map((data, i) =>
      layout.web
        ? picked(
            data,
            members.filter((m) => m.role === "edit_images")[i]?.display_name ??
              "Reference",
          )
        : data,
    );
  if (wire.id_images && wire.id_images.length !== 1)
    throw new Error(
      "Reattach all identity photos in a client that supports multiple photos before generating.",
    );
  const identity = wire.id_image ?? wire.id_images?.[0];
  if (wire.id_image && wire.id_images)
    throw new Error("The retained identity photos are ambiguous.");
  if (identity)
    patch.identityImage = picked(
      identity,
      name(wire.id_image ? "identity_image" : "identity_images"),
    );
  for (const [wireField, formField, role] of [
    ["mask_image", "maskImage", "mask_image"],
    ["control_image", "controlImage", "control_image"],
    ["audio_file", "audioFile", "audio_file"],
    ["source_video", "sourceVideo", "source_video"],
    ["extend_video", "extendVideo", "extend_video"],
  ] as const) {
    const bytes = wire[wireField];
    if (bytes)
      patch[formField] =
        !layout.web &&
        (formField === "maskImage" || formField === "controlImage")
          ? bytes
          : picked(bytes, name(role));
  }
  if (
    wire.keyframes?.some(
      (frame) =>
        !Number.isSafeInteger(frame.frame) ||
        frame.frame < 0 ||
        typeof frame.image !== "string" ||
        !frame.image,
    )
  )
    throw new Error(
      "The retained keyframes are damaged. Reattach them before generating.",
    );
  if (wire.keyframes)
    patch.keyframes = wire.keyframes.map((frame) => ({
      frame: frame.frame,
      image: picked(frame.image, frame.name ?? "Keyframe"),
    }));
  if (layout.boundary && !layout.h3 && wire.keyframes) {
    if (wire.keyframes.length !== 2 || wire.keyframes[0]?.frame !== 0)
      throw new Error(
        "This client cannot restore this boundary-frame layout. Reattach the frames before generating.",
      );
    source({
      base64: wire.keyframes[0].image,
      filename: wire.keyframes[0].name ?? "First frame",
    });
    patch.endFrame = picked(
      wire.keyframes[1]!.image,
      wire.keyframes[1]!.name ?? "Last frame",
    );
    patch.keyframes = [];
  }
  if (layout.h3 && (wire.source_image || wire.keyframes)) {
    if (wire.keyframes && wire.keyframes.length !== 1)
      throw new Error("The retained closing frame is ambiguous.");
    const boundary = (image: Image) => ({
      filename: image.filename,
      data: image.base64,
      mimeType: "image/png",
      width: imageDimensionsFromBase64(image.base64)?.width ?? null,
      height: imageDimensionsFromBase64(image.base64)?.height ?? null,
    });
    patch.h3Authoring = {
      ...(wire.source_image
        ? {
            firstFrame: boundary({
              base64: wire.source_image,
              filename: name("source_image"),
            }),
          }
        : {}),
      ...(wire.keyframes?.[0]
        ? {
            lastFrame: boundary({
              base64: wire.keyframes[0].image,
              filename: wire.keyframes[0].name ?? "Last frame",
            }),
          }
        : {}),
    };
    delete patch.sourceImage;
    delete patch.sourceImageName;
    delete patch.imageAttachments;
    patch.keyframes = [];
  }
  return patch;
}

/** Atomic, bounded restore. Monotonic observation catches attach-then-remove,
 * while scalar edits remain live. Never replace a user's authored attachment. */
export async function restoreRetainedDraftMedia(options: {
  filename: string;
  origin: ApiTarget;
  inventory: RetainedSourceMediaInventory;
  metadata?: RetainedSourceMediaMetadataLike;
  read: () => object;
  layout: RetainedDraftLayout;
  isCurrent: () => boolean;
  signal?: AbortSignal;
  maxBytes?: number;
}): Promise<{
  patch: Record<string, unknown>;
  inventory: RetainedSourceMediaInventory;
} | null> {
  if (options.inventory.availability !== "available") {
    if (retainedSourceMediaDisclosable(options.metadata))
      throw new Error(
        retainedSourceMediaDisclosure(options.inventory.availability) ??
          "The original inputs could not be restored. Reuse this print again.",
      );
    return null;
  }
  const read = () => options.read() as Record<string, unknown>;
  const before = read();
  const filled = (value: unknown) =>
    typeof value === "string"
      ? value.length > 0
      : Array.isArray(value)
        ? value.length > 0
        : value != null && Boolean((value as { base64?: string }).base64);
  const occupied = (role: string) => {
    if (role === "source_image" || role === "stage_source:0")
      return options.layout.h3
        ? Boolean(
            (before.h3Authoring as { firstFrame?: { data?: string } })
              ?.firstFrame?.data,
          )
        : filled(
            options.layout.web ? before.imageAttachments : before.sourceImage,
          );
    if (role === "keyframes")
      return (
        filled(before.keyframes) ||
        (options.layout.boundary &&
          !options.layout.h3 &&
          filled(
            options.layout.web ? before.imageAttachments : before.sourceImage,
          )) ||
        filled(before.endFrame) ||
        Boolean(
          (before.h3Authoring as { lastFrame?: { data?: string } })?.lastFrame
            ?.data,
        )
      );
    if (role === "edit_images")
      return filled(
        options.layout.web && twoWells(options.layout.sourceMode)
          ? before.referenceImages
          : before.imageAttachments,
      );
    const field = (
      {
        identity_image: "identityImage",
        identity_images: "identityImage",
        mask_image: "maskImage",
        control_image: "controlImage",
        audio_file: "audioFile",
        audio_file_path: "audioFile",
        source_video: "sourceVideo",
        source_video_path: "sourceVideo",
        extend_video: "extendVideo",
        extend_video_path: "extendVideo",
      } as Record<string, string>
    )[role];
    return filled(before[field!]);
  };
  let members = options.inventory.members.filter(
    (m) => legacyRoles.has(m.role) && !occupied(m.role),
  );
  if (
    !members.some(
      (m) => m.role === "source_image" || m.role === "stage_source:0",
    )
  )
    members = members.filter((m) => m.role !== "mask_image");
  if (!members.length)
    return {
      patch: {},
      inventory: {
        ...options.inventory,
        members: options.inventory.members.filter(
          (m) => !legacyRoles.has(m.role),
        ),
      },
    };
  const bodySize = members.reduce(
    (n, m) => n + Math.ceil(m.size_bytes / 3) * 4,
    0,
  );
  if (
    !Number.isSafeInteger(bodySize) ||
    bodySize > (options.maxBytes ?? 64 * 1024 * 1024) ||
    members.some((m) => !Number.isSafeInteger(m.size_bytes) || m.size_bytes < 0)
  )
    throw new Error(
      "These inputs exceed the editor's restoration limit. Reattach smaller input files before generating.",
    );
  const origin = { ...options.origin };
  const signal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(15_000)])
    : AbortSignal.timeout(15_000);
  const roleValue = (role: string) => {
    const form = read();
    if (role === "source_image" || role === "stage_source:0")
      return options.layout.h3
        ? (form.h3Authoring as { firstFrame?: object })?.firstFrame
        : options.layout.web
          ? form.imageAttachments
          : form.sourceImage;
    if (role === "keyframes")
      return [
        form.keyframes,
        form.endFrame,
        options.layout.boundary && !options.layout.h3
          ? options.layout.web
            ? form.imageAttachments
            : form.sourceImage
          : null,
        (form.h3Authoring as { lastFrame?: object })?.lastFrame,
      ];
    if (role === "edit_images")
      return options.layout.web && twoWells(options.layout.sourceMode)
        ? form.referenceImages
        : form.imageAttachments;
    return form[
      (
        {
          identity_image: "identityImage",
          identity_images: "identityImage",
          mask_image: "maskImage",
          control_image: "controlImage",
          audio_file: "audioFile",
          audio_file_path: "audioFile",
          source_video: "sourceVideo",
          source_video_path: "sourceVideo",
          extend_video: "extendVideo",
          extend_video_path: "extendVideo",
        } as Record<string, string>
      )[role]!
    ];
  };
  const dirtyRoles = new Set<string>();
  const roles = [...new Set(members.map((member) => member.role))];
  const stop = watch(
    () => roles.map((role) => JSON.stringify(roleValue(role))),
    (next, previous) => {
      roles.forEach((role, index) => {
        if (next[index] !== previous[index]) dirtyRoles.add(role);
      });
    },
    { flush: "sync" },
  );
  const model = JSON.stringify([before.model, before.pipeline]);
  const instance = async () => {
    const response = await apiFetchTo(origin, "/api/status", { signal });
    const status = (await response.json()) as { instance_id?: string };
    return status.instance_id;
  };
  try {
    const initialInstance = await instance();
    const wire = await relayRetainedSourceMedia(
      options.filename,
      members,
      {},
      origin,
      signal,
    );
    if (initialInstance !== (await instance()))
      throw new Error(
        "The source machine restarted. Reuse this print again before generating.",
      );
    if (!options.isCurrent() || signal.aborted) return null;
    if (JSON.stringify([read().model, read().pipeline]) !== model) return null;
    if (dirtyRoles.has("source_image") || dirtyRoles.has("stage_source:0"))
      dirtyRoles.add("mask_image");
    const cleanWire = { ...wire } as Record<string, unknown>;
    const wireFields: Record<string, string> = {
      source_image: "source_image",
      "stage_source:0": "source_image",
      mask_image: "mask_image",
      control_image: "control_image",
      edit_images: "edit_images",
      identity_image: "id_image",
      identity_images: "id_images",
      keyframes: "keyframes",
      audio_file: "audio_file",
      audio_file_path: "audio_file",
      source_video: "source_video",
      source_video_path: "source_video",
      extend_video: "extend_video",
      extend_video_path: "extend_video",
    };
    for (const role of dirtyRoles) delete cleanWire[wireFields[role]!];
    const patch = projectRetainedDraftMedia(
      cleanWire,
      members.filter((member) => !dirtyRoles.has(member.role)),
      options.layout,
    );
    if (patch.h3Authoring)
      patch.h3Authoring = {
        ...(read().h3Authoring as object),
        ...(patch.h3Authoring as object),
      };
    return {
      patch,
      inventory: {
        ...options.inventory,
        members: options.inventory.members.filter(
          (m) => !legacyRoles.has(m.role),
        ),
      },
    };
  } finally {
    stop();
  }
}
