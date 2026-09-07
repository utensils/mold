import type {
  GenerationImageReferenceRole,
  NamedViewsProfile,
} from "./generated/generationProfileV1";
import type { GenerationReference } from "./generationReferences";

export type NamedViewRole = GenerationImageReferenceRole;

export interface NamedViewImage {
  base64: string;
  filename: string;
  mimeType: string;
  width: number;
  height: number;
  /** Browser draft-media key; never sent to the server. */
  draftId?: string;
}

export type NamedViewsState = Partial<Record<NamedViewRole, NamedViewImage>>;

export const NAMED_VIEW_LABELS: Record<NamedViewRole, string> = {
  front: "Front",
  left: "Left",
  back: "Back",
  right: "Right",
};

export function emptyNamedViews(): NamedViewsState {
  return {};
}

export function setNamedView(
  state: NamedViewsState | null | undefined,
  role: NamedViewRole,
  image: NamedViewImage | null,
): NamedViewsState {
  const next = { ...(state ?? {}) };
  if (image) next[role] = image;
  else delete next[role];
  return next;
}

export function activeNamedViewsProfile(
  profile: NamedViewsProfile | null | undefined,
): NamedViewsProfile | null {
  return profile?.mode === "adjustable" ? profile : null;
}

export function namedViewValidationError(
  state: NamedViewsState | null | undefined,
  profile: NamedViewsProfile | null | undefined,
): string | null {
  const active = activeNamedViewsProfile(profile);
  if (!active) return null;
  const views = active.roles.flatMap((role) =>
    state?.[role] ? [[role, state[role]!] as const] : [],
  );
  if (views.length < active.min_count) {
    return `Add at least ${active.min_count === 1 ? "one named view" : `${active.min_count} named views`}.`;
  }
  if (views.length > active.max_count)
    return `Use no more than ${active.max_count} named views.`;
  for (const [role, image] of views) {
    if (
      !image.base64 ||
      image.width < 1 ||
      image.height < 1 ||
      !["image/png", "image/jpeg"].includes(image.mimeType)
    ) {
      return `${NAMED_VIEW_LABELS[role]} view could not be decoded as a PNG or JPEG image.`;
    }
  }
  return null;
}

export function serializeNamedViews(
  state: NamedViewsState | null | undefined,
  profile: NamedViewsProfile | null | undefined,
): GenerationReference[] {
  const active = activeNamedViewsProfile(profile);
  if (!active) return [];
  return active.roles.flatMap((role) => {
    const image = state?.[role];
    return image
      ? [
          {
            kind: "named_image" as const,
            role,
            media: { authority: "inline" as const, data: image.base64 },
            provenance: { name: image.filename },
            mime_type: image.mimeType,
            width: image.width,
            height: image.height,
          },
        ]
      : [];
  });
}
