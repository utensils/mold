import type { InjectionKey } from "vue";
import type { SourceMediaSelection } from "./sourceMediaBrowser";

export function sourceAddress(selection: SourceMediaSelection): string {
  return `source:${selection.owner.filename}#${selection.member.member_id}`;
}
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;

/** A non-secret selector is mandatory: member IDs belong to one host copy. */
export function sourceCommand(selection: SourceMediaSelection): string {
  let url: URL;
  try {
    url = new URL(selection.owner.target.baseUrl);
  } catch {
    throw new Error("The machine URL cannot be used in a CLI command.");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw new Error(
      "The machine URL contains private or unsupported information.",
    );
  return `mold --host ${quote(selection.owner.target.baseUrl)} library source-media ${quote(selection.owner.filename)} --member ${quote(selection.member.member_id)} --output ${quote(sourceFileName(selection))}`;
}

export function sourceFileName(selection: SourceMediaSelection): string {
  const name = selection.member.display_name
    .split(/[\\/]/)
    .pop()
    ?.replace(/[\u0000-\u001f\u007f]/g, "");
  // A keyframe's display name may name the JSON envelope, while /content is
  // the original image. Never offer that original as a JSON file.
  if (selection.member.role === "keyframes" && (!name || /\.json$/i.test(name)))
    return "keyframe.png";
  return name && name !== "." && name !== ".." ? name : "source-media";
}

export interface NativeSourceFileActions {
  copyImage(blob: Blob, current: () => boolean): Promise<void>;
  saveFile(blob: Blob, filename: string, current: () => boolean): Promise<void>;
}
export const NATIVE_SOURCE_FILE_ACTIONS: InjectionKey<NativeSourceFileActions> =
  Symbol("native-source-file-actions");
export function sourceMimeHint(selection: SourceMediaSelection): string | null {
  const suffix = selection.member.display_name.split(".").pop()?.toLowerCase();
  return (
    (
      {
        png: "image/png",
        jpg: "image/jpeg",
        jpeg: "image/jpeg",
        webp: "image/webp",
        gif: "image/gif",
        mp4: "video/mp4",
        mov: "video/quicktime",
        webm: "video/webm",
        wav: "audio/wav",
        mp3: "audio/mpeg",
      } as Record<string, string>
    )[suffix ?? ""] ?? null
  );
}
