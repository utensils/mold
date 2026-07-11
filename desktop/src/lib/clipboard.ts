import { apiFetch } from "./api/client";
import { inTauri } from "./ipc";

interface CopyImageDeps {
  fetchImage: (path: string) => Promise<Uint8Array>;
  native: boolean;
  fromBytes?: (bytes: Uint8Array) => Promise<unknown>;
  writeImage?: (image: unknown) => Promise<void>;
  browserWrite?: (items: ClipboardItem[]) => Promise<void>;
  mimeType?: string;
}

async function fetchImage(path: string): Promise<Uint8Array> {
  const response = path.startsWith("mold-local:") ? await fetch(path) : await apiFetch(path);
  return new Uint8Array(await response.arrayBuffer());
}

export async function copyImageBytesToClipboard(
  path: string,
  provided?: Partial<CopyImageDeps>,
): Promise<void> {
  const native = provided?.native ?? inTauri();
  const bytes = await (provided?.fetchImage ?? fetchImage)(path);
  if (native) {
    const fromBytes =
      provided?.fromBytes ??
      (async (data: Uint8Array) => {
        const { Image } = await import("@tauri-apps/api/image");
        return Image.fromBytes(data);
      });
    const writeImage =
      provided?.writeImage ??
      (async (image: unknown) => {
        const clipboard = await import("@tauri-apps/plugin-clipboard-manager");
        await clipboard.writeImage(image as Parameters<typeof clipboard.writeImage>[0]);
      });
    const image = await fromBytes(bytes);
    try {
      await writeImage(image);
    } finally {
      if (
        typeof image === "object" &&
        image !== null &&
        "close" in image &&
        typeof image.close === "function"
      ) {
        await image.close();
      }
    }
    return;
  }

  const type = provided?.mimeType ?? "image/png";
  const blob = new Blob([new Uint8Array(bytes).buffer], { type });
  const write =
    provided?.browserWrite ?? ((items: ClipboardItem[]) => navigator.clipboard.write(items));
  await write([new ClipboardItem({ [type]: blob })]);
}

export function copyBase64ImageToClipboard(image: string, mimeType: string): Promise<void> {
  const bytes = Uint8Array.from(atob(image), (character) => character.charCodeAt(0));
  return copyImageBytesToClipboard("", {
    fetchImage: async () => bytes,
    mimeType,
  });
}
