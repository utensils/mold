import { apiFetch } from "./api/client";
import { inTauri, ipc } from "./ipc";

interface CopyImageDeps {
  fetchImage: (path: string) => Promise<Uint8Array>;
  native: boolean;
  nativeWrite?: (bytes: Uint8Array) => Promise<void>;
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
    const nativeWrite = provided?.nativeWrite ?? (async (data) => ipc.clipboardWriteImage(data));
    await nativeWrite(bytes);
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
