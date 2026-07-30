import { ipc } from "./ipc";

export async function copyLocalOutputPath(filename: string): Promise<string> {
  const path = await ipc.localOutputFilePath(filename);
  if (!path) {
    throw new Error("This print is remote-only and has no local file path.");
  }
  await navigator.clipboard.writeText(path);
  return path;
}
