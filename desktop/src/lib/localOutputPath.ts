import { ipc } from "./ipc";

export async function copyLocalOutputPath(filename: string, fromTrash = false): Promise<string> {
  const path = await ipc.localOutputFilePath(filename, fromTrash);
  if (!path) {
    throw new Error("This print is not available as a local file.");
  }
  await navigator.clipboard.writeText(path);
  return path;
}
