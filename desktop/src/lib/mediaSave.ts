import {
  downloadVideoExport,
  videoExportPath,
  type VideoExportOptions,
} from "@studio/lib/videoExport";
import { apiFetchTo, type ApiTarget } from "./api/client";
import { inTauri, ipc, type SavedMedia } from "./ipc";
import { useToastStore } from "../stores/toasts";

export async function saveGalleryMedia(
  target: ApiTarget | null,
  filename: string,
  outputFilename = filename,
  exportOptions: VideoExportOptions | null = null,
): Promise<SavedMedia> {
  if (inTauri()) {
    return ipc.saveGalleryMedia(
      target,
      filename,
      outputFilename,
      exportOptions as unknown as Record<string, unknown> | null,
    );
  }

  if (!target) throw new Error("The media host is no longer connected.");

  const response = await apiFetchTo(
    target,
    exportOptions
      ? videoExportPath(filename)
      : `/api/gallery/image/${encodeURIComponent(filename)}`,
    exportOptions
      ? {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(exportOptions),
        }
      : {},
  );
  downloadVideoExport(await response.blob(), outputFilename);
  return { filename: outputFilename, path: outputFilename, directory: "Downloads" };
}

export function showSavedMediaToast(
  toasts: ReturnType<typeof useToastStore>,
  saved: SavedMedia,
): void {
  toasts.push(`Saved ${saved.filename}`, "info", {
    description: `To ${saved.directory}`,
    ...(inTauri()
      ? {
          action: {
            label: "Show in folder",
            run: () => {
              void ipc
                .revealSavedMedia(saved.path)
                .catch((error) =>
                  toasts.push(error instanceof Error ? error.message : String(error), "error"),
                );
            },
          },
        }
      : {}),
    durationMs: 6000,
  });
}
