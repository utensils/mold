/** Desktop's local DB and older IPC rows predate the current HTTP contract.
 * These adapters describe that persisted input only. Shared wire declarations
 * stay strict; new fields flow through without a second hand-maintained copy. */
import type {
  OutputMetadata as ServerOutputMetadata,
  GalleryImage as ServerGalleryImage,
  GpuSnapshot as ServerGpuSnapshot,
  ResourceSnapshot as ServerResourceSnapshot,
} from "@studio/lib/api/serverTypes";

export interface OutputMetadata extends Omit<ServerOutputMetadata, "version" | "scheduler"> {
  version?: string | null;
  scheduler?: string | Record<string, unknown> | null;
  video_frames?: number | null;
  video_fps?: number | null;
}

export interface GalleryImage extends Omit<ServerGalleryImage, "metadata"> {
  metadata: OutputMetadata;
}

export interface GpuSnapshot extends Omit<
  ServerGpuSnapshot,
  "backend" | "vram_used_by_mold" | "vram_used_by_other"
> {
  backend: string;
  vram_used_by_mold?: number | null;
  vram_used_by_other?: number | null;
}

export interface ResourceSnapshot extends Omit<ServerResourceSnapshot, "gpus"> {
  gpus: GpuSnapshot[];
}
