import { apiFetchTo, type ApiTarget } from "./client";

/** One independently downloadable file carried by a durable gallery print. */
export interface GenerationAsset {
  asset_id: string;
  role: string;
  display_name: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
  width?: number | null;
  height?: number | null;
}

export function generationAssetPath(filename: string, assetId: string): string {
  return `/api/gallery/assets/${encodeURIComponent(filename)}/${encodeURIComponent(assetId)}`;
}

export async function generationAssetBlob(
  target: ApiTarget,
  filename: string,
  assetId: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await apiFetchTo(
    target,
    generationAssetPath(filename, assetId),
    signal ? { signal } : {},
  );
  if (!response.ok) {
    throw new Error(
      `Could not download generation asset (HTTP ${response.status})`,
    );
  }
  return response.blob();
}

export function generationAssetLabel(asset: GenerationAsset): string {
  switch (asset.role) {
    case "base_color":
      return "Download base color map";
    case "metallic_roughness":
      return "Download metallic-roughness map";
    case "normal":
      return "Download normal map";
    default:
      return `Download ${asset.display_name}`;
  }
}
