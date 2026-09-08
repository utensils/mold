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
