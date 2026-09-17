import Foundation

/// The primitive's baseColor image, when it carries one and it is EMBEDDED.
///
/// Port of `readBaseColorTexture`, `studio/lib/glb.ts:463-510`. Only
/// `pbrMetallicRoughness.baseColorTexture` is read — the metallic-roughness,
/// normal and occlusion maps mold also writes are not part of this shading
/// model — and an image with a `uri` is never followed, because a second
/// network fetch is not something a mesh file gets to ask for.
enum GLBTexture {

    static func baseColor(in document: [String: Any], bin: [UInt8],
                          primitive: [String: Any]) throws -> MeshTexture? {
        guard let rawMaterial = GLBDocument.present(primitive["material"]),
              let materialIndex = GLBDocument.integer(rawMaterial),
              let materials = document["materials"] as? [Any]
        else { return nil }
        let material = try GLBDocument.objectAt(materials, materialIndex, "material")
        guard let pbr = material["pbrMetallicRoughness"] as? [String: Any],
              let reference = pbr["baseColorTexture"] as? [String: Any]
        else { return nil }

        let textureIndex = try GLBDocument.int(reference, "index", "baseColorTexture")
        let texture = try GLBDocument.objectAt(
            GLBDocument.array(document, "textures", "document"), textureIndex, "texture")
        let sourceIndex = try GLBDocument.int(texture, "source", "texture")
        let image = try GLBDocument.objectAt(
            GLBDocument.array(document, "images", "document"), sourceIndex, "image")

        let viewIndex = try GLBDocument.int(image, "bufferView", "image")
        let view = try GLBDocument.objectAt(
            GLBDocument.array(document, "bufferViews", "document"), viewIndex,
            "image bufferView")
        let offset = try GLBDocument.int(view, "byteOffset", "image bufferView", fallback: 0)
        let length = try GLBDocument.int(view, "byteLength", "image bufferView")
        // Two file-supplied numbers ADDED, each at most 2^53 by
        // `GLBDocument.integer`: at most 2^54, so no overflow to trap on.
        guard offset + length <= bin.count else {
            throw GLBParseError(
                "the baseColor image ends at byte \(offset + length), past the end of the "
                    + "\(bin.count)-byte BIN chunk")
        }
        let mimeType = (image["mimeType"] as? String) ?? "image/png"
        return MeshTexture(data: Data(bin[offset..<(offset + length)]), mimeType: mimeType)
    }
}
