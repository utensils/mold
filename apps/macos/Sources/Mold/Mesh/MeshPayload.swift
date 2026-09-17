import CoreGraphics
import Foundation
import ImageIO
import MoldClient
import UniformTypeIdentifiers

/// A baseColor texture, decoded to bytes a GPU can take verbatim.
///
/// Decoded OFF the main actor with the mesh, because a 2048² PNG is the
/// expensive half of opening a textured print and `MTLDevice.makeTexture` from
/// ready bytes is a copy.
struct MeshTextureImage: Sendable {
    let width: Int
    let height: Int
    /// Premultiplied-last RGBA8, `width * 4` bytes a row.
    let rgba: Data
}

/// One mesh print, parsed and ready to upload.
///
/// Built off the main actor; `ParsedMesh` is all value types, so it crosses
/// back as ordinary `Sendable` state.
struct MeshPayload: Sendable {
    let mesh: ParsedMesh
    let texture: MeshTextureImage?

    /// Four times the largest texture mold bakes, on each axis and in total.
    nonisolated static let maximumTextureEdge = 4096
    nonisolated static let maximumTexturePixels = 4096 * 4096

    /// Parses the GLB and decodes its embedded baseColor image, if any.
    ///
    /// A texture that will not decode is DROPPED rather than failing the
    /// mesh: an untextured render still shades from its vertex colours, which
    /// is worth strictly more than a poster.
    nonisolated static func load(_ data: Data) throws -> MeshPayload {
        let mesh = try GLB.parse(data)
        guard let texture = mesh.baseColorTexture, mesh.uvs != nil else {
            return MeshPayload(mesh: mesh, texture: nil)
        }
        return MeshPayload(mesh: mesh, texture: decode(texture))
    }

    /// ImageIO rather than `NSImage`, because this runs off the main actor and
    /// wants the pixels rather than a drawable.
    nonisolated private static func decode(_ texture: MeshTexture) -> MeshTextureImage? {
        guard let source = CGImageSourceCreateWithData(texture.data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return nil }
        let width = image.width
        let height = image.height
        // The PIXEL COUNT, not each axis: a solid-colour 16384x16384 PNG is a
        // couple of kilobytes and expands to a 1 GiB array, with the `Data`
        // copy below making the peak 2 GiB -- from a few KB embedded in the
        // `.glb`. mold's own bakes are 2048 square at most (`hy3dpaint`), so
        // 4096 square is four times the largest texture this app will ever be
        // handed and a 64 MiB ceiling on what an untrusted one can ask for.
        guard width > 0, height > 0,
              width <= Self.maximumTextureEdge, height <= Self.maximumTextureEdge,
              width * height <= Self.maximumTexturePixels
        else { return nil }

        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = bytes.withUnsafeMutableBytes({ buffer in
            CGContext(data: buffer.baseAddress, width: width, height: height,
                      bitsPerComponent: 8, bytesPerRow: width * 4,
                      space: space, bitmapInfo: info)
        }) else { return nil }
        // glTF's UV origin is the image's TOP-left, and Core Graphics draws
        // bottom-up into this context, so the rows land in the order Metal's
        // own top-left origin wants. Never flip here.
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return MeshTextureImage(width: width, height: height, rgba: Data(bytes))
    }
}
