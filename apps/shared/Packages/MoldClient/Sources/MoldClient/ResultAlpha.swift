import Foundation
import ImageIO

/// Whether a just-finished render is drawn over the checkerboard alpha bed.
///
/// The Generate canvas holds a `BatchResult` -- a filename, no metadata -- so
/// it answers from the bytes it already fetched: the print's own embedded
/// `mold:parameters` (`OutputMetadata.showsAlphaBed`, the same rule the
/// Library reads), else the container's own alpha channel, which is what
/// `has_alpha` records and the only signal a WebP (no embedded parameters
/// reader here) carries. The bed is drawn only under the picture's own
/// rectangle, so an alpha channel with every pixel opaque hides it entirely.
public enum ResultAlpha {
    public static func showsBed(_ file: Data, named filename: String) -> Bool {
        if let json = EmbeddedPrintMetadata.json(in: file, named: filename),
           let metadata = try? MoldJSON.decoder.decode(OutputMetadata.self, from: json),
           metadata.showsAlphaBed {
            return true
        }
        guard let source = CGImageSourceCreateWithData(file as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                  as? [CFString: Any]
        else { return false }
        return properties[kCGImagePropertyHasAlpha] as? Bool ?? false
    }
}
