import Foundation
import MoldClient

/// Why the 3-D view could not start, in ONE sentence.
///
/// A viewer that cannot start is NEVER a black rectangle: no Metal device, a
/// refused download, a corrupt file and a shader that will not build all land
/// on the poster the gallery already has, with a line saying why
/// (`MeshViewer.vue:12-15`).
public enum MeshViewFailure: Error {
    case noDevice
    case shaders(String)
    case upload
    case unreadable(String)
    case transport(String)

    /// What the person reads under the poster.
    public var sentence: String {
        switch self {
        case .noDevice:
            "This Mac can't draw 3-D previews, so here's the poster."
        case .shaders:
            "The 3-D view couldn't start, so here's the poster."
        case .upload:
            "There wasn't room on the graphics card for this mesh, so here's the poster."
        case let .unreadable(reason):
            "This mesh file couldn't be read. \(reason) It is still in the Library."
        case let .transport(reason):
            "That mesh didn't arrive. \(reason)"
        }
    }
}
