import Foundation

/// 3-D export menu policy, shared with web, desktop and the phone.
///
/// Port of `studio/lib/meshExport.ts:1-78`. GLB is the only stored form;
/// OBJ / STL / PLY — and the animated turntables a host RENDERS — are produced
/// on request through `POST /api/gallery/export/:filename`. The menu is built
/// from the holding host's own `capabilities.mesh.export_formats` and NEVER
/// from a client constant: a host that adds a container adds a menu entry with
/// no client release.
///
/// What this owns is the SPLIT of that advertised list. Animated containers
/// share the turntable sheet's options, so they collapse into a single entry
/// that opens it; everything else is a one-click transcode. The stored
/// container is dropped — the server lists it first so a client can see what
/// it holds, but "Export as GLB" beside Save is not an export.
public enum MeshExport {
    /// Containers that are a RENDER of the mesh rather than the mesh.
    static let animated: Set<String> = ["gif", "apng", "webp"]
    /// The stored container, which no export menu offers.
    static let stored = "glb"

    /// The split of a host's advertised list.
    public struct Split: Equatable, Sendable {
        /// Direct one-click transcodes: one menu entry each, in the host's order.
        public let files: [String]
        /// Animated turntables, which share the turntable sheet's options.
        public let animations: [String]

        public init(files: [String], animations: [String]) {
            self.files = files
            self.animations = animations
        }
    }

    public static func normalise(_ format: String) -> String {
        format.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Whether an advertised export container is an animated turntable.
    public static func isAnimated(_ format: String) -> Bool {
        animated.contains(normalise(format))
    }

    /// The host's advertised list, lower-cased, minus the stored container,
    /// split into the two kinds of menu entry.
    public static func split(_ advertised: [String]?) -> Split {
        var files: [String] = []
        var animations: [String] = []
        for raw in advertised ?? [] {
            let format = normalise(raw)
            if format == stored { continue }
            if animated.contains(format) {
                animations.append(format)
            } else {
                files.append(format)
            }
        }
        return Split(files: files, animations: animations)
    }

    /// The name an exported mesh is saved under: the print's own stem with the
    /// requested container's extension. The gallery filename never changes —
    /// this only names the copy that leaves the app. The advertised list is
    /// the host's, so this deliberately does not validate the format.
    public static func filename(_ filename: String, format: String) -> String {
        // `filename.replace(/\.[^.]+$/, "")`, and NOT
        // `NSString.deletingPathExtension`, which keeps a dot-file's whole name
        // -- so `.glb` would have exported as `.glb.stl` rather than falling
        // back to the stem.
        var stem = filename
        if let dot = filename.lastIndex(of: "."), filename.index(after: dot) < filename.endIndex,
           !filename[filename.index(after: dot)...].contains(".") {
            stem = String(filename[..<dot])
        }
        return "\(stem.isEmpty ? "mold-mesh" : stem).\(normalise(format))"
    }

    /// Whether a container is one geometry options could apply to AT ALL.
    ///
    /// The structural rule — the stored form and the turntables are excluded —
    /// and deliberately permissive about containers this build has never heard
    /// of. The host's own `defaults` table is the authority on what it will
    /// actually accept, so `MeshExportGeometry.defaults(...)` is what a caller
    /// gates on.
    public static func takesGeometryOptions(_ format: String) -> Bool {
        let value = normalise(format)
        return value != stored && !animated.contains(value)
    }
}
