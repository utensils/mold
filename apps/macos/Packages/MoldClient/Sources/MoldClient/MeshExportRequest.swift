import Foundation

/// How many views a turntable renders, and how it plays back.
///
/// The server's bounds, from `GalleryExportRequest`
/// (`crates/mold-server/src/routes.rs:9452-9490`): frames 8…180 (36),
/// fps at most 30 (10), the rendered frame edge at most 2048 (512) — the
/// rasterizer's own ceiling, below the video export's 2160.
public struct MeshTurntableOptions: Equatable, Sendable {
    public static let frameBounds = 8...180
    public static let maximumFPS = 30
    public static let maximumDimension = 2048

    public var frames: Int
    public var fps: Int
    public var maxDimension: Int
    /// Render the object over nothing instead of the poster's slate ramp.
    public var transparent: Bool

    public init(frames: Int = 36, fps: Int = 10, maxDimension: Int = 512,
                transparent: Bool = false) {
        self.frames = frames
        self.fps = fps
        self.maxDimension = maxDimension
        self.transparent = transparent
    }

    /// The same options with every value inside the server's bounds, so a
    /// stepper cannot post a 422.
    public var clamped: MeshTurntableOptions {
        MeshTurntableOptions(
            frames: Swift.min(Swift.max(frames, Self.frameBounds.lowerBound),
                              Self.frameBounds.upperBound),
            fps: Swift.min(Swift.max(fps, 1), Self.maximumFPS),
            maxDimension: Swift.min(Swift.max(maxDimension, 16), Self.maximumDimension),
            transparent: transparent)
    }
}

/// The body posted to `POST /api/gallery/export/:filename`.
///
/// Port of `meshExportRequest`, `studio/lib/meshExport.ts:228-247`, plus the
/// turntable half the sheet fills in. With no geometry and no turntable it is
/// the bare `{ "format": … }` this app has always sent.
///
/// The two groups are MUTUALLY EXCLUSIVE and the server REFUSES the wrong
/// one — geometry keys on `glb` or on a turntable, turntable keys on a
/// geometry container — so this type carries at most one of them and the
/// factories below are the only way to build it.
public struct MeshExportRequest: Encodable, Equatable, Sendable {
    public let format: String
    public let geometry: MeshExportGeometry?
    public let turntable: MeshTurntableOptions?

    /// A geometry container. `geometry` is nil for an older host, which gets
    /// the bare format — never the keys it would silently ignore.
    public static func geometry(format: String,
                                _ geometry: MeshExportGeometry?) -> MeshExportRequest {
        MeshExportRequest(format: MeshExport.normalise(format),
                          geometry: MeshExport.takesGeometryOptions(format) ? geometry : nil,
                          turntable: nil)
    }

    /// An animated turntable.
    public static func turntable(format: String,
                                 _ options: MeshTurntableOptions) -> MeshExportRequest {
        MeshExportRequest(format: MeshExport.normalise(format), geometry: nil,
                          turntable: MeshExport.isAnimated(format) ? options.clamped : nil)
    }

    private enum CodingKeys: String, CodingKey {
        case format, sizeMm, upAxis, origin, frames, fps, maxDimension, transparent
    }

    /// Absent keys, never null ones: the server distinguishes "not asked" from
    /// a value, and `size_mm` is OMITTED for "as stored" because the wire has
    /// no way to ask a format whose default is a size to skip scaling.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(format, forKey: .format)
        if let geometry {
            try container.encodeIfPresent(geometry.sizeMm, forKey: .sizeMm)
            try container.encode(geometry.upAxis, forKey: .upAxis)
            try container.encode(geometry.origin, forKey: .origin)
        }
        if let turntable {
            try container.encode(turntable.frames, forKey: .frames)
            try container.encode(turntable.fps, forKey: .fps)
            try container.encode(turntable.maxDimension, forKey: .maxDimension)
            try container.encode(turntable.transparent, forKey: .transparent)
        }
    }
}
