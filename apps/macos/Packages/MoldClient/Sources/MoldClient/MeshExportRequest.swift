import Foundation

/// How many views a turntable renders, and how it plays back.
///
/// Every bound here is the SERVER's, cited to the line, and the per-field
/// ranges are not the whole of it: a turntable is refused before a frame
/// renders when the whole sweep's buffer exceeds `MAX_TURNTABLE_RGB_BYTES`,
/// so 36 frames -- the default -- at the offered 2048 px is 432 MiB and a
/// 422. The budget is modelled here, so a stepper can only ever offer a
/// combination the server accepts.
public struct MeshTurntableOptions: Equatable, Sendable {
    /// `FRAMES_RANGE` (`crates/mold-inference/src/hunyuan3d/turntable.rs:36`).
    public static let frameBounds = 8...180
    /// `FPS_RANGE` (`:41`).
    public static let fpsBounds = 1...30
    /// 240 is the server's own turntable floor (`routes.rs:10116`) and 2048 is
    /// `MAX_POSTER_SIZE`, the rasterizer's ceiling (`poster.rs:61`).
    public static let dimensionBounds = 240...2048
    /// `MAX_TURNTABLE_RGB_BYTES` (`turntable.rs:51`): the whole sweep's frame
    /// buffer, refused BEFORE a frame renders.
    public static let maximumFrameBytes = 256 * 1024 * 1024

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

    /// How many bytes a frame costs.
    ///
    /// A TRANSPARENT sweep keeps its coverage, so its frames are RGBA all the
    /// way to the encoder rather than RGB with a per-frame copy -- a quarter
    /// of the budget, which is why ticking the box alone can refuse an export
    /// that was fine without it (`turntable.rs:98-100`).
    public static func channels(transparent: Bool) -> Int { transparent ? 4 : 3 }

    /// The MOST frames the server will render at this size, which is the
    /// stepper's real upper bound. Never below the floor: at 2048 px with
    /// transparency -- the most expensive combination the server accepts --
    /// the budget still buys sixteen.
    public static func maximumFrames(atDimension dimension: Int,
                                     transparent: Bool) -> Int {
        let edge = Swift.min(Swift.max(dimension, dimensionBounds.lowerBound),
                             dimensionBounds.upperBound)
        let perFrame = edge * edge * channels(transparent: transparent)
        let affordable = perFrame > 0 ? maximumFrameBytes / perFrame : frameBounds.upperBound
        return Swift.max(frameBounds.lowerBound,
                         Swift.min(frameBounds.upperBound, affordable))
    }

    /// The same options with every value inside the server's bounds AND
    /// inside its frame budget, so a stepper cannot post a 422.
    public var clamped: MeshTurntableOptions {
        let edge = Swift.min(Swift.max(maxDimension, Self.dimensionBounds.lowerBound),
                             Self.dimensionBounds.upperBound)
        let ceiling = Self.maximumFrames(atDimension: edge, transparent: transparent)
        return MeshTurntableOptions(
            frames: Swift.min(Swift.max(frames, Self.frameBounds.lowerBound), ceiling),
            fps: Swift.min(Swift.max(fps, Self.fpsBounds.lowerBound),
                           Self.fpsBounds.upperBound),
            maxDimension: edge,
            transparent: transparent)
    }

    /// What to say when the size or transparency has taken frames off the
    /// table, or nil when the whole range is available. The server's own
    /// reason, in the app's words, BEFORE the export is attempted.
    public var budgetNote: String? {
        let ceiling = Self.maximumFrames(atDimension: maxDimension, transparent: transparent)
        guard ceiling < Self.frameBounds.upperBound else { return nil }
        let because = transparent ? " with a transparent background" : ""
        return "At \(clamped.maxDimension) px\(because) this machine renders at most "
            + "\(ceiling) views a turn."
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
