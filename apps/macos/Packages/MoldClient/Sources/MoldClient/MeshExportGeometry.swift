import Foundation

/// Which axis the exported file calls up.
public enum MeshUpAxis: String, Codable, Hashable, Sendable {
    case y
    case z
}

/// Where the exported file puts the origin relative to the mesh.
public enum MeshExportOrigin: String, Codable, Hashable, Sendable {
    case center
    case floor
}

/// One resolved set of geometry knobs, as a client holds it in a form.
///
/// `sizeMm == nil` means "as stored" — model units, no scaling at all.
public struct MeshExportGeometry: Codable, Hashable, Sendable {
    public var sizeMm: Double?
    public var upAxis: MeshUpAxis
    public var origin: MeshExportOrigin

    public init(sizeMm: Double?, upAxis: MeshUpAxis, origin: MeshExportOrigin) {
        self.sizeMm = sizeMm
        self.upAxis = upAxis
        self.origin = origin
    }
}

/// The bounds and default of the `size_mm` control, in millimetres.
public struct MeshSizeControl: Codable, Hashable, Sendable {
    public let min: Double
    public let max: Double
    public let `default`: Double

    public init(min: Double, max: Double, default: Double) {
        self.min = min
        self.max = max
        self.default = `default`
    }
}

/// `capabilities.mesh.export_geometry` — the host's own bounds, the axes and
/// origins it accepts, and its per-format defaults.
///
/// ABSENT on a host that predates the feature, which is the ONLY gate: a
/// client that cannot see this block must post `{ "format": … }` exactly as
/// before, because an older server DROPS unknown fields rather than refusing
/// them and would silently write the unscaled mesh the person thought they had
/// resized (`studio/lib/meshExport.ts:101-115`).
public struct MeshExportGeometryCapabilities: Codable, Hashable, Sendable {
    public let sizeMm: MeshSizeControl
    public let upAxes: [MeshUpAxis]
    public let origins: [MeshExportOrigin]
    /// Keyed by lower-case container; a format absent here takes no options.
    public let defaults: [String: MeshExportGeometry]

    public init(sizeMm: MeshSizeControl, upAxes: [MeshUpAxis], origins: [MeshExportOrigin],
                defaults: [String: MeshExportGeometry]) {
        self.sizeMm = sizeMm
        self.upAxes = upAxes
        self.origins = origins
        self.defaults = defaults
    }
}

public extension MeshExportGeometry {
    /// The host's defaults for one container, or `nil` meaning DO NOT OFFER
    /// the options at all — a host that predates the block, a container it
    /// does not scale (glb, the turntables), or one it simply does not list.
    /// A caller that gets `nil` posts the bare `{ "format": … }`.
    static func defaults(_ capabilities: MeshExportGeometryCapabilities?,
                         format: String) -> MeshExportGeometry? {
        guard let capabilities, MeshExport.takesGeometryOptions(format),
              let entry = capabilities.defaults[MeshExport.normalise(format)]
        else { return nil }
        let size = entry.sizeMm.flatMap { value -> Double? in
            guard value.isFinite else { return nil }
            return Swift.min(capabilities.sizeMm.max, Swift.max(capabilities.sizeMm.min, value))
        }
        return MeshExportGeometry(
            sizeMm: size,
            upAxis: capabilities.upAxes.contains(entry.upAxis)
                ? entry.upAxis : (capabilities.upAxes.first ?? .y),
            origin: capabilities.origins.contains(entry.origin)
                ? entry.origin : (capabilities.origins.first ?? .floor))
    }

    /// The extents the exported file will have, along ITS OWN X, Y and Z axes.
    ///
    /// The stored mesh is Y-up, so a Z-up export rotates `(x, y, z) ->
    /// (x, -z, y)` and its axes read width × depth × height; a Y-up export
    /// keeps the stored order. Scaling is uniform: the longest stored extent
    /// becomes `sizeMm`, and a nil `sizeMm` means model units are written
    /// verbatim. `nil` when there is no usable box — nothing about the export
    /// changes, only what can be SAID about it.
    static func dimensionsMm(bounds: MeshBounds?, sizeMm: Double?,
                             upAxis: MeshUpAxis) -> SIMD3<Double>? {
        guard let bounds else { return nil }
        let extents = bounds.max - bounds.min
        guard extents.x.isFinite, extents.y.isFinite, extents.z.isFinite,
              extents.min() >= 0
        else { return nil }
        let longest = extents.max()
        guard longest > 0 else { return nil }
        let scale = sizeMm.map { $0 / longest } ?? 1
        let ordered = upAxis == .z
            ? SIMD3<Double>(extents.x, extents.z, extents.y)
            : extents
        return ordered * scale
    }
}
