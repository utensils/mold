import Foundation

/// The one live sentence under the size control: what the exported file will
/// actually measure.
///
/// Port of `meshExportSizeLabel`, `studio/lib/meshExport.ts:200-226`. With a
/// box from the viewer it names all three extents; without one it can still
/// name the knob itself.
public extension MeshExportGeometry {
    static func sizeLabel(bounds: MeshBounds?, options: MeshExportGeometry) -> String {
        let dimensions = dimensionsMm(bounds: bounds, sizeMm: options.sizeMm,
                                      upAxis: options.upAxis)
        guard let sizeMm = options.sizeMm else {
            guard let dimensions else { return "as stored" }
            return "as stored (\(joined(dimensions, decimals: 2)))"
        }
        guard let dimensions else { return "longest side \(trimmed(sizeMm)) mm" }
        return "\(joined(dimensions, decimals: 1)) mm"
    }

    private static func joined(_ extents: SIMD3<Double>, decimals: Int) -> String {
        [extents.x, extents.y, extents.z]
            .map { String(format: "%.\(decimals)f", $0) }
            .joined(separator: " × ")
    }

    /// `120` rather than `120.0`, and `0` rather than an empty string — the
    /// reference trims trailing zeros off a one-decimal rendering.
    private static func trimmed(_ value: Double) -> String {
        var text = String(format: "%.1f", value)
        while text.contains("."), text.hasSuffix("0") { text.removeLast() }
        if text.hasSuffix(".") { text.removeLast() }
        return text.isEmpty ? "0" : text
    }
}
