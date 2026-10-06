import Foundation

/// Independently downloadable texture maps on the print's holding machine.
public struct GenerationAsset: Codable, Hashable, Sendable, Identifiable {
    public let assetId: String
    public let role: String
    public let displayName: String
    public let mediaType: String
    public let sizeBytes: UInt64
    public let sha256: String
    public let width: Int?
    public let height: Int?
    public var id: String { assetId }
}
