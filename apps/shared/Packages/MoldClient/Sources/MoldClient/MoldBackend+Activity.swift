import Foundation

/// Everything a machine is doing, including the work that never becomes a
/// queue row: preparation, prompt rewrites, standalone upscales, downloads,
/// and durable sequences.
public protocol MoldActivityBackend: Sendable {
    func activity() async throws -> ActiveWorkSnapshot
}
