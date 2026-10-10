import AppKit
import ImageIO
import MoldClient

/// Row previews survive native List recycling. Only a small decoded thumbnail is
/// retained, not the full input bytes; details still load their complete inputs.
@MainActor
final class QueuePreviewCache {
    struct Key: Hashable {
        let host: MoldHost
        let instance: String
        let job: String
    }
    struct Preview {
        let image: NSImage?
        let label: String
        let count: Int
    }
    private struct Cached {
        let value: Preview
        let expires: Date
    }
    private var values: [Key: Cached] = [:]
    private var order: [Key] = []
    private var requests: [Key: Task<Preview, Error>] = [:]
    private let lifetime: TimeInterval

    init(lifetime: TimeInterval = 60) { self.lifetime = lifetime }

    func preview(for key: Key, backend: any MoldBackend) async throws -> Preview {
        // A fling may create many rows while earlier thumbnail reads are slow.
        // Bound actual requests, and let offscreen callers cancel before they
        // acquire a lane. Existing callers still share an already-running read.
        while true {
            try Task.checkCancellation()
            if let cached = values[key], cached.expires > Date() {
                touch(key)
                return cached.value
            }
            if let request = requests[key] { return try await request.value }
            if requests.count < 4 { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        let request = Task {
            let inputs = try await backend.queueInputPreviews(id: key.job, firstOnly: true)
            let first = inputs.first { $0.bytes != nil }
            let image = first?.bytes.flatMap(Self.decode)
            return Preview(image: image, label: first?.input.label ?? "Source", count: inputs.count)
        }
        requests[key] = request
        defer { requests[key] = nil }
        let result = try await request.value
        values[key] = Cached(value: result, expires: Date().addingTimeInterval(lifetime))
        touch(key)
        while order.count > 128 { values.removeValue(forKey: order.removeFirst()) }
        return result
    }

    private func touch(_ key: Key) {
        order.removeAll { $0 == key }
        order.append(key)
    }

    private static func decode(_ data: Data) -> NSImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 192,
                kCGImageSourceShouldCacheImmediately: true,
              ] as CFDictionary) else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
    }
}
