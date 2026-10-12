import AVFoundation
import Foundation
import ObjectiveC
import UniformTypeIdentifiers

/// An AVPlayer item with an owned encrypted range loader. The player never receives
/// a remote HTTP URL or bearer ticket, and seeking does not download a whole movie.
public struct PlayableMedia: Sendable {
    public let localURL: URL?
    private let backend: HTTPBackend?
    private let mediaURL: URL?
    public init(url: URL) { localURL = url; backend = nil; mediaURL = nil }
    init(backend: HTTPBackend, url: URL) { self.backend = backend; mediaURL = url; localURL = nil }

    @MainActor public func playerItem() -> AVPlayerItem {
        guard let backend, let mediaURL else { return AVPlayerItem(url: localURL!) }
        let asset = AVURLAsset(url: URL(string: "mold-secure://media")!.appending(path: UUID().uuidString).appending(path: mediaURL.lastPathComponent))
        let loader = SecureMediaLoader(backend: backend, url: mediaURL)
        asset.resourceLoader.setDelegate(loader, queue: .main)
        let item = AVPlayerItem(asset: asset)
        objc_setAssociatedObject(item, &secureMediaLoaderKey, loader, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return item
    }
}

private nonisolated(unsafe) var secureMediaLoaderKey: UInt8 = 0

@MainActor private final class SecureMediaLoader: NSObject, AVAssetResourceLoaderDelegate {
    let backend: HTTPBackend
    let url: URL
    var tasks: [ObjectIdentifier: Task<Void, Never>] = [:]
    init(backend: HTTPBackend, url: URL) { self.backend = backend; self.url = url }
    deinit { for task in tasks.values { task.cancel() } }

    nonisolated func resourceLoader(_ resourceLoader: AVAssetResourceLoader,
                                    shouldWaitForLoadingOfRequestedResource request: AVAssetResourceLoadingRequest) -> Bool {
        MainActor.assumeIsolated {
            let id = ObjectIdentifier(request)
            tasks[id] = Task { await load(request); tasks.removeValue(forKey: id) }
        }
        return true
    }
    nonisolated func resourceLoader(_ resourceLoader: AVAssetResourceLoader, didCancel request: AVAssetResourceLoadingRequest) {
        MainActor.assumeIsolated { tasks.removeValue(forKey: ObjectIdentifier(request))?.cancel() }
    }

    private func load(_ loading: AVAssetResourceLoadingRequest) async {
        var bytes: MoldResponseBytes?
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 300
            if let key = backend.host.apiKey { request.setValue(key, forHTTPHeaderField: "X-Api-Key") }
            let dataRequest = loading.dataRequest
            let offset = dataRequest.map { max($0.requestedOffset, $0.currentOffset) } ?? 0
            let length = dataRequest.map { Int64($0.requestedLength) }
            guard offset >= 0, length.map({ $0 > 0 }) ?? true else { throw SecureConnectionError.protocolViolation }
            if let dataRequest, let length {
                let end = offset.addingReportingOverflow(length - 1)
                guard !end.overflow else { throw SecureConnectionError.protocolViolation }
                request.setValue(dataRequest.requestsAllDataToEndOfResource ? "bytes=\(offset)-" : "bytes=\(offset)-\(end.partialValue)", forHTTPHeaderField: "Range")
            } else { request.httpMethod = "HEAD" }
            let (body, response) = try await backend.response(request); bytes = body
            guard (200...299).contains(response.statusCode) else { throw await backend.streamRefusal(response, body) }
            if offset > 0, response.statusCode != 206 { throw SecureConnectionError.protocolViolation }
            let total = try MediaRangeResponse.totalLength(response, expectedOffset: offset)
            if let info = loading.contentInformationRequest {
                let mime = response.mimeType ?? "video/mp4"
                info.contentType = UTType(mimeType: mime)?.identifier ?? UTType.mpeg4Movie.identifier
                info.contentLength = total
                info.isByteRangeAccessSupported = response.statusCode == 206 || response.value(forHTTPHeaderField: "Accept-Ranges")?.lowercased() == "bytes"
            }
            if let dataRequest {
                var remaining = dataRequest.requestsAllDataToEndOfResource ? Int64.max : length!
                for try await chunk in body.chunks {
                    try Task.checkCancellation()
                    guard !loading.isCancelled else { throw CancellationError() }
                    let count = Int(min(Int64(chunk.count), remaining))
                    dataRequest.respond(with: Data(chunk.prefix(count)))
                    remaining -= Int64(count)
                    if remaining == 0 { break }
                }
            }
            await body.cancel()
            guard !loading.isCancelled else { return }
            loading.finishLoading()
        } catch {
            await bytes?.cancel()
            if !loading.isCancelled { loading.finishLoading(with: error) }
        }
    }
}

enum MediaRangeResponse {
    static func totalLength(_ response: HTTPURLResponse, expectedOffset: Int64) throws -> Int64 {
        if response.statusCode == 206 {
            guard let range = response.value(forHTTPHeaderField: "Content-Range"), range.hasPrefix("bytes ") else { throw SecureConnectionError.protocolViolation }
            let parts = range.dropFirst(6).split(separator: "/"), span = parts.first?.split(separator: "-") ?? []
            guard parts.count == 2, span.count == 2, let start = Int64(span[0]), let end = Int64(span[1]),
                  let total = Int64(parts[1]), start == expectedOffset, end >= start, total > end else { throw SecureConnectionError.protocolViolation }
            return total
        }
        guard response.expectedContentLength >= 0 else { throw SecureConnectionError.protocolViolation }
        return response.expectedContentLength
    }
}
