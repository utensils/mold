import AVFoundation
import CoreVideo
import Foundation
import Testing
@testable import MoldClient

@Suite(.serialized)
@MainActor struct SecureMediaPlaybackTests {
    private var fixture: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/playback-range-stress.mp4")
    }

    @Test func localFileControlDecodesAndAdvancesTheSameClip() async throws {
        try await assertPlayback(AVPlayerItem(url: fixture))
    }

    @Test(arguments: [false, true], [Duration.zero, .milliseconds(10)])
    func avFoundationLoadsAndSeeksThroughRangeAdapter(trashed: Bool, delay: Duration) async throws {
        // A generated 720p clip exercises real playback throughput; the old
        // 32 KiB sample only demonstrated metadata and single-frame decoding.
        // 16 KiB every 10 ms exceeds this clip's ~700 KiB/s bitrate. A 30 ms
        // interval starves playback and tests manual stall recovery instead.
        let transport = MediaRangeFixture(data: try Data(contentsOf: fixture), delay: delay)
        let backend = HTTPBackend(host: .init(name: "Encrypted fixture", baseURL: URL(string: "https://not-fetched.invalid")!, apiKey: "encrypted-api-credential"), transport: transport)
        let media = try await backend.playableMedia(for: "movie.mp4", trashed: trashed)
        #expect(media.localURL == nil)
        let item = media.playerItem()
        do { try await assertPlayback(item, media: media) }
        catch {
            let trace = await transport.requests.map { "\($0.httpMethod ?? "GET") \($0.value(forHTTPHeaderField: "Range") ?? "metadata")" }
            print("Playback request trace: \(trace)")
            throw error
        }
        #expect(await transport.requests.allSatisfy { $0.value(forHTTPHeaderField: "Range") != nil || $0.httpMethod == "HEAD" })
        #expect(await transport.requests.isEmpty == false)
        #expect(await transport.requests.allSatisfy { $0.value(forHTTPHeaderField: "X-Api-Key") == "encrypted-api-credential" })
        #expect(await transport.requests.allSatisfy { $0.url?.query == (trashed ? "view=trash" : nil) })
    }

    @Test func playbackPolicyFollowsTheCurrentMediaWhenAPlayerIsReused() {
        let player = AVPlayer()
        let backend = HTTPBackend(host: .init(name: "Fixture", baseURL: URL(string: "https://not-fetched.invalid")!), transport: MediaRangeFixture(data: Data()))
        PlayableMedia(backend: backend, url: fixture).configure(player)
        #expect(!player.automaticallyWaitsToMinimizeStalling)
        PlayableMedia(url: fixture).configure(player)
        #expect(player.automaticallyWaitsToMinimizeStalling)
    }

    private func assertPlayback(_ item: AVPlayerItem, media: PlayableMedia? = nil) async throws {
        let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        item.add(output)
        let player = AVPlayer(playerItem: item)
        media?.configure(player)
        defer { player.pause(); player.replaceCurrentItem(with: nil); item.asset.cancelLoading() }
        player.play()
        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(10)
        var decodedFrame = false
        while clock.now < deadline, player.currentTime().seconds < 2.8 || !decodedFrame {
            if let error = item.error { throw error }
            if let frame = output.copyPixelBuffer(forItemTime: player.currentTime(), itemTimeForDisplay: nil) {
                decodedFrame = CVPixelBufferGetWidth(frame) > 0
            }
            try await Task.sleep(for: .milliseconds(50))
        }
        try #require(item.status == .readyToPlay)
        try #require(player.currentTime().seconds >= 2.8, "The clip must sustain playback, not just load metadata or its first frame")
        try #require(decodedFrame)
        player.pause()
        #expect(await player.seek(to: CMTime(seconds: 1, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero))
        let duration = try await item.asset.load(.duration)
        #expect(duration.seconds > 0)
        let generator = AVAssetImageGenerator(asset: item.asset)
        let image = try await generator.image(at: CMTime(seconds: duration.seconds * 0.8, preferredTimescale: 600))
        #expect(image.image.width > 0)
    }
    @Test func malformedRangesNeverReachThePlayer() {
        let url = URL(string: "https://fixture/movie.mp4")!
        let response = HTTPURLResponse(url: url, statusCode: 206, httpVersion: nil, headerFields: ["Content-Range": "bytes 0-999/20"])!
        #expect(throws: SecureConnectionError.protocolViolation) { try MediaRangeResponse.totalLength(response, expectedOffset: 0) }
    }
}

private actor MediaRangeFixture: MoldHTTPTransport {
    let data: Data
    let delay: Duration
    var requests: [URLRequest] = []
    init(data: Data, delay: Duration = .zero) { self.data = data; self.delay = delay }
    func response(for request: URLRequest, body: MoldRequestBody) async throws -> (MoldResponseBytes, HTTPURLResponse) {
        requests.append(request)
        var headers = ["Content-Type": "video/mp4", "Accept-Ranges": "bytes", "Content-Length": String(data.count)]
        if request.httpMethod == "HEAD" {
            return (MoldResponseBytes(reader: MediaBytes(Data())), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: headers)!)
        }
        let range = request.value(forHTTPHeaderField: "Range")!.dropFirst(6).split(separator: "-", omittingEmptySubsequences: false)
        let start = Int(range[0])!, end = min(Int(range[1]) ?? (data.count - 1), data.count - 1)
        guard start <= end else { throw SecureConnectionError.protocolViolation }
        let bytes = data.subdata(in: start..<(end + 1))
        headers["Content-Range"] = "bytes \(start)-\(end)/\(data.count)"; headers["Content-Length"] = String(bytes.count)
        return (MoldResponseBytes(reader: MediaBytes(bytes, delay: delay)), HTTPURLResponse(url: request.url!, statusCode: 206, httpVersion: nil, headerFields: headers)!)
    }
}
private actor MediaBytes: MoldBodyReader {
    var data: Data
    let delay: Duration
    init(_ data: Data, delay: Duration = .zero) { self.data = data; self.delay = delay }
    func read() async throws -> Data? {
        if delay > .zero { try await Task.sleep(for: delay) }
        guard !data.isEmpty else { return nil }
        let next = Data(data.prefix(16_384)); data.removeFirst(next.count); return next
    }
    func cancel() { data = Data() }
}
