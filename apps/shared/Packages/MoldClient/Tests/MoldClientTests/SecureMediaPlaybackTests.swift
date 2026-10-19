import AVFoundation
import Combine
import Synchronization
import CoreVideo
import CoreGraphics
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

    @Test(arguments: [false, true])
    func depletedStreamRecoversDecodedFramesAndRespectsPause(pauseDuringOutage: Bool) async throws {
        let gate = MediaReadGate()
        let transport = MediaRangeFixture(data: try Data(contentsOf: fixture), delay: .milliseconds(10), gate: gate)
        let backend = HTTPBackend(host: .init(name: "Outage fixture", baseURL: URL(string: "https://not-fetched.invalid")!), transport: transport)
        let media = try await backend.playableMedia(for: "movie.mp4")
        let item = media.playerItem()
        let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        item.add(output)
        let player = AVPlayer(playerItem: item)
        let session = VideoPlaybackSession()
        defer { session.cancel(); player.replaceCurrentItem(with: nil); item.asset.cancelLoading() }
        try await media.prepare(player)
        session.attach(player)
        let stalls = Mutex(0)
        let notifications = NotificationCenter.default.publisher(for: .AVPlayerItemPlaybackStalled, object: item)
            .sink { _ in stalls.withLock { $0 += 1 } }
        let rates = player.publisher(for: \.rate).receive(on: DispatchQueue.main).sink { rate in
            MainActor.assumeIsolated { session.nativeRateChanged(rate) }
        }
        defer { rates.cancel(); notifications.cancel() }
        await gate.close()
        session.play()
        let stallDeadline = ContinuousClock.now + .seconds(5)
        while ContinuousClock.now < stallDeadline && stalls.withLock({ $0 }) == 0 {
            _ = output.copyPixelBuffer(forItemTime: player.currentTime(), itemTimeForDisplay: nil)
            try await Task.sleep(for: .milliseconds(20))
        }
        try #require(stalls.withLock { $0 } > 0, "The fixture must produce a real decoder-backed network stall")
        let held = player.currentTime().seconds
        try await Task.sleep(for: .seconds(1))
        #expect(player.currentTime().seconds - held < 0.1, "Do not run the clock ahead while decoded media is unavailable")
        #expect(session.intendsToPlay)
        if pauseDuringOutage { player.pause() }
        await gate.open()
        if pauseDuringOutage {
            try await Task.sleep(for: .milliseconds(500))
            #expect(player.rate == 0 && !session.intendsToPlay)
            session.play()
        }
        let deadline = ContinuousClock.now + .seconds(10)
        var decoded = CMTime.invalid
        while ContinuousClock.now < deadline && (!decoded.isNumeric || decoded.seconds < 2.5) {
            if let error = item.error { throw error }
            var displayed = CMTime.invalid
            if output.copyPixelBuffer(forItemTime: player.currentTime(), itemTimeForDisplay: &displayed) != nil,
               displayed.isNumeric { decoded = displayed }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(decoded.isNumeric && decoded.seconds >= 2.5, "Resume actual decoded presentation samples after the connection recovers")
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
        defer { player.pause(); player.replaceCurrentItem(with: nil); item.asset.cancelLoading() }
        try await media?.prepare(player)
        #expect(player.rate == 0, "Preparing a stream must not override the user's playback choice")
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
    @Test(arguments: [("endpoint-bframes.mp4", 2.0), ("endpoint-vfr.mp4", 3.0),
                      ("endpoint-audio-overhang.mp4", 2.0), ("endpoint-offset.mp4", 3.0),
                      ("endpoint-single.mp4", 0.0)], [false, true])
    func endpointIsTheFinalPresentationSampleOnLocalAndSecureMedia(fixture: (String, Double), secure: Bool) async throws {
        let url = self.fixture.deletingLastPathComponent().appending(path: fixture.0)
        let item: AVPlayerItem
        if secure {
            let transport = MediaRangeFixture(data: try Data(contentsOf: url))
            let backend = HTTPBackend(host: .init(name: "Endpoint fixture", baseURL: URL(string: "https://not-fetched.invalid")!, apiKey: "fixture-key"), transport: transport)
            item = try await backend.playableMedia(for: fixture.0).playerItem()
        } else { item = AVPlayerItem(url: url) }
        defer { item.asset.cancelLoading() }
        let endpoint = try await VideoPlaybackEndpoint.resolve(in: item.asset)
        #expect(endpoint.isNumeric)
        #expect(abs(endpoint.seconds - fixture.1) < 0.0001, "Use presentation order and edit-list mapping, not last decode sample or item duration")
        let generator = AVAssetImageGenerator(asset: item.asset)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let image = try await generator.image(at: endpoint)
        #expect(abs(image.actualTime.seconds - fixture.1) < 0.0001)
        var rgba = [UInt8](repeating: 0, count: 4)
        let drew = rgba.withUnsafeMutableBytes { pixels -> Bool in
            guard let context = CGContext(data: pixels.baseAddress, width: 1, height: 1,
                bitsPerComponent: 8, bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.draw(image.image, in: CGRect(x: 0, y: 0, width: 1, height: 1))
            return true
        }
        #expect(drew && rgba[2] > 180 && rgba[0] < 60, "The final frame is blue, not the preceding red frame or a black endpoint")
    }

    @Test(arguments: [false, true])
    func realAudioTailFinishesBeforeLocalOrSecureSessionHoldsTheVisualEndpoint(secure: Bool) async throws {
        let url = fixture.deletingLastPathComponent().appending(path: "endpoint-audio-overhang.mp4")
        let media: PlayableMedia
        if secure {
            let backend = HTTPBackend(host: .init(name: "Audio endpoint fixture", baseURL: URL(string: "https://not-fetched.invalid")!, apiKey: "fixture-key"), transport: MediaRangeFixture(data: try Data(contentsOf: url)))
            media = try await backend.playableMedia(for: "endpoint-audio-overhang.mp4")
        } else { media = PlayableMedia(url: url) }
        let item = media.playerItem()
        let player = AVPlayer(playerItem: item)
        player.isMuted = true
        let session = VideoPlaybackSession()
        defer { session.cancel(); player.pause(); player.replaceCurrentItem(with: nil); item.asset.cancelLoading() }
        media.configure(player); try await media.prepare(player)
        session.attach(player); player.play()
        let clock = ContinuousClock(), deadline = ContinuousClock.now + .seconds(10)
        while clock.now < deadline && (player.currentTime().seconds < 2.9 || player.rate != 0) {
            if let error = item.error { throw error }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(player.currentTime().seconds >= 2.9, "The actual player must reach the legitimate audio tail, beyond video end 2.25")
        #expect(player.rate == 0)
        #expect(!item.forwardPlaybackEndTime.isNumeric)
        session.handleEnd(repeating: false)
        let heldDeadline = clock.now + .seconds(3)
        while clock.now < heldDeadline && abs(player.currentTime().seconds - 2) > 0.001 {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(session.ended)
        #expect(abs(player.currentTime().seconds - 2) < 0.001)
        #expect(player.rate == 0)
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
    let gate: MediaReadGate?
    init(data: Data, delay: Duration = .zero, gate: MediaReadGate? = nil) { self.data = data; self.delay = delay; self.gate = gate }
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
        return (MoldResponseBytes(reader: MediaBytes(bytes, delay: delay, gate: gate)), HTTPURLResponse(url: request.url!, statusCode: 206, httpVersion: nil, headerFields: headers)!)
    }
}
private actor MediaBytes: MoldBodyReader {
    var data: Data
    let delay: Duration
    let gate: MediaReadGate?
    init(_ data: Data, delay: Duration = .zero, gate: MediaReadGate? = nil) { self.data = data; self.delay = delay; self.gate = gate }
    func read() async throws -> Data? {
        try await gate?.wait()
        if delay > .zero { try await Task.sleep(for: delay) }
        guard !data.isEmpty else { return nil }
        let next = Data(data.prefix(16_384)); data.removeFirst(next.count); return next
    }
    func cancel() { data = Data() }
}

private actor MediaReadGate {
    private var blocked = false
    func close() { blocked = true }
    func open() { blocked = false }
    func wait() async throws {
        while blocked { try await Task.sleep(for: .milliseconds(20)) }
    }
}
