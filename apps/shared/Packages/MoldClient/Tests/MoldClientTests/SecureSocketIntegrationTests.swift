import Foundation
import AVFoundation
import CoreVideo
import Testing
@testable import MoldClient

/// Opt-in UAT against the Rust fixture; no installed host or saved credential is touched.
struct SecureSocketIntegrationTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MOLD_SECURE_MEDIA_FIXTURE"] != nil))
    @MainActor func avFoundationDecodesAndSeeksAcrossAnActualNoiseWebSocket() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MOLD_SECURE_MEDIA_FIXTURE"])
        let fixture = try MoldJSON.decoder.decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let identity = try SecureIdentity(secret: SecureHex.decode(fixture.deviceIdentity, count: 64))
        let pins = SecureHost(binding: .init(signingKey: fixture.hostKey, noiseKey: fixture.hostNoiseKey, signature: fixture.hostBindingSignature),
            namespace: fixture.namespace, websocketURL: URL(string: "wss://fixture.invalid")!, grant: fixture.grant)
        let socket = try await SecureWebSocketConnection.connect(host: pins, identity: identity, endpoint: fixture.websocket)
        let transport = SecureHTTPTransport(connection: socket)
        do {
            let host = MoldHost(name: "Encrypted media fixture", baseURL: fixture.origin, apiKey: fixture.apiKey, secure: pins)
            var request = URLRequest(url: fixture.origin.appending(path: "api/gallery/image/fixture.mp4"))
            request.setValue(fixture.apiKey, forHTTPHeaderField: "X-Api-Key")
            let (download, _) = try await transport.response(for: request, body: .data(Data()))
            let downloaded = try await download.collected(upTo: 4 * 1024 * 1024)
            if let source = ProcessInfo.processInfo.environment["MOLD_SECURE_MEDIA_SOURCE"] {
                #expect(downloaded == (try Data(contentsOf: URL(fileURLWithPath: source))))
            }
            let backend = HTTPBackend(host: host, transport: transport)
            let media = try await backend.playableMedia(for: "fixture.mp4")
            #expect(media.localURL == nil)
            let item = media.playerItem()
            let duration = try await item.asset.load(.duration)
            #expect(duration.seconds >= 10)
            let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            item.add(output)
            let player = AVPlayer(playerItem: item)
            defer { player.pause(); player.replaceCurrentItem(with: nil) }
            player.play()
            let clock = ContinuousClock()
            let deadline = clock.now + .seconds(30)
            var decodedFrames = 0
            while player.currentTime().seconds < 11.8, clock.now < deadline {
                if let error = item.error { throw error }
                if let frame = output.copyPixelBuffer(forItemTime: player.currentTime(), itemTimeForDisplay: nil) {
                    #expect(CVPixelBufferGetWidth(frame) > 0)
                    decodedFrames += 1
                }
                try await Task.sleep(for: .milliseconds(50))
            }
            #expect(player.currentTime().seconds >= 11.8)
            #expect(decodedFrames > 0)
            let sought = await player.seek(to: CMTime(seconds: 8, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
            #expect(sought)
            let seekDeadline = clock.now + .seconds(5)
            var soughtFrame = false
            while player.currentTime().seconds < 8.15, clock.now < seekDeadline {
                if let error = item.error { throw error }
                if let frame = output.copyPixelBuffer(forItemTime: player.currentTime(), itemTimeForDisplay: nil) { soughtFrame = CVPixelBufferGetWidth(frame) > 0 }
                try await Task.sleep(for: .milliseconds(50))
            }
            #expect(player.currentTime().seconds >= 8.15)
            #expect(soughtFrame)
            await transport.close()
        } catch { await transport.close(); throw error }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MOLD_SECURE_FIXTURE"] != nil))
    func nativeWebSocketInteroperatesWithRustForHTTPUploadRangeAndEvents() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MOLD_SECURE_FIXTURE"])
        let fixture = try MoldJSON.decoder.decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let identity = try SecureIdentity(secret: SecureHex.decode(fixture.deviceIdentity, count: 64))
        let host = SecureHost(binding: .init(signingKey: fixture.hostKey, noiseKey: fixture.hostNoiseKey, signature: fixture.hostBindingSignature),
            namespace: fixture.namespace, websocketURL: URL(string: "wss://fixture.invalid")!, grant: fixture.grant)
        let socket = try await SecureWebSocketConnection.connect(host: host, identity: identity, endpoint: fixture.websocket)
        let transport = SecureHTTPTransport(connection: socket)
        do {
            var request = URLRequest(url: fixture.origin.appending(path: "api/status"))
            request.setValue(fixture.apiKey, forHTTPHeaderField: "X-Api-Key")
            let (status, response) = try await transport.response(for: request, body: .data(Data()))
            #expect(response.statusCode == 200)
            #expect(!(try await status.collected(upTo: 65_536)).isEmpty)
            request.url = fixture.origin.appending(path: "echo"); request.httpMethod = "POST"
            let payload = Data((0..<(1024 * 1024 + 13)).map { UInt8(truncatingIfNeeded: $0) })
            let (echo, echoResponse) = try await transport.response(for: request, body: .data(payload))
            #expect(echoResponse.statusCode == 200)
            #expect(try await echo.collected(upTo: payload.count + 1) == payload)
            request.url = fixture.origin.appending(path: "range"); request.httpMethod = "GET"
            request.setValue("bytes=100-196707", forHTTPHeaderField: "Range")
            let (range, rangeResponse) = try await transport.response(for: request, body: .data(Data()))
            #expect(rangeResponse.statusCode == 206)
            #expect(try await range.collected(upTo: 256 * 1024).count == 192 * 1024)
            request.url = fixture.origin.appending(path: "sse"); request.setValue(nil, forHTTPHeaderField: "Range")
            let (events, eventResponse) = try await transport.response(for: request, body: .data(Data()))
            #expect(eventResponse.statusCode == 200)
            let text = String(decoding: try await events.collected(upTo: 65_536), as: UTF8.self)
            #expect(text.components(separatedBy: "data:").count == 5)
            await transport.close()
        } catch { await transport.close(); throw error }
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MOLD_SECURE_INVITE"] != nil))
    func nativeInvitationClaimPinsTheHostAndReturnsAnAuthorizedDeviceGrant() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MOLD_SECURE_INVITE"])
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let payload = try MobilePairingPayload.parse(String(decoding: data, as: UTF8.self))
        let invitation = try #require(payload.secureInvitation)
        let identity = try SecureIdentity()
        let origin = try #require(URL(string: payload.baseURL))
        var endpoint = URLComponents(url: origin, resolvingAgainstBaseURL: false)!
        endpoint.scheme = "ws"; endpoint.path = "/_mold/secure"
        let connection = try await SecureWebSocketConnection.connectInvitation(invitation, identity: identity, endpoint: endpoint.url!)
        let claim = try await HTTPBackend.claimPairing(payload, invitation: invitation, identity: identity, connection: connection,
            origin: origin, clientName: "Swift secure fixture", clientKind: "macos")
        let host = try #require(claim.secure)
        try SecureAdmission.validate(host: host, device: identity.binding())
        let socket = try await SecureWebSocketConnection.connect(host: host, identity: identity, endpoint: endpoint.url!)
        let transport = SecureHTTPTransport(connection: socket)
        var request = URLRequest(url: origin.appending(path: "api/status"))
        request.setValue(claim.apiKey, forHTTPHeaderField: "X-Api-Key")
        do {
            let (bytes, response) = try await transport.response(for: request, body: .data(Data()))
            #expect(response.statusCode == 200)
            #expect(!(try await bytes.collected(upTo: 65_536)).isEmpty)
            await transport.close()
        } catch { await transport.close(); throw error }
    }

    private struct Fixture: Decodable {
        let origin: URL, websocket: URL
        let apiKey: String, deviceIdentity: String, hostKey: String, hostNoiseKey: String, hostBindingSignature: String, namespace: String
        let grant: SecureDeviceGrant
    }
}
