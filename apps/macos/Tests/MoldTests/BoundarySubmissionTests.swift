import CoreGraphics
import Foundation
import ImageIO
import MoldClient
import Testing

@testable import Mold

@MainActor struct BoundarySubmissionTests {
    @Test(arguments: [SourceFit.default, .padFit])
    func lastOnlyBoundaryFitsCapturedBatchWithoutChangingAuthoring(policy: SourceFit) async throws {
        var json = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(FakeFixtures.recipe())) as? [String: Any])
        json["capabilities"] = ["boundary_frames": ["mode": "adjustable", "wire": "h3-endpoints", "min_frames": 9,
                                                      "first_required": false, "last_required": false]]
        let recipe = try MoldJSON.decoder.decode(GenerationRecipe.self, from: JSONSerialization.data(withJSONObject: json))
        let context = try #require(CGContext(data: nil, width: 128, height: 64, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue))
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 128, height: 64))
        let image = try #require(context.makeImage())
        let original = try #require(SourceFitRender.encodePNG(image)).base64EncodedString()
        let host = MoldHost(name: "boundary", baseURL: URL(string: "http://boundary")!)
        let backend = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "minimax-h3-fl2va:official-bf16"
        controller.draft.width = 64; controller.draft.height = 64
        controller.draft.frames = 9
        controller.draft.batchSize = 2
        controller.draft.locksSeed = true; controller.draft.seed = 42
        controller.draft.media.adoptedReferenceCapabilities = recipe.capabilities
        controller.draft.media.sourceFit = policy
        controller.draft.media.keyframes = [.init(frame: 8, image: original, name: "last.png")]
        controller.submit(on: host, backend: backend, recipe: recipe)
        #expect(controller.run.isBusy)
        // Preparation must use the snapshot captured by the press.
        controller.draft.prompt = "edited after submitting"
        for _ in 0..<200 where backend.submittedAdmissions.isEmpty {
            try await Task.sleep(for: .milliseconds(10))
        }
        let admission = try #require(backend.submittedAdmissions.first)
        #expect(admission.requests.count == 2)
        #expect(admission.requests.map(\.seed) == [42, 43])
        #expect(Set(admission.requests.compactMap(\.batchId)).count == 1)
        for request in admission.requests {
            #expect(request.prompt != "edited after submitting")
            #expect(request.sourceImage == nil)
            #expect(request.maskImage == nil)
            #expect(request.sourceFit == policy)
            let frame = try #require(request.keyframes?.first)
            #expect(frame.frame == 8)
            #expect(ReferenceCanvas.uprightPixels(ofBase64: frame.image) == SourcePixels(width: 64, height: 64))
            #expect(frame.image != original)
            let bytes = try #require(Data(base64Encoded: frame.image))
            let source = try #require(CGImageSourceCreateWithData(bytes as CFData, nil))
            let fittedImage = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
            let pixels = try #require(CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8,
                bytesPerRow: 64, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue))
            pixels.draw(fittedImage, in: CGRect(x: 0, y: 0, width: 64, height: 64))
            let buffer = try #require(pixels.data).assumingMemoryBound(to: UInt8.self)
            #expect(buffer[32 * 64 + 32] > 240)
            // A pad keeps the white rectangle and adds dark bands; crop fills
            // the canvas with its white center, proving the chosen policy ran.
            #expect(policy == .padFit ? buffer[64 + 32] < 15 : buffer[64 + 32] > 240)
        }
        #expect(controller.draft.media.keyframes.first?.image == original)
        #expect(controller.draft.media.sourceFit == policy)
        controller.stopAll()
    }
}
