import Foundation
import ImageIO
import Testing
@testable import MoldClient

struct SourceMediaOriginalTests {
    private func png(width: Int) throws -> SourceMediaFile {
        let file = try SourceMediaFile(extension: "bin")
        let pixels = Data(repeating: 80, count: width * 4)
        let provider = try #require(CGDataProvider(data: pixels as CFData))
        let image = try #require(CGImage(width: width, height: 1, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try #require(CGImageDestinationCreateWithURL(file.url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return file
    }

    @Test func extensionlessImageIsInspectedAndDecodedWithinDisplayLimit() async throws {
        let file = try png(width: 5_000)
        let result = try await SourceMediaOriginal.inspect(file)
        #expect(result.kind == .image)
        #expect(try #require(result.image).image.width <= 4_096)
        #expect(result.image?.dimensions?.contains("5000") == true)
    }

    @Test func pathologicalImageDimensionsAreRejectedBeforeDecode() async throws {
        let file = try png(width: 65_537)
        await #expect(throws: SourceMediaOriginal.InspectionError.imageTooLarge) {
            try await SourceMediaOriginal.inspect(file)
        }
    }

    @Test func extensionlessVideoUsesActualTracks() async throws {
        var file: SourceMediaFile? = try SourceMediaFile(extension: "bin")
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/reference-video.mp4")
        try Data(contentsOf: fixture).write(to: try #require(file?.url))
        let result = try await SourceMediaOriginal.inspect(try #require(file))
        #expect(result.kind == .video)
        #expect(result.image == nil)
        #expect(result.url.pathExtension == "mp4")
        let originalAttributes = try FileManager.default.attributesOfItem(atPath: try #require(file?.url.path))
        let playbackAttributes = try FileManager.default.attributesOfItem(atPath: result.url.path)
        #expect((originalAttributes[.systemFileNumber] as? NSNumber) == (playbackAttributes[.systemFileNumber] as? NSNumber))
        file = nil
        #expect(!FileManager.default.fileExists(atPath: result.url.path))
    }

    @Test func extensionlessPCMIsAudio() async throws {
        let file = try SourceMediaFile(extension: "bin")
        var wav = Data("RIFF".utf8)
        func little<T: FixedWidthInteger>(_ value: T) -> Data {
            var value = value.littleEndian
            return withUnsafeBytes(of: &value) { Data($0) }
        }
        wav.append(little(UInt32(36 + 1600)))
        wav.append(Data("WAVEfmt ".utf8)); wav.append(little(UInt32(16)))
        wav.append(little(UInt16(1))); wav.append(little(UInt16(1)))
        wav.append(little(UInt32(8000))); wav.append(little(UInt32(16000)))
        wav.append(little(UInt16(2))); wav.append(little(UInt16(16)))
        wav.append(Data("data".utf8)); wav.append(little(UInt32(1600)))
        wav.append(Data(repeating: 0, count: 1600))
        try wav.write(to: file.url)
        #expect(try await SourceMediaOriginal.inspect(file).kind == .audio)
    }

    @Test func corruptContentReportsReadableFailureAndReleasesFile() async throws {
        var file: SourceMediaFile? = try SourceMediaFile(extension: "png")
        let url = try #require(file?.url)
        try Data("not an image or movie".utf8).write(to: url)
        do {
            _ = try await SourceMediaOriginal.inspect(try #require(file))
            Issue.record("Expected unsupported original")
        } catch let error as SourceMediaOriginal.InspectionError {
            #expect(error == .unsupported)
            #expect(error.localizedDescription.contains("unsupported or damaged"))
        }
        file = nil
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func containerSignatureAloneDoesNotAuthorizePlayback() async throws {
        let file = try SourceMediaFile(extension: "bin")
        try Data([82, 73, 70, 70, 255, 255, 255, 255, 87, 65, 86, 69]).write(to: file.url)
        await #expect(throws: SourceMediaOriginal.InspectionError.unsupported) {
            try await SourceMediaOriginal.inspect(file)
        }
    }

    @Test func cancelledInspectionDoesNotPublishMedia() async throws {
        let file = try SourceMediaFile(extension: "bin")
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await SourceMediaOriginal.inspect(file)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test func inspectionTimeoutCancelsSlowProbe() async throws {
        await #expect(throws: SourceMediaOriginal.InspectionError.timedOut) {
            try await SourceMediaOriginal.withTimeout(.milliseconds(10), cancel: {}) {
                try await Task.sleep(for: .seconds(60))
                return SourceMediaOriginal.Kind.audio
            }
        }
    }
}
