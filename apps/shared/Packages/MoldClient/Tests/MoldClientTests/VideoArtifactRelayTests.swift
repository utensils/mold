import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

private actor ArtifactSink {
    var bytes = Data()
    var loseReply = true
    var completed = false
    func status() throws -> VideoArtifactTransferStatus {
        try MoldJSON.decoder.decode(VideoArtifactTransferStatus.self, from: Data("""
        {"handle":"\(String(repeating:"f",count:64))","archive_identity_sha256":"\(String(repeating:"b",count:64))","offsets":{"provenance":"\(bytes.count)"},"complete":\(completed)}
        """.utf8))
    }
    func append(offset: UInt64, input: Data) throws -> VideoArtifactTransferStatus {
        #expect(offset == UInt64(bytes.count))
        bytes.append(input)
        if loseReply { loseReply = false; throw URLError(.networkConnectionLost) }
        return try status()
    }
    func finish() throws -> VideoArtifactTransferStatus { completed = true; return try status() }
}

struct VideoArtifactRelayTests {
    func offer(_ archive: String, count: Int) throws -> RetainedSourceMedia.TransferOffer {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("""
        {"video_job":{"id":"parent","source_frames":"0","total_frames":"144","artifacts":{"version":1,"bundle_id":"\(String(repeating:"d",count:64))","provenance":{"sha256":"\(String(repeating:"c",count:64))","size_bytes":"\(count)"}}}}
        """.utf8))
        var offer = RetainedSourceMedia.TransferOffer(archiveIdentitySha256: String(repeating:archive,count:64), members: [],
            outputSha256: String(repeating:"e",count:64), outputSizeBytes: 3, metadata: metadata)
        offer.videoArtifactsComplete = true
        return offer
    }
    @Test func committedChunkWithLostReplyResumesAtExactOffsetAndCompletedRetryMovesNoBytes() async throws {
        let count = 4 * 1024 * 1024 + 3
        let original = try offer("a",count:count), destination = try offer("b",count:count)
        let source = FakeBackend(), target = FakeBackend(), sink = ArtifactSink()
        source.stub("videoArtifactBytes") { args in
            let offset = try #require(args[2] as? UInt64)
            #expect(args[3] as? String == "source-instance")
            return Data(repeating: 7, count: min(4 * 1024 * 1024, count - Int(offset)))
        }
        target.stub("beginVideoArtifacts") { _ in try await sink.status() }
        target.stub("appendVideoArtifact") { args in
            try await sink.append(offset: #require(args[3] as? UInt64), input: #require(args[4] as? Data))
        }
        target.stub("completeVideoArtifacts") { _ in try await sink.finish() }
        let authority = RetainedSourceMedia.ArtifactTransferAuthority(sourceInstance:"source-instance",destinationInstance:"destination-instance")
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.mirrorArtifacts(source: original, destination: destination, sourceFilename:"original.mp4",
                targetFilename:"copy.mp4",origin:source,target:target,captured:authority,isCurrent:{true})
        }
        #expect(await sink.bytes.count == 4 * 1024 * 1024)
        try await RetainedSourceMedia.mirrorArtifacts(source: original, destination: destination, sourceFilename:"original.mp4",
            targetFilename:"copy.mp4",origin:source,target:target,captured:authority,isCurrent:{true})
        #expect(await sink.bytes.count == count)
        #expect(target.count("appendVideoArtifact") == 2)
        try await RetainedSourceMedia.mirrorArtifacts(source: original, destination: destination, sourceFilename:"original.mp4",
            targetFilename:"copy.mp4",origin:source,target:target,captured:authority,isCurrent:{true})
        #expect(source.count("videoArtifactBytes") == 2)
    }
    @Test func ownerChangeAfterReadPreventsTargetMutation() async throws {
        let original = try offer("a",count:3), destination = try offer("b",count:3)
        let source = FakeBackend(), target = FakeBackend(), sink = ArtifactSink()
        source.stub("videoArtifactBytes") { _ in Data([1,2,3]) }
        target.stub("beginVideoArtifacts") { _ in try await sink.status() }
        await #expect(throws: CancellationError.self) {
            try await RetainedSourceMedia.mirrorArtifacts(source: original,destination:destination,sourceFilename:"original.mp4",
                targetFilename:"copy.mp4",origin:source,target:target,
                captured:.init(sourceInstance:"source",destinationInstance:"target"),
                isCurrent:{source.count("videoArtifactBytes") == 0})
        }
        #expect(target.count("appendVideoArtifact") == 0 && target.count("completeVideoArtifacts") == 0)
    }
    @Test func missingPositiveCapabilityRefusesBeforeOutputImport() async throws {
        let original = try offer("a",count:3), source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)",returning:original)
        source.stub("capabilities()",returning:try MoldJSON.decoder.decode(Capabilities.self,from:Data("{}".utf8)))
        target.stub("capabilities()",returning:try MoldJSON.decoder.decode(Capabilities.self,from:Data("{}".utf8)))
        await #expect(throws:(any Error).self) {
            _ = try await RetainedSourceMedia.prepareMirror(for:"original.mp4",metadata:nil,from:source,to:target)
        }
        #expect(target.count("importGallery(_:)") == 0 && target.count("beginVideoArtifacts") == 0)
    }
    @Test func malformedHandleAndImpossibleOffsetsNeverStartReadingSource() async throws {
        for (handle, offset) in [("../private", "0"), (String(repeating:"f",count:64), "4")] {
            let original = try offer("a",count:3), destination = try offer("b",count:3)
            let source = FakeBackend(), target = FakeBackend()
            let invalid = try MoldJSON.decoder.decode(VideoArtifactTransferStatus.self,from:Data("""
            {"handle":"\(handle)","archive_identity_sha256":"\(String(repeating:"b",count:64))","offsets":{"provenance":"\(offset)"},"complete":false}
            """.utf8))
            target.stub("beginVideoArtifacts",returning:invalid)
            await #expect(throws:(any Error).self) {
                try await RetainedSourceMedia.mirrorArtifacts(source:original,destination:destination,sourceFilename:"original.mp4",
                    targetFilename:"copy.mp4",origin:source,target:target,
                    captured:.init(sourceInstance:"source",destinationInstance:"target"),isCurrent:{true})
            }
            #expect(source.count("videoArtifactBytes") == 0 && target.count("appendVideoArtifact") == 0)
        }
    }
    @Test func missingOrCorruptSourceAssetsAreNotACompletedCopy() async throws {
        let source = FakeBackend(), target = FakeBackend()
        var incomplete = try offer("a",count:3)
        incomplete.videoArtifactsComplete = false
        source.stub("retainedMediaTransferOffer(for:)",returning:incomplete)
        await #expect(throws:(any Error).self) {
            _ = try await RetainedSourceMedia.prepareMirror(for:"original.mp4",metadata:nil,from:source,to:target)
        }
        #expect(target.calls.isEmpty)
    }
    @Test func cachedMovieWithNoConditioningPinsStillRepairsPublicationAssets() async throws {
        var original = try offer("a",count:3)
        original.artifactAuthority = .init(sourceInstance:"source",destinationInstance:"target")
        var copied = try offer("b",count:3)
        copied.videoArtifactsComplete = false
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)",returning:original)
        let initialCopy = copied
        target.stub("retainedMediaTransferOffer(for:)") { _ in
            var answer = initialCopy
            answer.videoArtifactsComplete = target.count("retainedMediaTransferOffer(for:)") > 1
            return answer
        }
        let complete = try MoldJSON.decoder.decode(VideoArtifactTransferStatus.self,from:Data("""
        {"handle":"\(String(repeating:"f",count:64))","archive_identity_sha256":"\(String(repeating:"b",count:64))","offsets":{"provenance":"3"},"complete":true}
        """.utf8))
        target.stub("beginVideoArtifacts",returning:complete)
        let result = try await RetainedSourceMedia.mirrorSources(for:"original.mp4",metadata:nil,from:source,to:target,
            as:"copy.mp4",capturedSourceOffer:original)
        #expect(result == .complete)
        #expect(target.count("beginVideoArtifacts") == 1)
        #expect(target.count("importRetainedMedia(_:for:)") == 0)
    }
    @Test func oldSourceCannotDowngradeArtifactMovieToFinishedFileOnly() async throws {
        let original = try offer("a",count:3), source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)",throwing:MoldClientError.http(status:404,code:nil,message:nil))
        source.stub("retainedSourceMedia(for:)",returning:RetainedSourceMedia.Inventory(availability:.unavailableLegacy))
        await #expect(throws:(any Error).self) {
            _ = try await RetainedSourceMedia.prepareMirror(for:"original.mp4",metadata:original.metadata,from:source,to:target)
        }
        #expect(target.calls.isEmpty)
    }
    @Test func completedUploadWithoutFreshDestinationEvidenceNeverCompletesMirror() async throws {
        for complete in [false, true] {
            var original = try offer("a",count:3)
            original.artifactAuthority = .init(sourceInstance:"source",destinationInstance:"target")
            let destination = try offer("b",count:3)
            let source = FakeBackend(), target = FakeBackend()
            source.stub("retainedMediaTransferOffer(for:)",returning:original)
            target.stub("retainedMediaTransferOffer(for:)") { _ in
                var answer = destination
                if target.count("retainedMediaTransferOffer(for:)") > 1 {
                    answer.videoArtifactsComplete = complete
                    if complete { // Receiver reports assets, but exact archive was replaced.
                        return RetainedSourceMedia.TransferOffer(archiveIdentitySha256:String(repeating:"c",count:64),
                            members:[],outputSha256:answer.outputSha256,outputSizeBytes:answer.outputSizeBytes,metadata:answer.metadata)
                    }
                }
                return answer
            }
            target.stub("beginVideoArtifacts",returning:try MoldJSON.decoder.decode(VideoArtifactTransferStatus.self,from:Data("""
            {"handle":"\(String(repeating:"f",count:64))","archive_identity_sha256":"\(String(repeating:"b",count:64))","offsets":{"provenance":"3"},"complete":true}
            """.utf8)))
            await #expect(throws:(any Error).self) {
                _ = try await RetainedSourceMedia.mirrorSources(for:"original.mp4",metadata:nil,from:source,to:target,
                    as:"copy.mp4",capturedSourceOffer:original)
            }
        }
    }
}
