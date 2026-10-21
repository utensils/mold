import Foundation
import Testing
@testable import MoldClient

struct VideoArtifactContractTests {
    @Test func publishedArtifactsKeepOpaqueIdentityAndLosslessDecimalCounts() throws {
        let raw = Data("""
        {"version":1,"bundle_id":"\(String(repeating:"a",count:64))","provenance":{"sha256":"\(String(repeating:"b",count:64))","size_bytes":"18446744073709551615"},"pcm":{"file":{"sha256":"\(String(repeating:"c",count:64))","size_bytes":"12"},"sample_rate":24000,"channels":1,"frames":"3"}}
        """.utf8)
        let artifacts = try MoldJSON.decoder.decode(VideoJobArtifacts.self, from: raw)
        #expect(artifacts.provenance.sizeBytes == UInt64.max)
        #expect(artifacts.pcm?.frames == 3)
        #expect(artifacts.bundleId == String(repeating:"a",count:64))
        let object = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(artifacts)) as? [String:Any])
        #expect((object["provenance"] as? [String:Any])?["size_bytes"] as? String == "18446744073709551615")
    }
    @Test func artifactOffsetsRemainLosslessAcrossResume() throws {
        let status = try MoldJSON.decoder.decode(VideoArtifactTransferStatus.self, from: Data("""
        {"handle":"opaque","archive_identity_sha256":"\(String(repeating:"a",count:64))","offsets":{"provenance":"18446744073709551615","pcm":"4"},"complete":false}
        """.utf8))
        #expect(status.offsets.provenance == UInt64.max)
        #expect(status.offsets.pcm == 4)
    }
    @Test func malformedPcmDescriptorsAreRefusedBeforeTransport() throws {
        for pair in [(3,24000),(1,192001),(0,24000)] {
            let raw = Data("""
            {"version":1,"bundle_id":"\(String(repeating:"a",count:64))","provenance":{"sha256":"\(String(repeating:"b",count:64))","size_bytes":"10"},"pcm":{"file":{"sha256":"\(String(repeating:"c",count:64))","size_bytes":"12"},"sample_rate":\(pair.1),"channels":\(pair.0),"frames":"3"}}
            """.utf8)
            let descriptor = try MoldJSON.decoder.decode(VideoJobArtifacts.self,from:raw)
            #expect(throws:(any Error).self) { try descriptor.validate() }
        }
    }
}
