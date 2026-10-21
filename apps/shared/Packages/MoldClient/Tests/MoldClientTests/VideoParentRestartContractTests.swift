import Foundation
import Testing

@testable import MoldClient

struct VideoParentRestartContractTests {
    @Test func opaqueExportPreservesFullSeedAndLiteralBackslashes() throws {
        let raw = #"{"generation":{"seed":18446744073709551615,"prompt":"literal\\new"},"duration_ms":"30000"}"#
        let wire = try JSONSerialization.data(withJSONObject: [
            "source_instance_id": "original", "source_revision": "revision", "destination_instance_id": "target",
            "request_json": raw,
        ])
        let exported = try MoldJSON.decoder.decode(VideoParentRestartExport.self, from: wire)
        let request = ChainRestartExportRequest(
            sourceInstanceId: "original", sourceRevision: "revision", destinationInstanceId: "target")
        #expect(try exported.validated(for: request) == raw)
        #expect(throws: (any Error).self) {
            try exported.validated(
                for: .init(sourceInstanceId: "original", sourceRevision: "newer", destinationInstanceId: "target"))
        }
    }
    @Test func sameInstancePrefixGrantRequiresExactHashSizeAndBothOwnerBindings() throws {
        let prefix: [String: Any] = [
            "filename": "original.mp4", "archive_identity_sha256": String(repeating: "a", count: 64),
            "sha256": String(repeating: "b", count: 64), "size_bytes": "18446744073709551615", "source_ref": "ref",
            "upload_spec_json": #"{"metadata":{"seed":18446744073709551615}}"#,
            "retained_upload": [
                "handle": "upload", "sha256": String(repeating: "b", count: 64), "size_bytes": "18446744073709551615",
            ],
        ]
        func exported(source: String = "same", destination: String = "same", changedHash: Bool = false) throws
            -> VideoParentRestartExport
        {
            var value = prefix
            if changedHash {
                value["retained_upload"] = [
                    "handle": "upload", "sha256": String(repeating: "c", count: 64),
                    "size_bytes": "18446744073709551615",
                ]
            }
            return try MoldJSON.decoder.decode(
                VideoParentRestartExport.self,
                from: JSONSerialization.data(withJSONObject: [
                    "source_instance_id": source, "source_revision": "revision", "destination_instance_id": destination,
                    "request_json": "{}", "prefix": value,
                ]))
        }
        let same = try exported()
        #expect(try same.localPrefixUpload(destinationInstanceId: "same")?.sizeBytes == UInt64.max)
        #expect(throws: (any Error).self) { try same.localPrefixUpload(destinationInstanceId: "replacement") }
        #expect(throws: (any Error).self) {
            try exported(source: "source", destination: "target").localPrefixUpload(destinationInstanceId: "target")
        }
        #expect(throws: (any Error).self) {
            try exported(changedHash: true).localPrefixUpload(destinationInstanceId: "same")
        }
    }
}
