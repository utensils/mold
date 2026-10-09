import Foundation
import Testing
@testable import MoldClient

struct QueueFailureDetailsTests {
    @Test func preservesMachineDiagnosticsSeparatelyFromFriendlySummary() throws {
        let entry = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"h","state":"held","held_reason":"The render failed.","error_detail":"CUDA_ERROR_ILLEGAL_ADDRESS in attention"}"#.utf8))
        #expect(QueueFailureDetails.diagnostic(entry) == "CUDA_ERROR_ILLEGAL_ADDRESS in attention")
        #expect(QueueFailureDetails.copyText(entry, machine: "hal9000").contains("Job: h"))
    }
    @Test func olderThinHeldRowsKeepTheirAvailableReason() throws {
        let entry = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"h","state":"held","held_reason":"Model file is missing"}"#.utf8))
        #expect(QueueFailureDetails.diagnostic(entry) == "Model file is missing")
    }
    @Test func retryClearsStaleFailureDetails() throws {
        let entry = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"h","state":"queued","error_detail":"stale failure"}"#.utf8))
        #expect(QueueFailureDetails.diagnostic(entry) == nil)
    }
}
