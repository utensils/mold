import Foundation
import Testing

@testable import MoldClient

private func child(revision: UInt64?, updatedAtMs: Int64?, state: BatchChildState) -> BatchChild {
    BatchChild(index: 0, jobId: "j", state: state, error: nil, errorCode: nil,
               retryable: nil, revision: revision, updatedAtMs: updatedAtMs, result: nil)
}

@Test func revisionDecidesWhichViewIsNewer() {
    let older = child(revision: 4, updatedAtMs: 9_000, state: .held)
    let newer = child(revision: 5, updatedAtMs: 1_000, state: .accepted)
    // A retry moves a child BACKWARD in state and its timestamp can trail.
    // Ordering on the timestamp would drop the update entirely.
    #expect(newer.supersedes(older))
    #expect(!older.supersedes(newer))
}

@Test func withoutRevisionAuthorityTheTimestampDecides() {
    let older = child(revision: 0, updatedAtMs: 1_000, state: .accepted)
    let newer = child(revision: nil, updatedAtMs: 2_000, state: .running)
    #expect(newer.supersedes(older))
}

@Test func aBatchIsSettledOnlyWhenNoChildCanStillChange() {
    func status(_ states: [BatchChildState]) -> BatchStatus {
        BatchStatus(id: "b", clientBatchId: "c", instanceId: nil, durable: true,
                    children: states.enumerated().map {
                        BatchChild(index: $0.offset, jobId: "j\($0.offset)", state: $0.element,
                                   error: nil, errorCode: nil, retryable: nil, revision: 1,
                                   updatedAtMs: nil, result: nil)
                    })
    }
    #expect(status([.complete, .failed]).isSettled)
    #expect(!status([.complete, .running]).isSettled)
    // Held is live: the host is still going to run it.
    #expect(!status([.held]).isSettled)
}

@Test func decodesABatchStatusPayload() throws {
    let json = Data("""
    {"id":"b1","client_batch_id":"c1","instance_id":"i","durable":true,
     "children":[{"index":0,"job_id":"j1","state":"complete","revision":7,
                  "updated_at_ms":1789,"result":{"filename":"mold-x.png","seed":42,
                  "generation_time_ms":3800}}]}
    """.utf8)
    let status = try MoldJSON.decoder.decode(BatchStatus.self, from: json)
    #expect(status.isSettled)
    #expect(status.children[0].result?.filename == "mold-x.png")
    #expect(status.children[0].result?.seed == 42)
    #expect(status.children[0].revision == 7)
}
