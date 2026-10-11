import Foundation
import MoldClientTesting
import Testing
@testable import MoldClient

struct QueueTransferSourceTests {
    @Test(arguments: [401, 500, 503])
    func failedReadCannotBecomeMissing(status: Int) async {
        let backend = FakeBackend()
        let error = MoldClientError.http(status: status, code: nil, message: "unavailable")
        backend.stub("queueJob(id:)", throwing: error)
        await #expect(throws: MoldClientError.self) {
            try await QueueTransferSource.read(jobID: "j", instanceID: "source", reservedProtocol: false,
                hasPriorAttempt: true, backend: backend)
        }
    }

    @Test func timeoutAndCancellationPropagate() async {
        let backend = FakeBackend()
        backend.stub("queueJob(id:)", throwing: URLError(.timedOut))
        await #expect(throws: URLError.self) {
            try await QueueTransferSource.read(jobID: "j", instanceID: "source", reservedProtocol: false,
                hasPriorAttempt: true, backend: backend)
        }
        backend.stub("queueJob(id:)", throwing: CancellationError())
        await #expect(throws: CancellationError.self) {
            try await QueueTransferSource.read(jobID: "j", instanceID: "source", reservedProtocol: false,
                hasPriorAttempt: true, backend: backend)
        }
    }

    @Test func missingRequiresPriorDestination() async throws {
        let backend = FakeBackend()
        let error = MoldClientError.http(status: 404, code: nil, message: nil)
        backend.stub("queueJob(id:)", throwing: error)
        #expect(try await QueueTransferSource.read(jobID: "j", instanceID: "source", reservedProtocol: false,
            hasPriorAttempt: true, backend: backend) == .missing)
        await #expect(throws: MoldClientError.self) {
            try await QueueTransferSource.read(jobID: "j", instanceID: "source", reservedProtocol: false,
                hasPriorAttempt: false, backend: backend)
        }
    }

    @Test(arguments: ["running", "complete", "cancelling"])
    func existingIneligibleJobIsNotMissing(state: String) async throws {
        let backend = FakeBackend()
        let detail = try MoldJSON.decoder.decode(QueueJobDetail.self, from: Data(
            "{\"job\":{\"id\":\"j\",\"state\":\"\(state)\",\"batch_id\":\"b\",\"client_batch_id\":\"cb\"}}".utf8))
        backend.stub("queueJob(id:)", returning: detail)
        #expect(try await QueueTransferSource.read(jobID: "j", instanceID: "source", reservedProtocol: true,
            hasPriorAttempt: true, backend: backend) == .ineligible)
    }

    @Test(arguments: ["queued", "paused", "held"])
    func eligibleJobKeepsExactAuthority(state: String) async throws {
        let backend = FakeBackend()
        let detail = try MoldJSON.decoder.decode(QueueJobDetail.self, from: Data(
            "{\"job\":{\"id\":\"j\",\"state\":\"\(state)\",\"batch_id\":\"b\",\"client_batch_id\":\"cb\"}}".utf8))
        backend.stub("queueJob(id:)", returning: detail)
        #expect(try await QueueTransferSource.read(jobID: "j", instanceID: "source", reservedProtocol: true,
            hasPriorAttempt: false, backend: backend) == .eligible(.init(instanceId: "source", batchId: "b", clientBatchId: "cb", jobId: "j")))
    }
}
