import Foundation
import MoldClientTesting
import Testing
@testable import MoldClient

@MainActor
struct QueueDownloadRecoveryTests {
    nonisolated private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try MoldJSON.decoder.decode(type, from: Data(json.utf8))
    }

    @Test func pressIsVisibleSynchronouslyAndRepeatedPressIsIgnored() async throws {
        let recovery = QueueDownloadRecovery()
        let fake = FakeBackend()
        let host = UUID()
        let entry = try decode(QueueEntry.self, #"{"id":"q","state":"held","model":"wan","batch_id":"b","client_batch_id":"c"}"#)
        let authority = try #require(entry.authority(instanceId: "instance"))
        let gate = AsyncStream<Void>.makeStream()
        fake.stub("status()") { _ in
            for await _ in gate.stream { break }
            return try self.decode(ServerStatus.self, #"{"version":"1","busy":false,"uptime_secs":0,"instance_id":"instance"}"#)
        }
        recovery.start(entry: entry, host: host, authority: authority, backend: fake)
        recovery.start(entry: entry, host: host, authority: authority, backend: fake)
        #expect(recovery.state(host: host, job: "q")?.phase == .starting)
        #expect(recovery.state(host: host, job: "q")?.isBusy == true)
        recovery.cancel(host: host, job: "q")
        gate.continuation.finish()
    }

    @Test func settlementRequiresEveryExactTicketAndIgnoresOldSameModelSuccess() throws {
        let jobs = [DownloadJob(id: "old", model: "wan", status: .completed),
                    DownloadJob(id: "primary", model: "wan", status: .completed)]
        #expect(QueueDownloadSettlement.resolve(ids: ["primary", "companion"], jobs: jobs) == .waiting)
        #expect(QueueDownloadSettlement.resolve(ids: ["new"], jobs: jobs) == .waiting)
        #expect(QueueDownloadSettlement.resolve(ids: ["primary"], jobs: jobs) == .ready)
        #expect(QueueDownloadSettlement.resolve(ids: [], jobs: jobs) == .waiting)
        #expect(QueueDownloadSettlement.resolve(ids: ["primary", "companion"], jobs: jobs + [
            DownloadJob(id: "companion", model: "encoder", status: .failed, error: "Disk full")
        ]) == .failed("Disk full"))
    }

    @Test func progressIncludesCompletedCompanionsAndUnknownSizesStayIndeterminate() {
        let completed = DownloadJob(id: "encoder", model: "encoder", status: .completed, bytesDone: 100, bytesTotal: 100)
        let active = DownloadJob(id: "primary", model: "wan", status: .active, bytesDone: 25, bytesTotal: 100)
        #expect(QueueDownloadSettlement.progress(jobs: [completed, active]).fraction == 0.625)
        #expect(QueueDownloadSettlement.progress(jobs: [completed, DownloadJob(id: "new", model: "wan", status: .active)]).fraction == nil)
    }

    @Test func knownMissingModelUsesReadableCopyAndOtherErrorsStayIntact() {
        #expect(QueueHold.missingModel("wan", sentence: "Run: mold pull wan").summary(modelName: "Wan", hostName: "Plato") == "Wan isn’t installed on Plato.")
        #expect(QueueHold.prose("Disk full", retryable: false).summary(modelName: "Wan", hostName: "Plato") == "Disk full")
    }
}

@MainActor
struct QueueDownloadRecoveryLifecycleTests {
    nonisolated private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try MoldJSON.decoder.decode(type, from: Data(json.utf8))
    }
    private func fixture() throws -> (QueueDownloadRecovery, FakeBackend, UUID, QueueEntry, QueueAuthority) {
        let fake = FakeBackend()
        let entry = try decode(QueueEntry.self, #"{"id":"q","state":"held","model":"wan","batch_id":"b","client_batch_id":"c","retryable":true}"#)
        fake.stub("status()", returning: try decode(ServerStatus.self, #"{"version":"1","busy":false,"uptime_secs":0,"instance_id":"instance"}"#))
        fake.stub("queueJob(id:)", returning: try decode(QueueJobDetail.self, #"{"job":{"id":"q","state":"held","model":"wan","batch_id":"b","client_batch_id":"c","retryable":true}}"#))
        fake.stub("startDownload(_:)", returning: try decode(DownloadTicket.self, #"{"id":"ticket"}"#))
        fake.stub("downloads()", returning: DownloadsListing())
        fake.stub("retryJob(_:)") { _ in () }
        return (QueueDownloadRecovery(), fake, UUID(), entry, entry.authority(instanceId: "instance")!)
    }
    private func settle(_ condition: () -> Bool) async throws {
        for _ in 0..<200 where !condition() { try await Task.sleep(for: .milliseconds(5)) }
        #expect(condition())
    }
    @Test func currentTicketSuccessRetriesOnceAndFailureNeverRetries() async throws {
        for success in [true, false] {
            let (recovery, fake, host, entry, authority) = try fixture()
            recovery.start(entry: entry, host: host, authority: authority, backend: fake, every: .milliseconds(5))
            try await settle { fake.count("downloads()") > 0 }
            recovery.observe(try decode(DownloadEvent.self, success ? #"{"type":"job_done","id":"ticket","model":"wan"}"# : #"{"type":"job_failed","id":"ticket","model":"wan","error":"Disk full"}"#), on: host)
            try await settle { recovery.state(host: host, job: "q")?.isBusy == false }
            #expect(fake.count("retryJob(_:)") == (success ? 1 : 0))
            #expect(recovery.state(host: host, job: "q")?.phase == (success ? .complete : .failed))
        }
    }
    @Test func licenceDismissalAndApprovalSettleTheExactAttempt() async throws {
        for approve in [true, false] {
            let (recovery, fake, host, entry, authority) = try fixture()
            let refusal = LicenseRefusal(id: "terms", name: "Terms", url: "https://example.com", canonical: "https://example.com", sha256: "abc", summary: "Terms")
            fake.stub("startDownload(_:)") { _ in throw MoldClientError.licenseRequired(refusal, mismatch: false) }
            var approved: (@MainActor () -> Void)?
            recovery.start(entry: entry, host: host, authority: authority, backend: fake, every: .milliseconds(5), license: { _, _, resume in approved = resume; return true })
            try await settle { recovery.state(host: host, job: "q")?.phase == .license }
            #expect(fake.count("retryJob(_:)") == 0)
            if approve {
                fake.stub("startDownload(_:)", returning: try decode(DownloadTicket.self, #"{"id":"ticket"}"#))
                approved?()
                try await settle { fake.count("downloads()") > 0 }
                recovery.observe(try decode(DownloadEvent.self, #"{"type":"job_done","id":"ticket","model":"wan"}"#), on: host)
                try await settle { recovery.state(host: host, job: "q")?.phase == .complete }
                #expect(fake.count("retryJob(_:)") == 1)
            } else {
                recovery.cancel(host: host, job: "q")
                approved?()
                #expect(recovery.state(host: host, job: "q")?.phase == .cancelled)
                #expect(fake.count("retryJob(_:)") == 0)
            }
        }
    }
    @Test func changedJobAndChangedServerCannotRetry() async throws {
        let (recovery, fake, host, entry, authority) = try fixture()
        var current = true
        recovery.start(entry: entry, host: host, authority: authority, backend: fake, every: .milliseconds(5), isCurrent: { current })
        try await settle { fake.count("downloads()") > 0 }
        current = false
        recovery.observe(try decode(DownloadEvent.self, #"{"type":"job_done","id":"ticket","model":"wan"}"#), on: host)
        try await settle { recovery.state(host: host, job: "q")?.phase == .failed }
        #expect(fake.count("retryJob(_:)") == 0)
        fake.stub("status()", returning: try decode(ServerStatus.self, #"{"version":"1","busy":false,"uptime_secs":0,"instance_id":"replacement"}"#))
        recovery.start(entry: entry, host: host, authority: authority, backend: fake, every: .milliseconds(5))
        try await settle { recovery.state(host: host, job: "q")?.phase == .failed }
        #expect(fake.count("startDownload(_:)") == 1)
    }
    @Test func cancellationDuringFinalIdentityReadCannotRetry() async throws {
        let (recovery, fake, host, entry, authority) = try fixture()
        let status = try decode(ServerStatus.self, #"{"version":"1","busy":false,"uptime_secs":0,"instance_id":"instance"}"#)
        let gate = AsyncStream<Void>.makeStream()
        fake.stub("status()") { _ in
            if fake.count("status()") == 5 { for await _ in gate.stream { break } }
            return status
        }
        fake.stub("downloads()", returning: DownloadsListing(history: [DownloadJob(id: "ticket", model: "wan", status: .completed)]))
        recovery.start(entry: entry, host: host, authority: authority, backend: fake, every: .milliseconds(5))
        try await settle { fake.count("status()") == 5 }
        recovery.cancel(host: host, job: "q")
        gate.continuation.yield(()); gate.continuation.finish()
        try await Task.sleep(for: .milliseconds(30))
        #expect(fake.count("retryJob(_:)") == 0)
        #expect(recovery.state(host: host, job: "q")?.phase == .cancelled)
    }
    @Test func ambiguousRetryIsNeverSentTwice() async throws {
        let (recovery, fake, host, entry, authority) = try fixture()
        fake.stub("downloads()", returning: DownloadsListing(history: [DownloadJob(id: "ticket", model: "wan", status: .completed)]))
        fake.stub("retryJob(_:)") { _ in throw URLError(.timedOut) }
        recovery.start(entry: entry, host: host, authority: authority, backend: fake, every: .milliseconds(5))
        try await settle { recovery.state(host: host, job: "q")?.phase == .failed }
        #expect(fake.count("retryJob(_:)") == 1)
    }
}
