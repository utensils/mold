import Foundation
import Testing
@testable import MoldClient

@Suite struct StallDiagnosticTests {
    private let diagnostic = #"{"stage":"Unknown future decode stage","last_completed_work_unix_ms":1792348110328,"elapsed_stall_ms":300000,"cancel_requested":true,"device_still_held":true}"#

    @Test func oldQueueAndProgressRemainCompatible() throws {
        let entry = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"j","state":"running"}"#.utf8))
        let progress = try MoldJSON.decoder.decode(JobProgress.self, from: Data(#"{"stage":"Decoding"}"#.utf8))
        #expect(entry.stall == nil)
        #expect(progress.stall == nil)
    }

    @Test func pendingCancellationPreservesLiveStateAndDeviceOwnership() throws {
        let entry = try MoldJSON.decoder.decode(QueueEntry.self, from: Data("{\"id\":\"j\",\"state\":\"running\",\"stall\":\(diagnostic)}".utf8))
        #expect(entry.state == .running)
        #expect(entry.state.isLive)
        #expect(entry.stall?.cancelRequested == true)
        #expect(entry.stall?.deviceStillHeld == true)
        #expect(entry.stall?.statusMessage.contains("device still held") == true)
        #expect(entry.stall?.recoveryMessage.contains("other jobs") == true)
        #expect(try MoldJSON.decoder.decode(QueueEntry.self, from: MoldJSON.encoder.encode(entry)) == entry)
    }

    @Test func authoritativeNullClearsRecoveredProgress() throws {
        var progress = try MoldJSON.decoder.decode(JobProgress.self, from: Data("{\"stall\":\(diagnostic)}".utf8))
        #expect(progress.stall != nil)
        progress = try MoldJSON.decoder.decode(JobProgress.self, from: Data(#"{"stage":"Decoding","stall":null}"#.utf8))
        #expect(progress.stall == nil)
    }

    @Test func largeFiniteDurationsAndUnknownStagesDecodeWithoutOverflow() throws {
        let value = diagnostic.replacingOccurrences(of: "300000", with: "18446744073709551615")
        let stall = try MoldJSON.decoder.decode(StallDiagnostic.self, from: Data(value.utf8))
        #expect(stall.elapsedStallMs == UInt64.max)
        #expect(!stall.statusMessage.isEmpty)
        #expect(stall.stage == "Unknown future decode stage")
    }

    @Test func malformedDiagnosticCannotInventADeviceClaim() {
        #expect(throws: (any Error).self) {
            try MoldJSON.decoder.decode(StallDiagnostic.self, from: Data(#"{"stage":"Decode","elapsed_stall_ms":-1}"#.utf8))
        }
    }

    @Test func blockedDeviceReasonComesFromMatchingPlanItemOnly() throws {
        let listing = try MoldJSON.decoder.decode(QueueListing.self, from: Data(#"{"entries":[],"plan":{"work_items":[{"work_id":"held","blocked_device_reason":"Waiting for stalled device; device still held."},{"work_id":"healthy"}]}}"#.utf8))
        #expect(listing.blockedDeviceReason(for: "held")?.contains("device still held") == true)
        #expect(listing.blockedDeviceReason(for: "healthy") == nil)
        #expect(listing.blockedDeviceReason(for: "missing") == nil)
        let old = try MoldJSON.decoder.decode(QueueListing.self, from: Data(#"{"entries":[]}"#.utf8))
        #expect(old.blockedDeviceReason(for: "held") == nil)
    }
}

extension StallDiagnosticTests {
    @Test func curatedStallThresholdMatchesServerBoundsWithoutClaimingRecovery() throws {
        let key = try #require(SettingKeys.performance.first { $0.key == "queue.no_progress_timeout_seconds" })
        #expect(key.editor == .number(min: 0, max: 86_400, step: nil))
        #expect(key.help.contains("never releases"))
    }
}

@Test func stallDiagnosticsCopyRetainsHostJobAndOwnershipFacts() {
    let value = StallDiagnostic(stage: "Decode chunks", lastCompletedWorkUnixMs: 123, elapsedStallMs: 300000, cancelRequested: true, deviceStillHeld: true)
    let text = value.copyText(machine: "Workstation", job: "job-id")
    #expect(text.contains("Machine: Workstation"))
    #expect(text.contains("Job: job-id"))
    #expect(text.contains("Last completed work (Unix ms): 123"))
    #expect(text.contains("Device still held: true"))
    #expect(text.contains("Cancel requested: true"))
}
