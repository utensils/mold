import Foundation
import Testing

@testable import MoldClient

/// What this app believes a machine is doing, between two answers.
///
/// `activity-hal9000.json` is `GET /api/activity` on hal9000 (mold 0.29.0,
/// b015496e, captured 2026-09-17, keyless) -- an idle machine, which is the
/// shape that has to decode too. The rows below are built from the wire
/// shape in `crates/mold-core/src/types.rs:5030-5087` and the kinds
/// `routes_activity.rs:180-396` emits, because the captured machine had
/// nothing running and this lane does not write to a live host.
///
/// **Fails today**: nothing in this app reads `/api/activity`.
@MainActor
struct ActivityReconcileTests {

    /// One row, as the wire writes it. Every fixture in this bundle is built
    /// by decoding JSON, so the row and the snapshot below share one string.
    private func itemJSON(_ id: String, kind: String = "generation",
                          phase: String = "running", execution: String? = nil,
                          created: Int = 1_000, current: Int? = nil, total: Int? = nil,
                          stage: String? = nil, canCancel: Bool = true) -> String {
        #"""
        {"id": "\#(id)", "kind": "\#(kind)", "phase": "\#(phase)",
         "execution": \#(execution.map { "\"\($0)\"" } ?? "null"),
         "created_at_unix_ms": \#(created), "updated_at_unix_ms": \#(created),
         "current": \#(current.map { "\($0)" } ?? "null"),
         "total": \#(total.map { "\($0)" } ?? "null"),
         "stage": \#(stage.map { "\"\($0)\"" } ?? "null"),
         "can_cancel": \#(canCancel)}
        """#
    }

    private func item(_ id: String, kind: String = "generation", phase: String = "running",
                      execution: String? = nil, created: Int = 1_000,
                      current: Int? = nil, total: Int? = nil, stage: String? = nil,
                      canCancel: Bool = true) -> ActiveWorkItem {
        let json = itemJSON(id, kind: kind, phase: phase, execution: execution,
                            created: created, current: current, total: total,
                            stage: stage, canCancel: canCancel)
        return try! MoldJSON.decoder.decode(ActiveWorkItem.self, from: Data(json.utf8))
    }

    private func snapshot(_ items: [String], instance: String = "inst-1",
                          unavailable: [String] = []) -> ActiveWorkSnapshot {
        let kinds = unavailable.map { "\"\($0)\"" }.joined(separator: ",")
        let json = #"""
        {"instance_id": "\#(instance)", "observed_at_unix_ms": 9,
         "items": [\#(items.joined(separator: ","))],
         "unavailable_kinds": [\#(kinds)]}
        """#
        return try! MoldJSON.decoder.decode(ActiveWorkSnapshot.self, from: Data(json.utf8))
    }

    /// Two machine identities. `MoldHost.ID` is a UUID, so a test names them
    /// once rather than spelling one out per assertion.
    private let hostA = UUID()
    private let hostB = UUID()

    // MARK: - Decoding

    @Test func anIdleMachinesOwnAnswerDecodes() throws {
        let snapshot = try MoldJSON.decoder.decode(
            ActiveWorkSnapshot.self, from: RepoFixtures.fixture("activity-hal9000.json"))
        #expect(snapshot.instanceId == "ff00bea2-a8fc-4ffa-80c6-f5f80cfa5580")
        #expect(snapshot.items.isEmpty)
        #expect(snapshot.unavailableKinds.isEmpty, "an absent key is an empty list, not a failure")
    }

    /// A kind and a phase from a newer machine must not lose the snapshot --
    /// both are extensible strings on the wire by design.
    @Test func aKindThisBuildHasNeverHeardOfStillDecodes() throws {
        let json = #"""
        {"instance_id": "i", "observed_at_unix_ms": 1, "items": [
          {"id": "w-1", "kind": "reticulating", "phase": "splining",
           "created_at_unix_ms": 1, "updated_at_unix_ms": 1}]}
        """#
        let snapshot = try MoldJSON.decoder.decode(
            ActiveWorkSnapshot.self, from: Data(json.utf8))
        #expect(snapshot.items.first?.kind == "reticulating")
        #expect(snapshot.items.first?.canCancel == false, "absent can_cancel is a definitive no")
        #expect(snapshot.items.first?.phaseLabel == "splining")
        #expect(snapshot.items.first?.kindLabel == "Reticulating")
    }

    /// An ephemeral chain is ONE generation row, not a second kind of work --
    /// but the authority that owns it is the chain's.
    @Test func anEphemeralChainIsAGenerationOwnedByTheChainAuthority() {
        let chain = item("c-1", kind: "generation", execution: "chain")
        #expect(chain.kind == "generation")
        #expect(chain.authorityKind == "chain_generation")
        #expect(chain.kindLabel == "Generation")
        #expect(item("g-1").authorityKind == "generation")
    }

    // MARK: - Reconciling

    @Test func aGoodAnswerReplacesWholesale() {
        let previous = ActivityReconcile.host(
            routeURL: "http://workstation", previous: nil,
            result: .success(snapshot([itemJSON("a"), itemJSON("b")])))
        let next = ActivityReconcile.host(
            routeURL: "http://workstation", previous: previous,
            result: .success(snapshot([itemJSON("c")])))
        #expect(next.items.map(\.id) == ["c"])
        #expect(!next.stale)
        #expect(next.error == nil)
    }

    /// An offline machine is not evidence that its work vanished.
    @Test func aFailureKeepsTheLastVerifiedRowsAndSaysTheyAreStale() {
        let previous = ActivityReconcile.host(
            routeURL: "http://workstation", previous: nil, result: .success(snapshot([itemJSON("a")])))
        let next = ActivityReconcile.host(
            routeURL: "http://workstation", previous: previous,
            result: .failure(MoldClientError.unreachable("it is asleep")))
        #expect(next.items.map(\.id) == ["a"])
        #expect(next.stale)
        #expect(next.error?.contains("asleep") == true)
        #expect(next.instanceId == "inst-1", "the identity it last proved is remembered")
    }

    /// Unless the machine was re-pointed somewhere else -- those rows were a
    /// different box's.
    @Test func aFailureAfterTheAddressMovedKeepsNothing() {
        let previous = ActivityReconcile.host(
            routeURL: "http://workstation", previous: nil, result: .success(snapshot([itemJSON("a")])))
        let next = ActivityReconcile.host(
            routeURL: "http://socrates", previous: previous,
            result: .failure(MoldClientError.unreachable("no")))
        #expect(next.items.isEmpty)
    }

    /// An authority the machine could not READ: its rows are retained from
    /// the previous answer while every healthy kind is replaced.
    @Test func rowsOfAnUnreadableAuthorityAreRetained() {
        let previous = ActivityReconcile.host(
            routeURL: "http://workstation", previous: nil,
            result: .success(snapshot([itemJSON("g"), itemJSON("s", kind: "sequence")])))
        let next = ActivityReconcile.host(
            routeURL: "http://workstation", previous: previous,
            result: .success(snapshot([itemJSON("g2")], unavailable: ["sequence"])))
        #expect(next.items.map(\.id).sorted() == ["g2", "s"])
    }

    /// An ephemeral chain is retained under `chain_generation`, which is what
    /// the server names when it cannot read the chain tables -- reading its
    /// `kind` would keep every ORDINARY generation instead.
    @Test func anUnreadableChainAuthorityRetainsTheChainRowOnly() {
        let chain = itemJSON("c", kind: "generation", execution: "chain")
        let plain = itemJSON("g")
        let previous = ActivityReconcile.host(
            routeURL: "http://workstation", previous: nil, result: .success(snapshot([chain, plain])))
        let next = ActivityReconcile.host(
            routeURL: "http://workstation", previous: previous,
            result: .success(snapshot([], unavailable: ["chain_generation"])))
        #expect(next.items.map(\.id) == ["c"])
    }

    /// A machine that came back as a DIFFERENT box retains nothing, however
    /// its answer reads.
    @Test func aDifferentInstanceRetainsNothing() {
        let previous = ActivityReconcile.host(
            routeURL: "http://workstation", previous: nil,
            result: .success(snapshot([itemJSON("s", kind: "sequence")])))
        let next = ActivityReconcile.host(
            routeURL: "http://workstation", previous: previous,
            result: .success(snapshot([], instance: "inst-2", unavailable: ["sequence"])))
        #expect(next.items.isEmpty)
    }

    // MARK: - Merging

    /// Submission time only. A running job on one machine must not jump above
    /// an older queued job on another just because it started.
    @Test func theFleetIsOrderedBySubmissionTimeAlone() {
        let old = ActivityHostSnapshot(
            routeURL: "http://a", items: [item("old", phase: "queued", created: 10)])
        let new = ActivityHostSnapshot(
            routeURL: "http://b", items: [item("new", phase: "running", created: 20)])
        let rows = ActivityReconcile.merged([(host: hostA, snapshot: old), (host: hostB, snapshot: new)])
        #expect(rows.map(\.item.id) == ["new", "old"])
    }

    /// Equal timestamps keep the order the machines sent.
    @Test func equalTimestampsKeepTheMachinesOwnOrder() {
        let host = ActivityHostSnapshot(
            routeURL: "http://a",
            items: [item("first", created: 5), item("second", created: 5)])
        let rows = ActivityReconcile.merged([(host: hostA, snapshot: host)])
        #expect(rows.map(\.item.id) == ["first", "second"])
    }

    /// Two machines can hold the same job id.
    @Test func aRowIsIdentifiedByItsMachineToo() {
        let a = ActivityHostSnapshot(routeURL: "http://a", items: [item("same")])
        let b = ActivityHostSnapshot(routeURL: "http://b", items: [item("same")])
        let ids = ActivityReconcile.merged([(host: hostA, snapshot: a), (host: hostB, snapshot: b)])
            .map(\.id)
        #expect(Set(ids).count == 2)
    }

    /// A stale machine's rows say so, and so does a row whose own authority
    /// could not be read even on a fresh answer.
    @Test func aRowSaysWhenItIsTheLastThingHeard() {
        let host = ActivityHostSnapshot(
            routeURL: "http://a", items: [item("g"), item("s", kind: "sequence")],
            unavailableKinds: ["sequence"])
        let rows = ActivityReconcile.merged([(host: hostA, snapshot: host)])
        #expect(rows.first { $0.item.id == "s" }?.stale == true)
        #expect(rows.first { $0.item.id == "s" }?.unavailableKind == true)
        #expect(rows.first { $0.item.id == "g" }?.stale == false)
    }

    // MARK: - What a row says

    @Test func aPreparingRowNamesTheComponent() throws {
        let json = #"""
        {"id": "p", "kind": "generation", "phase": "preparing",
         "created_at_unix_ms": 1, "updated_at_unix_ms": 1,
         "preparation_progress": {"component": "t5xxl", "bytes_done": 1, "bytes_total": 2}}
        """#
        let row = try MoldJSON.decoder.decode(ActiveWorkItem.self, from: Data(json.utf8))
        #expect(row.phaseLabel == "Preparing · t5xxl")
    }

    /// Steps for a chain's stage counter, bytes for a download and for a
    /// generation loading its weights -- the one place `current`/`total` are
    /// bytes at all (`activity.ts:36-43`).
    @Test func onlyBytesAreShownAsBytes() {
        #expect(item("c", execution: "chain", current: 2, total: 5, stage: "Clip")
            .phaseLabel == "Clip · 2/5")
        #expect(item("g", phase: "running", current: 12, total: 30).phaseLabel == "running · 12/30")
        let loading = item("g", phase: "loading", current: 1_000_000, total: 4_000_000)
        #expect(loading.phaseLabel.contains("MB"))
        let download = item("d", kind: "download", phase: "downloading",
                            current: 1_000_000, total: 4_000_000)
        #expect(download.phaseLabel.contains("MB"))
    }

    /// An underscored phase is prose, never a raw identifier.
    @Test func anUnderscoredPhaseReadsAsWords() {
        #expect(item("x", phase: "warm_wait").phaseLabel == "warm wait")
    }
}
