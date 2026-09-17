import Foundation
import Testing

@testable import MoldClient

// The FIRST frame every `/api/downloads/stream` subscriber gets
// (`types.rs:13079-13081`). It is how this app learns about jobs it did not
// start -- a `mold pull` at a terminal, the web app on the same machine --
// and `downloadEvents` decodes it with `try?` and `continue`s, so a shape
// change makes every snapshot vanish and the popover permanently empty for
// anyone else's downloads. Silently. `StreamTests` only ever fed it a
// `job_progress`-shaped frame.

private final class SnapshotTransport: StubTransport {
    nonisolated(unsafe) static var responses: [String: (status: Int, body: Data)] = [:]
    override class func response(for path: String) -> (status: Int, body: Data)? {
        responses[path]
    }
}

private let job =
    #"{"id":"d1","model":"flux-dev:q8","status":"active","files_done":1,"files_total":3,"#
    + #""bytes_done":10,"bytes_total":30,"current_file":"transformer.safetensors","#
    + #""started_at":1700000000}"#

/// One `event: download` frame, framed exactly as mold sends it.
private func frame(_ data: String) -> (status: Int, body: Data) {
    (200, Data("event: download\r\ndata: \(data)\r\n\r\n".utf8))
}

private func events(_ data: String) async throws -> [DownloadEvent] {
    SnapshotTransport.responses["/api/downloads/stream"] = frame(data)
    var received: [DownloadEvent] = []
    for try await event in SnapshotTransport.backend().downloadEvents() {
        received.append(event)
    }
    return received
}

@Suite(.serialized)
struct DownloadSnapshotTests {

    @Test func theFirstFrameCarriesEveryJobThisMachineIsRunning() async throws {
        let received = try await events(
            #"{"type":"snapshot","listing":{"active_jobs":["# + job
                + #"],"queued":[],"history":[]}}"#)
        #expect(received.count == 1)
        let listing = try #require(received.first?.listing)
        #expect(listing.activeJobs.count == 1)
        #expect(listing.activeJobs[0].id == "d1")
        #expect(listing.activeJobs[0].model == "flux-dev:q8")
        #expect(listing.activeJobs[0].status == .active)
        #expect(listing.activeJobs[0].bytesTotal == 30)
        #expect(listing.queued.isEmpty)
        #expect(listing.history.isEmpty)
    }

    /// `active` is the compatibility view of the first job and `active_jobs`
    /// is what a new client reads -- an older host sends one, a current one
    /// sends both, and either has to decode.
    @Test func anOlderHostsCompatibilityFrameStillDecodes() async throws {
        let received = try await events(
            #"{"type":"snapshot","listing":{"active":"# + job
                + #","queued":[],"history":[]}}"#)
        let listing = try #require(received.first?.listing)
        #expect(listing.active?.id == "d1")
        #expect(listing.activeJobs.isEmpty)
    }

    /// A status this build has never heard of degrades the one field rather
    /// than losing the whole snapshot -- the `OpenWireEnum` rule, which is
    /// exactly what a `try?`-and-`continue` decode would otherwise hide.
    @Test func aStatusThisBuildCannotSpellKeepsTheRestOfTheSnapshot() async throws {
        let received = try await events(
            #"{"type":"snapshot","listing":{"active_jobs":[],"queued":[],"history":"#
                + #"[{"id":"d2","model":"m","status":"verifying","files_done":3,"#
                + #""files_total":3,"bytes_done":30,"bytes_total":30}]}}"#)
        let listing = try #require(received.first?.listing)
        #expect(listing.history.count == 1)
        #expect(listing.history[0].status == .unknown)
        #expect(listing.history[0].id == "d2")
    }

    /// An ordinary delta carries no listing at all, which is the difference
    /// `DownloadStore` routes on.
    @Test func aProgressFrameCarriesNoListing() async throws {
        let received = try await events(
            #"{"type":"job_progress","id":"d1","bytes_done":20,"bytes_total":30}"#)
        #expect(received.first?.listing == nil)
        #expect(received.first?.fraction == 20.0 / 30.0)
        #expect(received.first?.isTerminal == false)
    }
}
