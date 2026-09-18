import Foundation
import MoldClient
import Testing

@testable import Mold

/// Use These Settings, past the recipe: which machine is asked what it kept,
/// what is said when nobody kept it, and how the bytes reach the render.
///
/// **Fails today**: the app never asks a host what it retained, so a reuse
/// restores a recipe with no picture in it and says nothing about why.
@MainActor
struct ReuseTests {
    private func machine(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func member(_ role: String = "source_image", _ id: String = "m1")
        -> RetainedSourceMedia.Member {
        RetainedSourceMedia.Member(memberId: id, role: role, displayName: role,
                                   sizeBytes: 4)
    }

    /// A print that recorded a source digest, so an unavailable answer about
    /// it IS worth a sentence.
    private func conditioned() -> OutputMetadata {
        decode("""
        {"prompt":"p","model":"m","seed":1,"steps":4,"guidance":1.0,"width":8,
         "height":8,"version":"0.29.0","source_image_sha256":"abc"}
        """)
    }

    /// A text-to-image print: nothing was ever conditioned on.
    private func plain() -> OutputMetadata {
        decode("""
        {"prompt":"p","model":"m","seed":1,"steps":4,"guidance":1.0,"width":8,
         "height":8,"version":"0.29.0"}
        """)
    }

    private func decode(_ json: String) -> OutputMetadata {
        try! MoldJSON.decoder.decode(OutputMetadata.self, from: Data(json.utf8))
    }

    private func request() -> GenerateRequest {
        GenerateRequest(prompt: "p", model: "m", width: 8, height: 8, steps: 4,
                        guidance: 1)
    }

    // MARK: - Asking every machine

    @Test func asksEveryKnownCopyAndKeepsTheOneThatCanActuallyHandItOver() async {
        let plato = machine("plato"), hal = machine("hal")
        let onPlato = FakeBackend(host: plato), onHal = FakeBackend(host: hal)
        // The mirror lists the print but holds no private archive for it.
        onPlato.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
        onHal.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [plato, hal]) { $0.id == plato.id ? onPlato : onHal }
        let store = ReuseStore(hosts: hosts)

        let fence = store.begin()
        await store.probe([PrintID(host: plato.id, filename: "a.png"),
                           PrintID(host: hal.id, filename: "a.png")],
                          fence: fence, disclosing: conditioned())

        #expect(store.authority?.origin == hal.id)
        #expect(store.authority?.members.count == 1)
        // Both were asked: stopping at the first blank is how a mirror hides
        // the machine that actually made the print.
        #expect(onPlato.retainedInventoryRequests == ["a.png"])
        #expect(onHal.retainedInventoryRequests == ["a.png"])
        #expect(store.notice == nil)
    }

    @Test func prefersAConcreteFailureOverAMirrorsBlank() async {
        let plato = machine("plato"), hal = machine("hal")
        let onPlato = FakeBackend(host: plato), onHal = FakeBackend(host: hal)
        onPlato.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
        onHal.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableAuth)
        let hosts = HostStore(hosts: [plato, hal]) { $0.id == plato.id ? onPlato : onHal }
        let store = ReuseStore(hosts: hosts)

        await store.probe([PrintID(host: plato.id, filename: "a.png"),
                           PrintID(host: hal.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())

        #expect(store.authority == nil)
        #expect(store.notice
            == "Connect this machine with an API key to restore its private source media.")
    }

    /// **Fails today**: a LATER copy answering a state this build cannot name
    /// replaces the concrete one already held, and `disclosure(.unknown)` is
    /// nil -- so a newer machine on the fleet silences the sentence a machine
    /// that answered plainly had already earned.
    @Test func aStateThisBuildCannotNameNeverErasesAConcreteAnswer() async {
        let plato = machine("plato"), hal = machine("hal")
        let onPlato = FakeBackend(host: plato), onHal = FakeBackend(host: hal)
        onPlato.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .unavailableMissingOrCorrupt)
        onHal.retainedInventories["a.png"] = try! MoldJSON.decoder.decode(
            RetainedSourceMedia.Inventory.self,
            from: Data(#"{"availability":"unavailable_quarantined"}"#.utf8))
        let hosts = HostStore(hosts: [plato, hal]) { $0.id == plato.id ? onPlato : onHal }
        let store = ReuseStore(hosts: hosts)

        await store.probe([PrintID(host: plato.id, filename: "a.png"),
                           PrintID(host: hal.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())

        #expect(store.notice
            == "This print\u{2019}s retained source media is missing or damaged. "
            + "Reattach it before developing.")
    }

    @Test func oneUnreachableCopyNeverHidesAReachableArchive() async {
        let plato = machine("plato"), hal = machine("hal")
        let onPlato = FakeBackend(host: plato), onHal = FakeBackend(host: hal)
        // Nothing planted: the fake throws, exactly as an unreachable machine
        // would.
        onHal.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [plato, hal]) { $0.id == plato.id ? onPlato : onHal }
        let store = ReuseStore(hosts: hosts)

        await store.probe([PrintID(host: plato.id, filename: "a.png"),
                           PrintID(host: hal.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())

        #expect(store.authority?.origin == hal.id)
        #expect(store.notice == nil)
    }

    /// The rule that keeps a picture that never had a source quiet.
    @Test func aTextToImagePrintIsToldNothingAtAll() async {
        let plato = machine("plato")
        let backend = FakeBackend(host: plato)
        backend.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let store = ReuseStore(hosts: hosts)

        await store.probe([PrintID(host: plato.id, filename: "a.png")],
                          fence: store.begin(), disclosing: plain())

        // Asked anyway -- the server is the only authority on what it kept.
        #expect(backend.retainedInventoryRequests == ["a.png"])
        #expect(store.notice == nil)
    }

    /// The SECOND reuse, while the first is still in the air. The older
    /// answer describes a print nobody is looking at, and installing it would
    /// hydrate the new render from the old print's archive.
    @Test func aSecondReuseWinsOverAProbeStillInFlight() async {
        let plato = machine("plato")
        let backend = FakeBackend(host: plato)
        backend.retainedInventories["old.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let store = ReuseStore(hosts: hosts)

        let stale = store.begin()
        _ = store.begin()  // a second Use These Settings
        await store.probe([PrintID(host: plato.id, filename: "old.png")],
                          fence: stale, disclosing: conditioned())

        #expect(store.authority == nil)
    }

    // MARK: - How the bytes reach the render

    private func hydration(origin: MoldHost.ID, hosts: HostStore,
                           members: [RetainedSourceMedia.Member])
        -> RetainedMediaHydration {
        RetainedMediaHydration(
            authority: ReuseStore.Authority(filename: "a.png", origin: origin,
                                            members: members),
            hosts: hosts)
    }

    /// Same machine, one child: a HANDLE. Nothing moves -- the host already
    /// holds the bytes -- and the handle is bound to the request going out.
    @Test func onTheMachineThatMadeItTheHostHydratesItself() async throws {
        let plato = machine("plato")
        let backend = FakeBackend(host: plato)
        backend.retainedSession = try MoldJSON.decoder.decode(
            RetainedSourceMedia.ReuseSession.self,
            from: Data(#"{"instance_id":"i","expires_at":9,"request_sha256":"s","session_handle":"handle-1"}"#.utf8))
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let admission = BatchAdmission(clientBatchId: "batch-1", requests: [request()])

        let sending = try await RetainedMedia.hydrated(
            admission,
            with: hydration(origin: plato.id, hosts: hosts, members: [member()]),
            on: plato, backend: backend)

        #expect(sending.retainedMediaSession == "handle-1")
        // Not a byte was downloaded.
        #expect(backend.retainedMemberRequests.isEmpty)
        // And it was minted against the request actually being submitted.
        let minted = try #require(backend.retainedSessionRequests.first)
        #expect(minted.members == ["m1"])
        #expect(minted.target == admission.requests[0])
        // The idempotency fence is carried, never re-minted.
        #expect(sending.clientBatchId == "batch-1")
    }

    /// Another machine: a RELAY. A session cannot travel, so the bytes do.
    @Test func onAnotherMachineTheBytesTravelAndTheHandleDoesNot() async throws {
        let plato = machine("plato"), hal = machine("hal")
        let onPlato = FakeBackend(host: plato), onHal = FakeBackend(host: hal)
        onHal.retainedMemberBytes["m1"] = Data([9, 9, 9])
        let hosts = HostStore(hosts: [plato, hal]) { $0.id == plato.id ? onPlato : onHal }

        let sending = try await RetainedMedia.hydrated(
            BatchAdmission(clientBatchId: "batch-1", requests: [request()]),
            with: hydration(origin: hal.id, hosts: hosts, members: [member()]),
            on: plato, backend: onPlato)

        #expect(sending.retainedMediaSession == nil)
        #expect(sending.requests[0].sourceImage == Data([9, 9, 9]).base64EncodedString())
        // Read from the machine that HOLDS it, never from the one it is going to.
        #expect(onHal.retainedMemberRequests == ["m1"])
        #expect(onPlato.retainedMemberRequests.isEmpty)
        #expect(onPlato.retainedSessionRequests.isEmpty)
        #expect(sending.clientBatchId == "batch-1")
    }

    /// A batch of four on the print's OWN machine. A session binds exactly one
    /// child, so four siblings take the relay rather than being refused --
    /// four copies of one render wanting the same picture is an ordinary
    /// thing to ask for.
    @Test func aBatchOfFourTakesTheRelayEvenAtHomeAndFetchesTheBytesOnce()
        async throws {
        let plato = machine("plato")
        let backend = FakeBackend(host: plato)
        backend.retainedMemberBytes["m1"] = Data([7])
        let hosts = HostStore(hosts: [plato]) { _ in backend }

        let sending = try await RetainedMedia.hydrated(
            BatchAdmission(requests: Array(repeating: request(), count: 4)),
            with: hydration(origin: plato.id, hosts: hosts, members: [member()]),
            on: plato, backend: backend)

        #expect(sending.retainedMediaSession == nil)
        #expect(sending.requests.count == 4)
        #expect(sending.requests.allSatisfy {
            $0.sourceImage == Data([7]).base64EncodedString()
        })
        // One fetch, four siblings.
        #expect(backend.retainedMemberRequests == ["m1"])
        #expect(backend.retainedSessionRequests.isEmpty)
    }

    /// A picture someone reattached by hand wins, and the retained one is not
    /// asked for at all -- so the host's own target-conflict refusal can never
    /// fire for something this client chose to send.
    @Test func aPictureSomebodyAttachedThemselvesIsNeverOverwritten() async throws {
        let plato = machine("plato")
        let backend = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        var mine = request()
        mine.sourceImage = "MINE"

        let sending = try await RetainedMedia.hydrated(
            BatchAdmission(requests: [mine]),
            with: hydration(origin: plato.id, hosts: hosts, members: [member()]),
            on: plato, backend: backend)

        #expect(sending.requests[0].sourceImage == "MINE")
        #expect(sending.retainedMediaSession == nil)
        #expect(backend.retainedSessionRequests.isEmpty)
        #expect(backend.retainedMemberRequests.isEmpty)
    }

    /// `POST /api/chain-jobs` is not one of the three doors that redeem a
    /// session, so a clip long enough to be rendered in pieces cannot be
    /// hydrated from the archive. That is SAID rather than rendered without
    /// the picture it was supposed to start from.
    @Test func aLongClipSaysWhatThatRouteCannotBringBack() async {
        let plato = machine("plato")
        let backend = FakeBackend(host: plato)
        backend.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let store = ReuseStore(hosts: hosts)

        // Nothing held yet: a chain with no retained print says nothing.
        store.warnIfTheRouteCannotCarryMedia(chained: true)
        #expect(store.notice == nil)

        await store.probe([PrintID(host: plato.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())
        #expect(store.authority != nil)
        // An ordinary render still says nothing -- it CAN carry the media.
        store.warnIfTheRouteCannotCarryMedia(chained: false)
        #expect(store.notice == nil)

        store.warnIfTheRouteCannotCarryMedia(chained: true)
        #expect(store.notice?.contains("renders it in pieces") == true)
    }
}
