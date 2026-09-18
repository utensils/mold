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
        let workstation = machine("workstation"), hal = machine("hal")
        let onWorkstation = FakeBackend(host: workstation), onHal = FakeBackend(host: hal)
        // The mirror lists the print but holds no private archive for it.
        onWorkstation.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
        onHal.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [workstation, hal]) { $0.id == workstation.id ? onWorkstation : onHal }
        let store = ReuseStore(hosts: hosts)

        let fence = store.begin()
        await store.probe([PrintID(host: workstation.id, filename: "a.png"),
                           PrintID(host: hal.id, filename: "a.png")],
                          fence: fence, disclosing: conditioned())

        #expect(store.authority?.origin == hal.id)
        #expect(store.authority?.members.count == 1)
        // Both were asked: stopping at the first blank is how a mirror hides
        // the machine that actually made the print.
        #expect(onWorkstation.retainedInventoryRequests == ["a.png"])
        #expect(onHal.retainedInventoryRequests == ["a.png"])
        #expect(store.notice == nil)
    }

    @Test func prefersAConcreteFailureOverAMirrorsBlank() async {
        let workstation = machine("workstation"), hal = machine("hal")
        let onWorkstation = FakeBackend(host: workstation), onHal = FakeBackend(host: hal)
        onWorkstation.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
        onHal.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableAuth)
        let hosts = HostStore(hosts: [workstation, hal]) { $0.id == workstation.id ? onWorkstation : onHal }
        let store = ReuseStore(hosts: hosts)

        await store.probe([PrintID(host: workstation.id, filename: "a.png"),
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
        let workstation = machine("workstation"), hal = machine("hal")
        let onWorkstation = FakeBackend(host: workstation), onHal = FakeBackend(host: hal)
        onWorkstation.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .unavailableMissingOrCorrupt)
        onHal.retainedInventories["a.png"] = try! MoldJSON.decoder.decode(
            RetainedSourceMedia.Inventory.self,
            from: Data(#"{"availability":"unavailable_quarantined"}"#.utf8))
        let hosts = HostStore(hosts: [workstation, hal]) { $0.id == workstation.id ? onWorkstation : onHal }
        let store = ReuseStore(hosts: hosts)

        await store.probe([PrintID(host: workstation.id, filename: "a.png"),
                           PrintID(host: hal.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())

        #expect(store.notice
            == "This print\u{2019}s retained source media is missing or damaged. "
            + "Reattach it before developing.")
    }

    @Test func oneUnreachableCopyNeverHidesAReachableArchive() async {
        let workstation = machine("workstation"), hal = machine("hal")
        let onWorkstation = FakeBackend(host: workstation), onHal = FakeBackend(host: hal)
        // Nothing planted: the fake throws, exactly as an unreachable machine
        // would.
        onHal.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [workstation, hal]) { $0.id == workstation.id ? onWorkstation : onHal }
        let store = ReuseStore(hosts: hosts)

        await store.probe([PrintID(host: workstation.id, filename: "a.png"),
                           PrintID(host: hal.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())

        #expect(store.authority?.origin == hal.id)
        #expect(store.notice == nil)
    }

    /// The rule that keeps a picture that never had a source quiet.
    @Test func aTextToImagePrintIsToldNothingAtAll() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)

        await store.probe([PrintID(host: workstation.id, filename: "a.png")],
                          fence: store.begin(), disclosing: plain())

        // Asked anyway -- the server is the only authority on what it kept.
        #expect(backend.retainedInventoryRequests == ["a.png"])
        #expect(store.notice == nil)
    }

    /// The SECOND reuse, while the first is still in the air. The older
    /// answer describes a print nobody is looking at, and installing it would
    /// hydrate the new render from the old print's archive.
    @Test func aSecondReuseWinsOverAProbeStillInFlight() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedInventories["old.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)

        let stale = store.begin()
        _ = store.begin()  // a second Use These Settings
        await store.probe([PrintID(host: workstation.id, filename: "old.png")],
                          fence: stale, disclosing: conditioned())

        #expect(store.authority == nil)
    }

    // MARK: - Putting the authority down again

    /// Arms a store as a real reuse does: probe, then record the draft the
    /// recipe landed in.
    private func armed(_ store: ReuseStore, backend: FakeBackend, host: MoldHost,
                       draft: RenderDraft) async {
        backend.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        await store.probe([PrintID(host: host.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())
        store.arm(draft)
    }

    /// **Fails today**: nothing ever puts the authority down, so reuse print
    /// A, retype the prompt, pick another model, Develop -- and the host
    /// hydrates A's picture into a render that has nothing to do with it,
    /// with nothing on screen having said so.
    @Test func aDraftThatHasMovedOnNoLongerCarriesThePrintsPicture() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)
        var draft = RenderDraft()
        draft.prompt = "the print's own prompt"
        await armed(store, backend: backend, host: workstation, draft: draft)
        #expect(store.pending(for: draft) != nil)

        draft.prompt = "something else entirely"
        #expect(store.pending(for: draft) == nil)
        #expect(store.take(for: draft) == nil)
        // And it is GONE, not merely hidden: the next press cannot find it.
        #expect(store.authority == nil)
    }

    /// **Fails today**: the authority outlives the render that used it, so
    /// every later Develop silently conditions on the same print.
    @Test func theSubmitThatTakesTheAuthorityIsTheLastOneToHaveIt() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)
        let draft = RenderDraft()
        await armed(store, backend: backend, host: workstation, draft: draft)

        #expect(store.take(for: draft) != nil)
        #expect(store.take(for: draft) == nil)
        #expect(store.authority == nil)
    }

    /// A press with nothing pending must not bump the fence under a probe
    /// still in the air, nor wipe a sentence nobody has read yet.
    @Test func aPressWithNothingPendingDisturbsNothing() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedInventories["a.png"] =
            RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)
        await store.probe([PrintID(host: workstation.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())
        #expect(store.notice != nil)

        #expect(store.take(for: RenderDraft()) == nil)
        #expect(store.notice != nil)
    }

    /// **Fails today**: Sequence B. Reuse a print, delete it or let its Mac
    /// sleep, and EVERY later Develop fails identically -- forever, with no
    /// affordance to clear it and nothing calling `clear()`.
    @Test func aPrintTheMachineCanNoLongerHonourNeverRefusesASecondRender() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)
        let draft = RenderDraft()
        await armed(store, backend: backend, host: workstation, draft: draft)

        // The render that took it fails: nothing was planted, so the mint
        // throws exactly as a purged print's 409 would.
        let taken = try? #require(store.take(for: draft))
        await #expect(throws: (any Error).self) {
            _ = try await RetainedMedia.hydrated(
                BatchAdmission(requests: [self.request()]),
                with: RetainedMediaHydration(authority: taken!, hosts: hosts),
                on: workstation, backend: backend)
        }
        // The next press has nothing to fail on.
        #expect(store.take(for: draft) == nil)
        #expect(store.authority == nil)
    }

    /// The person can put it down themselves, from the sentence that names it.
    @Test func theAttachmentSaysWhatItIsAndCanBeRemoved() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)
        let draft = RenderDraft()
        await armed(store, backend: backend, host: workstation, draft: draft)

        let sentence = try? #require(store.attachmentSentence(for: draft))
        #expect(sentence?.contains("a.png") == true)
        #expect(sentence?.contains("workstation") == true)
        store.clear()
        #expect(store.attachmentSentence(for: draft) == nil)
        #expect(store.take(for: draft) == nil)
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
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedSession = try MoldJSON.decoder.decode(
            RetainedSourceMedia.ReuseSession.self,
            from: Data(#"{"instance_id":"i","expires_at":9,"request_sha256":"s","session_handle":"handle-1"}"#.utf8))
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let admission = BatchAdmission(clientBatchId: "batch-1", requests: [request()])

        let sending = try await RetainedMedia.hydrated(
            admission,
            with: hydration(origin: workstation.id, hosts: hosts, members: [member()]),
            on: workstation, backend: backend)

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
        let workstation = machine("workstation"), hal = machine("hal")
        let onWorkstation = FakeBackend(host: workstation), onHal = FakeBackend(host: hal)
        onHal.retainedMemberBytes["m1"] = Data([9, 9, 9])
        let hosts = HostStore(hosts: [workstation, hal]) { $0.id == workstation.id ? onWorkstation : onHal }

        let sending = try await RetainedMedia.hydrated(
            BatchAdmission(clientBatchId: "batch-1", requests: [request()]),
            with: hydration(origin: hal.id, hosts: hosts, members: [member()]),
            on: workstation, backend: onWorkstation)

        #expect(sending.retainedMediaSession == nil)
        #expect(sending.requests[0].sourceImage == Data([9, 9, 9]).base64EncodedString())
        // Read from the machine that HOLDS it, never from the one it is going to.
        #expect(onHal.retainedMemberRequests == ["m1"])
        #expect(onWorkstation.retainedMemberRequests.isEmpty)
        #expect(onWorkstation.retainedSessionRequests.isEmpty)
        #expect(sending.clientBatchId == "batch-1")
    }

    /// A batch of four on the print's OWN machine. A session binds exactly one
    /// child, so four siblings take the relay rather than being refused --
    /// four copies of one render wanting the same picture is an ordinary
    /// thing to ask for.
    @Test func aBatchOfFourTakesTheRelayEvenAtHomeAndFetchesTheBytesOnce()
        async throws {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedMemberBytes["m1"] = Data([7])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }

        let sending = try await RetainedMedia.hydrated(
            BatchAdmission(requests: Array(repeating: request(), count: 4)),
            with: hydration(origin: workstation.id, hosts: hosts, members: [member()]),
            on: workstation, backend: backend)

        #expect(sending.retainedMediaSession == nil)
        #expect(sending.requests.count == 4)
        #expect(sending.requests.allSatisfy {
            $0.sourceImage == Data([7]).base64EncodedString()
        })
        // One fetch, four siblings.
        #expect(backend.retainedMemberRequests == ["m1"])
        #expect(backend.retainedSessionRequests.isEmpty)
    }

    /// **Fails today**: a mint that answers `_ARCHIVE_CHANGED` -- the print
    /// re-published between the probe and the submit, which is exactly the
    /// transient the 120 s TTL exists for -- kills the render outright.
    @Test func aHandleThatWentStaleIsMintedOnceMore() async throws {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedSessionFailures = [
            MoldClientError.http(status: 409,
                                 code: "RETAINED_MEDIA_REUSE_ARCHIVE_CHANGED",
                                 message: "gallery item identity changed"),
        ]
        backend.retainedSession = try MoldJSON.decoder.decode(
            RetainedSourceMedia.ReuseSession.self,
            from: Data(#"{"instance_id":"i","expires_at":9,"request_sha256":"s","session_handle":"second"}"#.utf8))
        let hosts = HostStore(hosts: [workstation]) { _ in backend }

        let sending = try await RetainedMedia.hydrated(
            BatchAdmission(requests: [request()]),
            with: hydration(origin: workstation.id, hosts: hosts, members: [member()]),
            on: workstation, backend: backend)

        #expect(sending.retainedMediaSession == "second")
        #expect(backend.retainedSessionRequests.count == 2)
    }

    /// And when the second mint fails too, the bytes are on this very machine
    /// -- so carry them rather than refuse a render it can obviously make.
    @Test func aSessionThatKeepsFailingFallsBackToTheBytes() async throws {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        let stale = MoldClientError.http(
            status: 409, code: "RETAINED_MEDIA_REUSE_ARCHIVE_CHANGED", message: nil)
        backend.retainedSessionFailures = [stale, stale]
        backend.retainedMemberBytes["m1"] = Data([5])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }

        let sending = try await RetainedMedia.hydrated(
            BatchAdmission(requests: [request()]),
            with: hydration(origin: workstation.id, hosts: hosts, members: [member()]),
            on: workstation, backend: backend)

        #expect(sending.retainedMediaSession == nil)
        #expect(sending.requests[0].sourceImage == Data([5]).base64EncodedString())
    }

    /// A refusal that describes the ARCHIVE is not asked twice, and it
    /// reaches the pane as this app's sentence with the way forward in it --
    /// never the host's own API prose.
    @Test func aSettledRefusalIsSaidOnceInThisAppsWords() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedSessionFailures = [
            MoldClientError.http(
                status: 409, code: "RETAINED_SOURCE_MEDIA_UNAVAILABLE",
                message: "retained source media is unavailable"),
        ]
        let hosts = HostStore(hosts: [workstation]) { _ in backend }

        var said: String?
        do {
            _ = try await RetainedMedia.hydrated(
                BatchAdmission(requests: [request()]),
                with: hydration(origin: workstation.id, hosts: hosts, members: [member()]),
                on: workstation, backend: backend)
        } catch {
            said = error.sentence
        }
        let sentence = said ?? ""
        #expect(sentence.contains("no longer has its source media"))
        #expect(sentence.contains("press Develop again"))
        #expect(!sentence.contains("retained source media is unavailable"))
        // Asked once: a settled answer would only be given again.
        #expect(backend.retainedSessionRequests.count == 1)
    }

    /// A picture someone reattached by hand wins, and the retained one is not
    /// asked for at all -- so the host's own target-conflict refusal can never
    /// fire for something this client chose to send.
    @Test func aPictureSomebodyAttachedThemselvesIsNeverOverwritten() async throws {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        var mine = request()
        mine.sourceImage = "MINE"

        let sending = try await RetainedMedia.hydrated(
            BatchAdmission(requests: [mine]),
            with: hydration(origin: workstation.id, hosts: hosts, members: [member()]),
            on: workstation, backend: backend)

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
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)

        // Nothing held yet: a chain with no retained print says nothing.
        store.warnIfTheRouteCannotCarryMedia(chained: true, outgoing: request())
        #expect(store.notice == nil)

        await store.probe([PrintID(host: workstation.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())
        #expect(store.authority != nil)
        // An ordinary render still says nothing -- it CAN carry the media.
        store.warnIfTheRouteCannotCarryMedia(chained: false, outgoing: request())
        #expect(store.notice == nil)

        store.warnIfTheRouteCannotCarryMedia(chained: true, outgoing: request())
        #expect(store.notice?.contains("renders it in pieces") == true)
    }

    /// **Fails today**: a long clip reused from a print carries nothing, and
    /// the app only says so. The chain door redeems no session, but the chain
    /// WIRE carries the bytes per stage, so the picture belongs in the well.
    @Test func aLongClipReusedFromAPrintGetsItsPictureInTheWell() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedMemberBytes["m1"] = Data([3, 1, 4])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let authority = ReuseStore.Authority(
            filename: "a.mp4", origin: workstation.id, members: [member()])

        var draft = RenderDraft()
        let wanted = RetainedSourcePicture.member(of: authority, forHydrating: request())
        #expect(wanted?.memberId == "m1")
        let fetched = await RetainedSourcePicture.fetch(
            try! #require(wanted), of: authority, hosts: hosts)
        guard case let .picture(picture) = fetched else {
            Issue.record("the picture should have been fetched")
            return
        }
        RetainedSourcePicture.place(picture, named: authority.filename, in: &draft)

        #expect(draft.media.sourceImage == Data([3, 1, 4]).base64EncodedString())
        #expect(draft.media.sourceImageName == "a.mp4")
        // The picked picture too, so a later re-fit cannot crop a crop.
        #expect(draft.media.sourceImageOriginal == draft.media.sourceImage)
    }

    /// UAT 2026-09-17 #3: the banner said "Using the source media from a.png"
    /// while the well stayed empty and there was no Strength control -- the
    /// host applied the picture at submit and nothing on screen showed it.
    ///
    /// **Fails today**: the store has no step that puts the picture in the
    /// well on an ordinary render; only the long-clip route did.
    @Test func aReusedPrintsPictureLandsInTheWellOnEveryRoute() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedMemberBytes["m1"] = Data([3, 1, 4])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)
        let draft = RenderDraft()
        await armed(store, backend: backend, host: workstation, draft: draft)
        // Said while the picture is still the host's to apply...
        #expect(store.attachmentSentence(for: draft) != nil)

        let placed = await store.placePicture(in: draft, outgoing: request())

        #expect(placed?.media.sourceImage == Data([3, 1, 4]).base64EncodedString())
        #expect(placed?.media.sourceImageName == "a.png")
        // ...and by the well, not a banner, once it is there. The authority
        // survives the store's own edit.
        #expect(store.pending(for: placed!) != nil)
        #expect(store.attachmentSentence(for: placed!) == nil)
    }

    /// A picture of the person's own is never replaced, and a well that
    /// already holds one is not a fetch.
    @Test func aWellWithAPictureInItIsLeftAlone() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)
        var draft = RenderDraft()
        draft.media.sourceImage = "mine"
        await armed(store, backend: backend, host: workstation, draft: draft)
        var mine = request()
        mine.sourceImage = "mine"

        #expect(await store.placePicture(in: draft, outgoing: mine) == nil)
        #expect(!backend.calls.contains("retainedSourceMediaBytes"))
    }

    @Test func aChainCarriesOnlyThePictureAndNeverAMaskOrAFace() {
        let authority = ReuseStore.Authority(
            filename: "a.mp4", origin: UUID(),
            members: [member("mask_image", "m1"), member("identity_image", "m2")])
        // Nothing a chain body can carry: `AutoChainRequest` has one media
        // field, and the warning path is what covers the rest.
        #expect(RetainedSourcePicture.member(of: authority, forHydrating: request()) == nil)
    }

    @Test func aChainNeverOverwritesAPictureAlreadyInTheWell() {
        let authority = ReuseStore.Authority(
            filename: "a.mp4", origin: UUID(), members: [member()])
        var mine = request()
        mine.sourceImage = "MINE"
        #expect(RetainedSourcePicture.member(of: authority, forHydrating: mine) == nil)
    }

    /// **Fails today**: the warning is gated on the authority alone, so a
    /// long clip whose source you attached BY HAND is still told to attach
    /// one. Nothing would have been hydrated -- there is nothing to say.
    @Test func aLongClipWithAPictureAlreadyAttachedIsToldNothing() async {
        let workstation = machine("workstation")
        let backend = FakeBackend(host: workstation)
        backend.retainedInventories["a.png"] = RetainedSourceMedia.Inventory(
            availability: .available, members: [member()])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ReuseStore(hosts: hosts)
        await store.probe([PrintID(host: workstation.id, filename: "a.png")],
                          fence: store.begin(), disclosing: conditioned())

        var mine = request()
        mine.sourceImage = "MINE"
        store.warnIfTheRouteCannotCarryMedia(chained: true, outgoing: mine)
        #expect(store.notice == nil)
    }
}
