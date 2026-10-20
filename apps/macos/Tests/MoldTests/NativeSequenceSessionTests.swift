import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor
struct NativeSequenceSessionTests {
    @Test func pendingAdmissionSurvivesRelaunchWithoutCredentialsOrMedia() throws {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let host = MoldHost(name: "Plato", baseURL: URL(string: "http://plato:7680")!, apiKey: "private")
        let record = NativeSequenceSession.Pending(id: UUID().uuidString, host: host.id,
            origin: host.baseURL.absoluteString, installation: "installation")
        let first = NativeSequenceSession(defaults: defaults)
        try first.remember(record)
        let second = NativeSequenceSession(defaults: defaults)
        #expect(second.pending == [record])
        let serialized = defaults.data(forKey: NativeSequenceSession.pendingKey)!
        #expect(!String(decoding: serialized, as: UTF8.self).contains("private"))
        second.forget(record.id)
        #expect(NativeSequenceSession(defaults: defaults).pending.isEmpty)
    }

    @Test func invalidatedCapabilitiesCannotBeRepopulatedByOldMachineReply() async throws {
        let backend = SequenceBackendFixture()
        await backend.holdLimits()
        let store = ChainLimitsStore()
        let host = UUID()
        store.refresh(host: host, model: "old", fps: 24, backend: backend)
        for _ in 0 ..< 100 where !(await backend.hasLimitsWaiter) { await Task.yield() }
        store.invalidate(host: host)
        await backend.releaseLimits()
        for _ in 0 ..< 100 { await Task.yield() }
        #expect(store.limits(host: host, model: "old", fps: 24) == nil)
    }

    @Test func switchingModelsFitsEveryClipToTheNewSupportedGrid() {
        let session = NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        session.begin(prompt: "A robot", frames: 97)
        session.draft?.stages[1].frames = 400
        session.fitLengths(minimum: 107, maximum: 345, step: 17)
        #expect(session.draft?.stages.map(\.frames) == [107, 345])
        #expect(session.message?.contains("adjusted") == true)
    }

    @Test func scalarClipDraftSurvivesRelaunch() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let session = NativeSequenceSession(defaults: defaults)
        session.begin(prompt: "A robot", frames: 107)
        session.draft?.stages[1].prompt = "The robot waves"
        let relaunched = NativeSequenceSession(defaults: defaults)
        #expect(relaunched.draft?.stages.map(\.prompt) == ["A robot", "The robot waves"])
    }
    @Test func sameJobIDOnTwoMachinesHasSeparateRecoveryAuthority() throws {
        let session = NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let one = UUID(), two = UUID()
        try session.remember(.init(id: "same-id", host: one, origin: "http://one", installation: "one"))
        try session.remember(.init(id: "same-id", host: two, origin: "http://two", installation: "two"))
        #expect(session.pending.count == 2)
        session.forget("same-id", host: one)
        #expect(session.pending.map(\.host) == [two])
    }

    @Test func startingFreshDoesNotWithdrawAcceptedWork() {
        let session = NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        session.begin(prompt: "a robot", frames: 107)
        #expect(session.draft?.stages.count == 2)
        session.startFresh()
        #expect(session.draft == nil)
    }
}

@MainActor
struct NativeSequenceAdmissionTests {
    private func settle(_ predicate: () -> Bool) async {
        for _ in 0 ..< 2_000 {
            if predicate() { return }
            await Task.yield()
        }
    }
    @Test func acceptedButLostReplyRecoversKnownIDWithoutResubmitting() async throws {
        let host = MoldHost(name: "Plato", baseURL: URL(string: "http://plato:7680")!)
        let backend = SequenceBackendFixture()
        let session = NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        session.begin(prompt: "A robot waters a garden", frames: 107)
        session.draft?.stages[1].prompt = "The robot waves"
        var request = GenerateRequest(prompt: "A robot waters a garden", model: "minimax-h3-fl2va:fasth3-v2-int8",
            width: 768, height: 512, steps: 9, guidance: 0, seed: 42)
        request.fps = 24
        session.submit(base: request, host: host, installation: "same", backend: backend, current: { true })
        await settle { session.jobs.first?.detail?.state == .completed }
        #expect(session.jobs.first?.detail?.galleryFilename == "verified.mp4")
        #expect(session.pending.isEmpty)
        #expect(await backend.creates == 1)
        if let completed = session.jobs.first { session.dismiss(completed) }
        #expect(session.jobs.isEmpty)
    }
    @Test func definitiveRefusalDoesNotLeaveAnUncertainPendingJob() async {
        let host = MoldHost(name: "Plato", baseURL: URL(string: "http://plato:7680")!)
        let backend = SequenceBackendFixture()
        await backend.reject()
        let session = NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        session.begin(prompt: "A robot", frames: 107)
        session.draft?.stages[1].prompt = "The robot waves"
        let base = GenerateRequest(prompt: "A robot", model: "minimax-h3-fl2va:fasth3-v2-int8",
            width: 768, height: 512, steps: 9, guidance: 0)
        session.submit(base: base, host: host, installation: "same", backend: backend, current: { true })
        await settle { !session.preparing }
        #expect(session.pending.isEmpty)
        #expect(session.jobs.isEmpty)
        #expect(await backend.creates == 1)
    }
    @Test func editedClipsDuringPreparationDoNotSubmitStalePlan() async {
        let host = MoldHost(name: "Plato", baseURL: URL(string: "http://plato:7680")!)
        let backend = SequenceBackendFixture()
        let session = NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        session.begin(prompt: "A robot", frames: 107)
        session.draft?.stages[1].prompt = "The robot waves"
        let base = GenerateRequest(prompt: "A robot", model: "minimax-h3-fl2va:fasth3-v2-int8",
            width: 768, height: 512, steps: 9, guidance: 0)
        session.submit(base: base, host: host, installation: "same", backend: backend, current: { true }, prepare: {
            session.draft?.stages[1].prompt = "Edited while preparing"
            return base
        })
        await settle { !session.preparing || session.validatedPlan == nil }
        for _ in 0 ..< 100 { await Task.yield() }
        #expect(await backend.creates == 0)
        #expect(session.validatedPlan == nil)
    }

    @Test func stopBeforeAcceptanceSettlesAbsentJobWithoutPollingForever() async {
        let host = MoldHost(name: "Plato", baseURL: URL(string: "http://plato:7680")!)
        let backend = SequenceBackendFixture()
        await backend.holdCreate()
        let session = NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        session.begin(prompt: "A robot", frames: 107)
        session.draft?.stages[1].prompt = "The robot waves"
        let base = GenerateRequest(prompt: "A robot", model: "minimax-h3-fl2va:fasth3-v2-int8",
            width: 768, height: 512, steps: 9, guidance: 0)
        session.submit(base: base, host: host, installation: "same", backend: backend, current: { true })
        await settle { !session.jobs.isEmpty }
        if let job = session.jobs.first { session.cancel(job, backend: backend) }
        await settle { session.jobs.isEmpty }
        #expect(session.pending.isEmpty)
        #expect(await backend.operationCancels == 1)
        await backend.releaseCreate()
        await settle { !session.preparing }
        #expect(session.jobs.isEmpty)
    }

    @Test func continuedReferencesRequireExplicitRecoveryAfterRemoval() async throws {
        let host = MoldHost(name: "Plato", baseURL: URL(string: "http://plato:7680")!)
        let backend = SequenceBackendFixture()
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let session = NativeSequenceSession(defaults: defaults)
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"verified.mp4","metadata":{"model":"minimax-h3-ref2va:comfy-pruned-int8-turbo-4step","prompt":"A robot","chain_job_id":"source"},"timestamp":1790000000,"format":"mp4"}"#.utf8))
        let entry = LibraryEntry(host: host, print: print)
        session.openContinuation(entry: entry, host: host, installation: "same", backend: backend, current: { true }, opened: { _ in
            session.hasVisibleReferences = true
        })
        await settle { !session.preparing }
        #expect(session.continuation != nil)
        #expect(await backend.privateReads == 1)
        #expect(!session.needsReferenceDecision)
        session.hasVisibleReferences = false
        #expect(session.needsReferenceDecision)
        let base = GenerateRequest(prompt: "Continue", model: "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step",
            width: 768, height: 512, steps: 4, guidance: 0)
        session.submit(base: base, host: host, installation: "same", backend: backend, current: { true })
        await settle { !session.preparing }
        #expect(await backend.creates == 0, "Alternate submit paths must honor the explicit reference decision")
        #expect(session.message?.contains("references") == true)
        session.continueWithoutReferences()
        #expect(!session.needsReferenceDecision)
        let relaunched = NativeSequenceSession(defaults: defaults)
        #expect(relaunched.restoringSource)
        #expect(!relaunched.needsReferenceDecision, "An explicit discard cannot revive old references")
    }

    @Test func changedRouteRefusesBeforeMutation() async {
        let host = MoldHost(name: "Plato", baseURL: URL(string: "http://plato:7680")!)
        let backend = SequenceBackendFixture()
        let session = NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        session.begin(prompt: "A robot", frames: 107)
        let base = GenerateRequest(prompt: "A robot", model: "minimax-h3-fl2va:fasth3-v2-int8",
            width: 768, height: 512, steps: 9, guidance: 0)
        session.submit(base: base, host: host, installation: nil, backend: backend, current: { false })
        await settle { !session.preparing }
        #expect(await backend.creates == 0)
        #expect(session.jobs.isEmpty)
        #expect(session.message?.contains("machine changed") == true)
    }
}

private actor SequenceBackendFixture: MoldChainBackend {
    var creates = 0
    var shouldReject = false
    var createsHeld = false
    var createGate: CheckedContinuation<Void, Never>?
    var operationCancels = 0
    var privateReads = 0
    var limitsHeld = false
    var limitsGate: CheckedContinuation<Void, Never>?
    var hasLimitsWaiter: Bool { limitsGate != nil }
    func holdLimits() { limitsHeld = true }
    func releaseLimits() { limitsHeld = false; limitsGate?.resume(); limitsGate = nil }
    func holdCreate() { createsHeld = true }
    func releaseCreate() { createsHeld = false; createGate?.resume(); createGate = nil }
    func cancelChainOperation(id: String, operationId: String) async throws { operationCancels += 1 }

    func reject() { shouldReject = true }
    var id = ""
    func createChainJob(_ request: AutoChainRequest, operationId: String) async throws -> CreateChainJobResponse {
        throw URLError(.unsupportedURL)
    }
    func createChainJob(_ request: AuthoredChainRequest, operationId: String) async throws -> CreateChainJobResponse {
        creates += 1; id = operationId
        if createsHeld { await withCheckedContinuation { createGate = $0 } }
        if operationCancels > 0 { throw MoldClientError.http(status: 409, code: "CHAIN_MUTATION_CANCELLED", message: "Cancelled") }
        if shouldReject { throw MoldClientError.http(status: 422, code: "REFUSED", message: "Unsupported setting") }
        throw URLError(.networkConnectionLost)
    }
    func validateChain(_ request: AuthoredChainRequest) async throws -> ChainValidationResponse {
        try MoldJSON.decoder.decode(ChainValidationResponse.self, from: Data(#"{"model":"minimax-h3-fl2va:fasth3-v2-int8","width":768,"height":512,"fps":24,"motion_tail_frames":1,"stage_count":2,"estimated_total_frames":213,"estimated_duration_ms":8875,"stages":[],"warnings":[]}"#.utf8))
    }
    func chainJob(id: String, includeSourceMedia: Bool) async throws -> ChainJobDetail {
        if !includeSourceMedia { return try await chainJob(id: id) }
        privateReads += 1
        return try MoldJSON.decoder.decode(ChainJobDetail.self, from: Data("""
        {"id":"\(id)","state":"completed","model":"minimax-h3-ref2va:comfy-pruned-int8-turbo-4step","stage_count":1,"current_stage":0,
        "finalizes":[{"gallery_filename":"verified.mp4"}],"script":{"schema":"mold.chain.v1","chain":{"model":"minimax-h3-ref2va:comfy-pruned-int8-turbo-4step","width":768,"height":512,"fps":24,"steps":4,"guidance":0,"strength":1,"motion_tail_frames":0,"output_format":"mp4","enable_audio":true},
        "stage":[{"prompt":"A robot","frames":107,"transition":"smooth","generation_references":[{"kind":"image","mime_type":"image/png","width":1,"height":1,"media":{"authority":"inline","data":"AA=="}}]}]}}
        """.utf8))
    }
    func chainJob(id: String) async throws -> ChainJobDetail {
        if operationCancels > 0 { throw MoldClientError.http(status: 404, code: "MISSING", message: "No job") }
        return try MoldJSON.decoder.decode(ChainJobDetail.self, from: Data("{\"id\":\"\(id)\",\"state\":\"completed\",\"model\":\"minimax-h3-fl2va:fasth3-v2-int8\",\"stage_count\":2,\"current_stage\":1,\"finalizes\":[{\"take\":1,\"gallery_filename\":\"verified.mp4\"}]}".utf8))
    }
    nonisolated func chainJobEvents(id: String) -> AsyncThrowingStream<ChainJobEvent, Error> { AsyncThrowingStream { $0.finish() } }
    func cancelChainJob(id: String) async throws {}
    func resumeChainJob(id: String) async throws {}
    func chainLimits(model: String, fps: Int?) async throws -> ChainLimits {
        if limitsHeld { await withCheckedContinuation { limitsGate = $0 } }
        return try MoldJSON.decoder.decode(ChainLimits.self, from: Data(#"{"model":"minimax-h3-fl2va:fasth3-v2-int8","frames_per_clip_cap":345,"frames_per_clip_recommended":107,"fps":24,"max_stages":16,"max_total_frames":5520,"supports_sequence":true,"authoring":{"motion_tail_frames":1,"carry_mode":"last-rgb-frame","min_frames":107,"frame_step":17,"frame_offset":5},"linked_parts":{"supported":true,"max_stages_per_part":16,"max_parts":null}}"#.replacingOccurrences(of: "minimax-h3-fl2va:fasth3-v2-int8", with: model).utf8))
    }
}
