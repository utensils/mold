import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor struct ReferenceNormalizationRoutingTests {
    @Test func pendingOriginalsNeverTriggerSpeculativePlacement() async {
        let host = MoldHost(name: "machine", baseURL: URL(string: "http://machine")!)
        let fake = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in fake }
        var draft = RenderDraft()
        draft.media.generationReferences = [.freshVideo(.init(data: Data([1, 2, 3]), mimeType: "video/quicktime", name: "a.mov"))]
        let probe = PlacementProbe(debounce: .zero)
        probe.refresh(draft: draft, model: "model", on: host, hosts: hosts)
        for _ in 0..<10 { await Task.yield() }
        #expect(fake.placementRequests.isEmpty)
        #expect(await probe.settle(draft: draft, model: "model", on: host, hosts: hosts) == false)
        #expect(fake.placementRequests.isEmpty)
    }

    @Test func autoFiltersForCapableHealthyModelHostAndHardSelectionDoesNotFallback() async throws {
        let old = MoldHost(name: "old", baseURL: URL(string: "http://old")!)
        let capable = MoldHost(name: "capable", baseURL: URL(string: "http://capable")!, apiKey: "key")
        let oldBackend = FakeBackend(host: old), currentBackend = FakeBackend(host: capable)
        oldBackend.modelRows = [FakeFixtures.model("flux-dev:q4")]
        currentBackend.modelRows = oldBackend.modelRows
        let hosts = HostStore(hosts: [old, capable]) { $0.id == old.id ? oldBackend : currentBackend }
        let status = try MoldJSON.decoder.decode(ServerStatus.self, from: Data(#"{"version":"test","busy":false,"uptime_secs":1}"#.utf8))
        hosts.reachability = [old.id: .up(status), capable.id: .up(status)]
        hosts.capabilities[capable.id] = try MoldJSON.decoder.decode(Capabilities.self, from: Data(Self.advertisement.utf8))
        let models = ModelStore(hosts: hosts)
        await models.refresh()
        #expect(models.referenceNormalizationHost(model: "flux-dev:q4", hardSelection: nil, adopted: old.id)?.id == capable.id)
        #expect(models.referenceNormalizationHost(model: "flux-dev:q4", hardSelection: old.id, adopted: old.id) == nil)
        #expect(models.referenceNormalizationHost(model: "flux-dev:q4", hardSelection: capable.id, adopted: old.id)?.id == capable.id)
        var missing = try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(currentBackend.modelRows[0])) as! [String: Any]
        missing["downloaded"] = false
        currentBackend.modelRows = [try MoldJSON.decoder.decode(Model.self, from: JSONSerialization.data(withJSONObject: missing))]
        await models.refresh(on: capable.id)
        #expect(models.referenceNormalizationHost(model: "flux-dev:q4", hardSelection: nil, adopted: old.id)?.id == capable.id,
                "Preparation must preserve ordinary missing-model download/retry placement")
        hosts.reachability[capable.id] = .down("offline")
        #expect(models.referenceNormalizationHost(model: "flux-dev:q4", hardSelection: nil, adopted: old.id) == nil)
    }

    static let advertisement = #"{"reference_uploads":{"available":true,"protocol_version":2,"requires_api_key":true,"session_path":"/api/reference-sessions","upload_path":"/api/reference-upload","session_handle_header":"x-session","upload_handle_header":"x-upload","max_file_bytes":4096,"max_session_bytes":8192,"max_active_sessions":4,"session_ttl_ms":60000,"video_normalization":{"available":true,"protocol_version":1,"input_roles":["reference_video"],"candidate_session_path":"/api/candidates","finalize_path":"/api/finalize","candidate_digest_header":"x-candidate","max_original_file_bytes":4096,"max_canonical_file_bytes":4096,"max_session_bytes":8192,"accepted_mime_types":["video/quicktime"],"accepted_extensions":["mov"],"accepted_video_codecs":["hevc"],"canonical_mime_type":"video/mp4"}}}"#
}
